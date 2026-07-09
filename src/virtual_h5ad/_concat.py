"""Virtualize a collection of h5ad files as ONE sparse matrix in Icechunk.

This module hides the virtualization details. The user-facing surface is:

    virtualize_matrix(source, repo_path)  -> build the repo (references, no copy)
    open_anndata(repo_path)               -> a normal AnnData backed by the repo
    add_qc_metrics(repo_path)             -> compute QC over the virtual X, store native

Several datasets can share one repo as named ``group``\\s (``group="allen"`` /
``group="ocular"``); open a group with ``open_anndata(repo, group="allen")``.

Slice the returned AnnData (``adata[mask].to_memory()``) to read only the cells
you select; it streams just those rows' chunks from the source files. For
scattered reads over high-latency storage, raise zarr's fan-out first:
``zarr.config.set({"async.concurrency": 128})``.

How it works (the part you can ignore): each file's numeric arrays (sparse `X`,
dense `obsm` embeddings) are stitched into one logical array using Zarr
*rectilinear* (variable-length) chunk grids, so each chunk can point straight at a
byte range in the original file. VirtualiZarr won't concatenate ragged chunk grids
itself, so we assemble the references here. String/categorical `obs`/`var` (and
computed values) are written natively into the repo. See FINDINGS.md.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser

# Icechunk's Rust backend logs a LocalFileSystem warning on every commit; quiet it
# so notebooks stay readable. Set before icechunk is imported (lazily, in functions
# below); respects an override.
os.environ.setdefault("ICECHUNK_LOG", "off")

# zarr 3 warns once per array-create that autosharding will change; we create many
# arrays per repo, which floods notebooks. The default is fine for us — quiet it.
warnings.filterwarnings("ignore", message=".*autosharding.*", category=UserWarning)

zarr.config.set({"array.rectilinear_chunks": True})

MATRIX_GROUP = "X"
CSR_ARRAYS = ("data", "indices", "indptr")


def _is_remote(path) -> bool:
    return str(path).startswith(("http://", "https://"))


def _name(path: str) -> str:
    return str(path).rsplit("/", 1)[-1]


def _prefixed(group: str | None, *parts: str) -> str:
    """Zarr path for `parts`, under `group` if a dataset group is in use."""
    return "/".join([group, *parts]) if group else "/".join(parts)


def _registry(root: str) -> ObjectStoreRegistry:
    """Local object-store registry for parsing files under `root`."""
    return ObjectStoreRegistry({f"file://{root}": LocalStore(prefix=root)})


def _open_h5(path: str):
    """Open an h5ad for small metadata reads (X attrs, indptr, obs/var).

    Local files open directly; remote URLs are read with HTTP range requests via
    fsspec (used by the no-download streaming path).
    """
    import h5py

    if _is_remote(path):
        import fsspec

        return h5py.File(fsspec.open(path).open())
    return h5py.File(path)


def _download(
    url: str, dest_dir: str, block: int = 32 * 1024 * 1024, batch: int = 16
) -> str:
    """Fetch a remote file once via parallel range requests, for local indexing.

    Indexing HDF5 means walking its chunk B-tree, whose metadata is scattered
    through the file; doing that over HTTP is thousands of latency-bound range
    requests. Pulling the whole file once — in `batch` concurrent `block`-sized
    range GETs to saturate bandwidth — then parsing locally is far faster and
    kinder to the server. The file is scratch; the repo only references `url`.
    """
    import obstore
    from obstore.store import HTTPStore

    # absolute: a relative path makes file://<rel> parse its first segment as a
    # host, which breaks registry prefix-stripping (path doubling) downstream.
    dest = os.path.abspath(os.path.join(dest_dir, _name(url)))
    if os.path.exists(dest):
        return dest  # reuse a cached copy
    base = url.rsplit("/", 1)[0] + "/"
    name = _name(url)
    store = HTTPStore.from_url(base, client_options={"allow_http": True})
    size = obstore.head(store, name)["size"]
    starts = list(range(0, size, block))
    with open(dest, "wb") as f:
        for i in range(0, len(starts), batch):  # bounded memory: batch*block in flight
            grp = starts[i : i + batch]
            parts = obstore.get_ranges(
                store, name, starts=grp, lengths=[min(block, size - s) for s in grp]
            )
            for p in parts:
                f.write(memoryview(p))
    return dest


def _manifest_array(url: str, group: str, name: str, registry, reader_factory=None):
    """The VirtualiZarr ManifestArray for a single named array in `group`.

    Local files take the native path: `_construct_manifest_array` reads only the
    HDF5 chunk B-tree (tens of MB) off an h5py-opened dataset. The `HDFParser` route
    reads through obstore's 1 MiB-block reader, which amplifies the B-tree's many
    scattered reads into a near-full-file scan — catastrophic for chunk-dense CSR
    arrays (see notes/virtualizarr-single-cell-frictions.md). Both yield a
    byte-identical ManifestArray. `filepath=url` keeps ref paths consistent (and
    `write_concat_virtual_array` overrides the location with `published` anyway).

    Remote streaming (`download=False`, `reader_factory` passed) keeps the parser
    path: it opens the array in isolation (drop its siblings) because
    `open_virtual_dataset` funnels through an xarray Dataset that auto-names each
    array's leading axis `phony_dim_0` and then refuses siblings whose other axes
    disagree (e.g. `X_pca` k=50 vs `X_umap` k=2), and its `reader_factory` buffers
    reads into fewer, larger requests. The native path needs neither trick —
    `_construct_manifest_array` takes a single dataset directly.
    """
    import h5py

    if not _is_remote(url):
        from virtualizarr.parsers.hdf.hdf import _construct_manifest_array

        with h5py.File(url_to_path(url)) as f:
            return _construct_manifest_array(
                filepath=url, dataset=f[f"{group}/{name}"], group=group
            )

    with _open_h5(url_to_path(url)) as f:
        siblings = [
            k for k in f[group] if k != name and isinstance(f[group][k], h5py.Dataset)
        ]
    kw = {"reader_factory": reader_factory} if reader_factory is not None else {}
    vds = open_virtual_dataset(
        url=url,
        registry=registry,
        parser=HDFParser(group=group, drop_variables=siblings, **kw),
    )
    return vds[name].data


def url_to_path(url: str) -> str:
    """Strip a ``file://`` scheme so h5py/os can open the path; leave URLs alone."""
    return url[len("file://") :] if str(url).startswith("file://") else url


def write_concat_virtual_array(
    store, dest_path: str, src_key: str, marrs: list, read_paths: list[str], locations
) -> dict:
    """Write `dest_path` = the per-file arrays concatenated along axis 0.

    Works for 1-D CSR components (`X/data`, `X/indices`) and N-D dense arrays
    (`obsm/X_pca`). The concatenated array is a Zarr *rectilinear* grid: axis 0's
    chunk edges are every file's HDF5 edges laid end to end; trailing axes keep the
    (shared) source grid. A chunk that is a full, non-padded chunk on **every** axis
    becomes a virtual reference to the source bytes. Any chunk that is a padded tail
    on some axis (HDF5 pads each array's final chunk per axis) is materialized as a
    native chunk holding only its valid elements — otherwise that padding would leak
    across a file (axis 0) or column (trailing axes) boundary. `src_key` is the
    array's path inside each source file (``"X/data"``, ``"obsm/X_pca"``);
    `locations` is one URL per file for the references (defaults to the parsed path).
    Returns virtual/native chunk counts.
    """
    if locations is None:
        locations = [None] * len(read_paths)
    first = marrs[0]
    ndim = len(first.shape)
    # trailing axes (everything past the concat axis) must match across files
    for ma in marrs:
        if tuple(ma.shape[1:]) != tuple(first.shape[1:]) or tuple(
            ma.chunks[1:]
        ) != tuple(first.chunks[1:]):
            raise ValueError(f"{src_key}: inconsistent trailing dims across files")

    # axis 0: walk the files, emitting one dest block per source chunk
    blocks0 = []  # {file, local0, e0, tail0}
    for fi, ma in enumerate(marrs):
        c0 = ma.chunks[0]
        n_full, rem = divmod(int(ma.shape[0]), c0)
        for j in range(n_full):
            blocks0.append({"file": fi, "local0": j, "e0": c0, "tail0": False})
        if rem:
            blocks0.append({"file": fi, "local0": n_full, "e0": rem, "tail0": True})
    edges = [[b["e0"] for b in blocks0]]

    # trailing axes: same grid for every file (checked above)
    tail_idx = {}  # axis -> index of the padded tail chunk, or None
    for a in range(1, ndim):
        ca, sa = first.chunks[a], int(first.shape[a])
        n_full, rem = divmod(sa, ca)
        edges.append([ca] * n_full + ([rem] if rem else []))
        tail_idx[a] = n_full if rem else None

    serializer, *compressors = first.metadata.codecs
    arr = zarr.open_group(store, mode="a").create_array(  # create parents if absent
        name=dest_path,
        shape=(sum(edges[0]), *[int(first.shape[a]) for a in range(1, ndim)]),
        dtype=first.dtype,
        chunks=edges,
        serializer=serializer,
        compressors=compressors,
        fill_value=first.metadata.fill_value,
    )

    # dest-offset lookup per axis (cumulative edge sums)
    offs = [np.concatenate([[0], np.cumsum(e)]) for e in edges]
    trailing_ranges = [range(len(edges[a])) for a in range(1, ndim)]
    specs, natives = [], []  # natives: (file, dest_index, src_index_in_file)
    for i0, b in enumerate(blocks0):
        ma = marrs[b["file"]]
        loc = locations[b["file"]]
        for tail in _product(trailing_ranges):
            dest_index = (i0, *tail)
            src_index = (b["local0"], *tail)
            is_tail = b["tail0"] or any(
                t == tail_idx[a + 1] for a, t in enumerate(tail)
            )
            if is_tail:
                natives.append((b["file"], dest_index, src_index))
            else:
                m = ma.manifest
                specs.append(
                    _VChunk(
                        index=dest_index,
                        location=loc or str(m._paths[src_index]),
                        offset=int(m._offsets[src_index]),
                        length=int(m._lengths[src_index]),
                    )
                )
    store.set_virtual_refs(array_path=arr.path, chunks=specs, validate_containers=True)

    # materialize native (padded-tail) chunks: read each file's valid sub-block once
    by_file: dict[int, list] = {}
    for fi, dest_index, src_index in natives:
        by_file.setdefault(fi, []).append((dest_index, src_index))
    for fi, items in by_file.items():
        ma = marrs[fi]
        with _open_h5(read_paths[fi]) as f:
            src = f[src_key]
            for dest_index, src_index in items:
                dslice = tuple(
                    slice(int(offs[a][dest_index[a]]), int(offs[a][dest_index[a] + 1]))
                    for a in range(ndim)
                )
                sslice = tuple(
                    slice(
                        src_index[a] * ma.chunks[a],
                        src_index[a] * ma.chunks[a]
                        + (dslice[a].stop - dslice[a].start),
                    )
                    for a in range(ndim)
                )
                arr[dslice] = src[sslice]
    return {"virtual": len(specs), "native": len(natives)}


def _product(ranges):
    """itertools.product over a list of ranges, yielding tuples (empty -> one ())."""
    import itertools

    return itertools.product(*ranges)


def _VChunk(index, location, offset, length):
    # Icechunk's VirtualChunkSpec is a positional-args pyo3 class; wrap for clarity.
    import icechunk as ic

    return ic.VirtualChunkSpec(
        index=list(index), location=location, offset=offset, length=length
    )


def _check_stackable(paths: list[str]) -> None:
    """Fail loudly unless every file is a CSR matrix over the same genes (var).

    Concatenating CSR `indices`/`data` by bytes assumes identical columns and CSR
    layout; otherwise the stacked result is silently wrong. Reads only the small
    X attrs + var index.
    """
    import anndata.io as aio

    ref = None
    for p in paths:
        with _open_h5(p) as f:
            if f[MATRIX_GROUP].attrs.get("encoding-type") != "csr_matrix":
                raise ValueError(f"{_name(p)}: X is not a CSR matrix.")
            names = aio.read_elem(f["var"]).index
        if ref is None:
            ref = names
        elif len(names) != len(ref) or not (names == ref).all():
            raise ValueError(
                f"{_name(p)} has a different gene set than {_name(paths[0])} — "
                "files must share genes to stack."
            )


def _shared_obsm(paths: list[str]) -> list[str]:
    """Dense `obsm` arrays present (with matching width) in *every* file.

    These are byte-range-virtualizable embeddings (e.g. `X_pca`, `X_umap`). Groups
    (sparse/dataframe obsm) and arrays missing from some file are skipped.
    """
    import h5py

    per_file = []
    for p in paths:
        with _open_h5(p) as f:
            if "obsm" not in f:
                return []
            g = f["obsm"]
            per_file.append(
                {k: g[k].shape[1:] for k in g if isinstance(g[k], h5py.Dataset)}
            )
    common = set(per_file[0])
    for d in per_file[1:]:
        common &= set(d)
    return sorted(k for k in common if all(d[k] == per_file[0][k] for d in per_file))


def _trailing_consistent(marrs: list) -> bool:
    """Whether every file chunks the obsm array's trailing axes identically.

    The rectilinear product grid needs one shared trailing-axis edge list, so a
    virtual concat is only possible when the files agree; otherwise we inline.
    """
    first = marrs[0]
    return all(
        tuple(ma.shape[1:]) == tuple(first.shape[1:])
        and tuple(ma.chunks[1:]) == tuple(first.chunks[1:])
        for ma in marrs
    )


def _inline_obsm(store, group: str | None, name: str, paths: list[str]) -> None:
    """Copy an obsm embedding into the repo as a native array (concatenated axis 0)."""
    import anndata.io as aio

    dsg = zarr.open_group(store, mode="r+")
    dest = dsg[group] if group else dsg
    if "obsm" not in dest:
        dest.create_group("obsm")
    arrs = [_read_elem(p, f"obsm/{name}") for p in paths]
    aio.write_elem(dest["obsm"], name, np.concatenate(arrs, axis=0))


def _combined_indptr(paths: list[str]):
    """Vertically-stacked CSR indptr: each file's row pointers shifted by prior nnz."""
    out = [np.array([0], dtype="int64")]
    nnz, n_var = 0, 0
    for p in paths:
        with _open_h5(p) as f:
            ip = f[f"{MATRIX_GROUP}/indptr"][:]
            n_var = int(f[MATRIX_GROUP].attrs["shape"][1])
        out.append(ip[1:].astype("int64") + nnz)
        nnz += int(ip[-1])
    return np.concatenate(out), n_var


def _resolve_paths(source) -> list[str]:
    """Accept HTTP URL(s), a list of files/URLs, a glob string, or a directory."""
    import glob

    if isinstance(source, (list, tuple)):
        paths = list(source)
    elif _is_remote(source):
        paths = [source]
    elif os.path.isdir(source):
        paths = sorted(glob.glob(os.path.join(source, "*.h5ad")))
    else:
        paths = sorted(glob.glob(source))
    if not paths:
        raise FileNotFoundError(f"no .h5ad files matched: {source!r}")
    if _is_remote(paths[0]):
        return paths
    return [os.path.abspath(p) for p in paths]


def _make_repo(repo_path, prefix: str, container_store, overwrite: bool, repo=None):
    """Get the Icechunk repo with a virtual container declared for `prefix`.

    Three cases: an externally-supplied `repo` (e.g. one handed back by Arraylake —
    we just ensure our container is in its config and persisted); a fresh local repo
    (`overwrite` recreates from scratch); or appending a group to an existing local
    repo (merge the new container into config).
    """
    import shutil

    import icechunk as ic

    vcc = ic.VirtualChunkContainer(prefix, container_store)
    if repo is not None:
        # externally-managed repo (Arraylake, S3, ...). Ensure our container is
        # configured and persisted so writes validate and future readers can resolve
        # refs. Idempotent: re-setting the same prefix just overwrites. (Writing refs
        # needs only the container present, not virtual-read authorization; a reader
        # of remote refs opens the repo authorized for `prefix` — public HTTP: none.)
        cfg = repo.config
        cfg.set_virtual_chunk_container(vcc)
        repo.save_config()
        return repo
    if overwrite and os.path.isdir(repo_path):
        shutil.rmtree(repo_path)
    storage = ic.local_filesystem_storage(os.path.abspath(repo_path))
    if os.path.isdir(repo_path):  # append a group: merge the new container into config
        cfg = ic.Repository.open(storage=storage).config
        cfg.set_virtual_chunk_container(vcc)
        creds = ic.containers_credentials(
            {p: None for p in (cfg.virtual_chunk_containers or {})}
        )
        # open WITH the merged config, then persist it so readers see every container
        repo = ic.Repository.open(
            storage=storage, config=cfg, authorize_virtual_chunk_access=creds
        )
        repo.save_config()
        return repo
    config = ic.RepositoryConfig.default()
    config.set_virtual_chunk_container(vcc)
    return ic.Repository.create(
        storage=storage,
        config=config,
        authorize_virtual_chunk_access=ic.containers_credentials({prefix: None}),
    )


def virtualize_matrix(
    source,
    repo_path: str | None = None,
    group: str | None = None,
    overwrite: bool = True,
    download: bool = True,
    cache_dir: str | None = None,
    repo=None,
) -> str:
    """Reference a collection's X (and obsm) as ONE virtual matrix in an Icechunk repo.

    `source` may be a list of files/URLs, a glob (``"dir/*.h5ad"``), or a directory.
    `X` (`data`/`indices`) and any shared dense `obsm` become single rectilinear
    virtual arrays spanning all files; `indptr` and obs/var are written into the
    repo. Pass `group` to place this dataset under a named subgroup so several
    datasets can share one repo (open with ``open_anndata(repo, group=...)``); use
    `overwrite=False` to append a group to an existing repo. Returns the snapshot id.

    Write target: pass `repo_path` for a local Icechunk repo, or `repo` for an
    already-open ``icechunk.Repository`` (e.g. one from Arraylake / an S3 store) —
    then reference the **remote** source URLs so the repo is portable. The repo
    handed in must have been opened authorized for the source prefix.

    For remote (`http(s)://`) sources there are two indexing strategies:
      - `download=True` (default): fetch each file once (parallel range GETs),
        index the local copy. Far faster and kinder to the server than walking the
        HDF5 chunk index over thousands of HTTP range requests.
      - `download=False`: parse directly over HTTP without downloading — slower,
        but needs no local scratch space.
    `cache_dir` keeps the downloaded copies there (reused on re-run, not deleted).
    Either way the repo references the remote URLs, so downstream reads stream only
    the cells touched. `overwrite` (default True) makes the call safe to re-run.
    """
    import shutil
    import tempfile

    import icechunk as ic

    if (repo_path is None) == (repo is None):
        raise ValueError("pass exactly one of repo_path (local) or repo (open repo)")
    published = _resolve_paths(source)  # the URLs/paths the repo will reference
    remote = _is_remote(published[0])
    if repo is not None and not remote:
        raise ValueError("an injected repo needs remote source URLs to stay portable")

    # downloads go to cache_dir (kept) or a temp dir (deleted in finally)
    scratch = None
    if remote and download:
        dl_dir = cache_dir or (scratch := tempfile.mkdtemp(prefix="vh5ad_"))
        os.makedirs(dl_dir, exist_ok=True)
    reader_factory = None
    try:
        if remote and download:  # fetch once, index locally
            read_paths = [_download(u, dl_dir) for u in published]
            parse_urls = [f"file://{p}" for p in read_paths]
            registry = _registry(
                os.path.commonpath([os.path.dirname(p) for p in read_paths])
            )
        elif remote:  # stream: parse over HTTP, fetched in parallel blocks
            from obspec_utils.readers import BufferedStoreReader
            from obspec_utils.splitting import SplittingReadableStore
            from obstore.store import HTTPStore

            read_paths = parse_urls = published
            base = os.path.commonprefix(published).rsplit("/", 1)[0] + "/"
            registry = ObjectStoreRegistry(
                {
                    base: SplittingReadableStore(
                        HTTPStore.from_url(base, client_options={"allow_http": True})
                    )
                }
            )
            reader_factory = BufferedStoreReader  # fewer, larger, buffered reads
        else:  # local files
            read_paths = published
            parse_urls = [f"file://{p}" for p in read_paths]
            registry = _registry(
                os.path.commonpath([os.path.dirname(p) for p in read_paths])
            )
            published = parse_urls

        _check_stackable(read_paths)

        if remote:
            prefix, container_store = (
                os.path.commonprefix(published).rsplit("/", 1)[0] + "/",
                ic.http_store(
                    {"allow_http": "true"}
                ),  # also permits plain-http (localhost)
            )
        else:
            root = os.path.commonpath([os.path.dirname(p) for p in read_paths])
            prefix, container_store = (
                f"file://{root}/",
                ic.local_filesystem_store(f"{root}/"),
            )

        repo = _make_repo(repo_path, prefix, container_store, overwrite, repo=repo)
        session = repo.writable_session("main")
        store = session.store

        # Idempotent / re-runnable: if this group was already committed by a prior
        # run, drop it and rewrite cleanly (otherwise create_array hits "array
        # already exists"). A crash *before* commit leaves nothing — Icechunk
        # sessions are atomic — so this only matters when re-running after success.
        root0 = zarr.open_group(store, mode="a")
        if group is not None and group in root0:
            del root0[group]

        # sparse X: data / indices as rectilinear virtual arrays
        for name in ("data", "indices"):
            marrs = [
                _manifest_array(u, MATRIX_GROUP, name, registry, reader_factory)
                for u in parse_urls
            ]
            write_concat_virtual_array(
                store,
                _prefixed(group, MATRIX_GROUP, name),
                f"{MATRIX_GROUP}/{name}",
                marrs,
                read_paths,
                published,
            )

        # dense obsm embeddings (X_pca, X_umap, ...): virtualize as rectilinear byte
        # refs when the per-file grids line up (always for a single file; that is
        # where it matters — a big embedding shouldn't be copied into the repo).
        # When multiple files chunk the trailing axis differently (HDF5 picks the
        # chunk shape from each file's size), fall back to inlining — obsm is tiny
        # next to X, so a native copy is cheap.
        obsm_names = _shared_obsm(read_paths)
        for name in obsm_names:
            marrs = [
                _manifest_array(u, "obsm", name, registry, reader_factory)
                for u in parse_urls
            ]
            if _trailing_consistent(marrs):
                write_concat_virtual_array(
                    store,
                    _prefixed(group, "obsm", name),
                    f"obsm/{name}",
                    marrs,
                    read_paths,
                    published,
                )
            else:
                _inline_obsm(store, group, name, read_paths)

        indptr, n_var = _combined_indptr(read_paths)
        g = zarr.open_group(store, mode="r+")
        g.create_array(
            name=_prefixed(group, MATRIX_GROUP, "indptr"),
            shape=indptr.shape,
            dtype=indptr.dtype,
            chunks=indptr.shape,
        )[:] = indptr
        xg = g[_prefixed(group, MATRIX_GROUP)]
        xg.attrs.update(
            {
                "encoding-type": "csr_matrix",
                "encoding-version": "0.1.0",
                "shape": [int(indptr.size - 1), n_var],
            }
        )
        _write_obs_var(g, group, read_paths)  # small metadata, native (self-contained)
        dsg = g[group] if group else g
        if obsm_names:
            dsg["obsm"].attrs["encoding-type"] = "dict"
        dsg.attrs["source_files"] = published
        dsg.attrs["obsm_keys"] = obsm_names
        return session.commit(
            f"virtual X{' + obsm' if obsm_names else ''} over {len(published)} files"
            + (f" [{group}]" if group else "")
        )
    finally:
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)


def _write_obs_var(group_root, group: str | None, paths: list[str]) -> None:
    """Copy obs (concatenated) and var (shared) into the repo as native arrays.

    obs/var are tiny next to X and dominated by strings/categoricals (not
    byte-range virtualizable), so we store them outright — VirtualiZarr's
    `loadable_variables` idea. Makes the repo self-contained: opening it later
    needs no access to the source files for metadata.
    """
    import anndata.io as aio
    import pandas as pd

    dest = group_root[group] if group else group_root
    obs = pd.concat([_read_elem(p, "obs") for p in paths])
    aio.write_elem(dest, "obs", obs)
    aio.write_elem(dest, "var", _read_elem(paths[0], "var"))


def _read_elem(path: str, key: str):
    import anndata.io as aio

    with _open_h5(path) as f:
        return aio.read_elem(f[key])


def open_repo(repo_path: str):
    """Open an Icechunk repo, self-authorizing every virtual container it declares.

    The container prefixes are stored in the repo config, so a reader needs no
    prior knowledge of where the source files live (local prefixes use no creds).
    """
    import icechunk as ic

    storage = ic.local_filesystem_storage(os.path.abspath(repo_path))
    prefixes = ic.Repository.open(storage=storage).config.virtual_chunk_containers or {}
    return ic.Repository.open(
        storage=storage,
        authorize_virtual_chunk_access=ic.containers_credentials(
            {p: None for p in prefixes}
        ),
    )


def _as_repo(repo_or_path):
    """An open Icechunk repo from either a local path or an already-open Repository.

    An injected Repository (e.g. from Arraylake) is used as-is — it was opened by
    the caller, authorized for its source prefixes; a path opens locally and
    self-authorizes from the repo config.
    """
    import icechunk as ic

    return (
        repo_or_path
        if isinstance(repo_or_path, ic.Repository)
        else open_repo(repo_or_path)
    )


def _open_root(repo_path, group: str | None = None, writable: bool = False):
    """The repo's dataset Zarr group (root, or the named `group`)."""
    repo = _as_repo(repo_path)
    session = (
        repo.writable_session("main") if writable else repo.readonly_session("main")
    )
    root = zarr.open_group(session.store, mode="r+" if writable else "r")
    node = root[group] if group else root
    return (node, session) if writable else node


def _csr(x):
    """Build a scipy CSR matrix from an X group (reads the referenced bytes)."""
    from scipy import sparse

    return sparse.csr_matrix(
        (x["data"][:], x["indices"][:], x["indptr"][:]), shape=tuple(x.attrs["shape"])
    )


def open_matrix(repo_path, group: str | None = None):
    """Open the repo and return X as one scipy CSR matrix (rows read on demand).

    No flags or source paths to set — the module enables rectilinear chunks on
    import and `open_repo` self-authorizes from the repo's own config.
    """
    return _csr(_open_root(repo_path, group)[MATRIX_GROUP])


def open_anndata(repo_path, group: str | None = None):
    """Open the repo as an AnnData with a lazy, virtual X over the whole collection.

    X is a backed sparse dataset: opening reads no expression data, and slicing
    (``adata[rows]``) fetches only those rows' chunks from the source files. obs
    and var were copied into the repo at build time, so opening touches no source
    files for metadata. Dense `obsm` embeddings, if virtualized, are exposed lazily
    (read on access). Call ``adata.to_memory()`` to materialize X.
    """
    import anndata as ad
    import anndata.io as aio

    root = _open_root(repo_path, group)
    obsm = {}
    if "obsm" in root:
        obsm = {k: root["obsm"][k] for k in list(root["obsm"].array_keys())}
    return ad.AnnData(
        X=aio.sparse_dataset(root[MATRIX_GROUP]),
        obs=aio.read_elem(root["obs"]),
        var=aio.read_elem(root["var"]),
        obsm=obsm or None,
    )


def add_qc_metrics(
    repo_path, group: str | None = None, rows: np.ndarray | None = None
) -> str:
    """Compute per-cell QC over the virtual X and store it back as native chunks.

    Demonstrates the read-virtual → compute → write-native loop. Two metrics:

    - ``n_genes`` — nonzero genes per cell. Derived from `indptr` alone, so it is
      computed for *every* cell without reading a single expression value.
    - ``total_counts`` — summed counts per cell. Needs the matrix, so it is read
      through the virtual X. Pass `rows` (an index array) to compute it on a subset
      only (e.g. a slice of a huge atlas); uncovered cells are left as NaN.

    The metrics are written as new native columns on the repo's `obs`, so a reader
    just sees them as ordinary AnnData columns. Returns the snapshot id.
    """
    import anndata.io as aio

    node, session = _open_root(repo_path, group, writable=True)
    indptr = node[f"{MATRIX_GROUP}/indptr"][:]
    n_genes = np.diff(indptr).astype("int32")  # free: no matrix read

    total = np.full(len(n_genes), np.nan, dtype="float64")
    x = aio.sparse_dataset(node[MATRIX_GROUP])
    if rows is None:
        total[:] = np.asarray(x[:, :].sum(axis=1)).ravel()
    else:
        rows = np.asarray(rows)
        total[rows] = np.asarray(x[rows, :].sum(axis=1)).ravel()

    obs = aio.read_elem(node["obs"])
    obs["n_genes"] = n_genes
    obs["total_counts"] = total
    del node["obs"]  # overwrite the native obs group with the augmented frame
    aio.write_elem(node, "obs", obs)
    covered = "all cells" if rows is None else f"{len(rows)} cells"
    return session.commit(f"add QC metrics (total_counts over {covered})")
