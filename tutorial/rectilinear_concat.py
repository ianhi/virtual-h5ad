"""Virtualize a collection of h5ad files as ONE sparse matrix in Icechunk.

This module hides the virtualization details. The user-facing surface is three
functions:

    virtualize_matrix(source, repo_path)  -> build the repo (references, no copy)
    open_anndata(repo_path)               -> a normal AnnData backed by the repo
    select_cells(repo_path, obs_indices)  -> a CSR submatrix, read lazily

How it works (the part you can ignore): the files' sparse `X` arrays are stitched
into one logical array using Zarr *rectilinear* (variable-length) chunk grids, so
each chunk can point straight at a byte range in the original file. VirtualiZarr
won't concatenate ragged chunk grids itself, so we assemble the references here.
See FINDINGS.md for the full story.
"""

from __future__ import annotations

import os

# Icechunk's Rust backend logs a LocalFileSystem warning on every commit; quiet it
# so notebooks stay readable. Set before icechunk is imported; respects an override.
os.environ.setdefault("ICECHUNK_LOG", "off")

import numpy as np
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser

zarr.config.set({"array.rectilinear_chunks": True})

MATRIX_GROUP = "X"
CSR_ARRAYS = ("data", "indices", "indptr")


def _is_remote(path) -> bool:
    return str(path).startswith(("http://", "https://"))


def _name(path: str) -> str:
    return str(path).rsplit("/", 1)[-1]


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
    store = HTTPStore.from_url(base)
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


def manifest_arrays(
    url: str, names, registry: ObjectStoreRegistry, reader_factory=None
) -> dict:
    """The VirtualiZarr ManifestArrays for the named CSR components of one file.

    One parse per file: `data` and `indices` share a length (nnz), so they open
    together cleanly. (`indptr` has a different length and would collide on the
    shared phony dimension, so it is never parsed alongside them.) `reader_factory`
    lets the streaming path pass a buffered reader for fewer, larger requests.
    """
    drop = [a for a in CSR_ARRAYS if a not in names]
    kw = {"reader_factory": reader_factory} if reader_factory is not None else {}
    vds = open_virtual_dataset(
        url=url,
        registry=registry,
        parser=HDFParser(group=MATRIX_GROUP, drop_variables=drop, **kw),
    )
    return {n: vds[n].data for n in names}


def write_concat_virtual_array(
    store, name: str, marrs: list, paths: list[str], locations=None
) -> dict:
    """Write `X/<name>` = the per-file arrays concatenated on axis 0 (rectilinear).

    Full chunks become virtual references to the source bytes; each file's padded
    final chunk (HDF5 pads it to full size) is materialized as one native chunk
    holding only its valid elements, so no padding leaks across a file boundary.
    `marrs` is one ManifestArray per file; `paths` are the local files to parse.
    `locations` (one URL per file) is what each reference points at — pass the
    remote URLs when `paths` are downloaded scratch copies; defaults to the parsed
    path. Returns virtual/native chunk counts.
    """
    if locations is None:
        locations = [None] * len(paths)
    first = marrs[0]
    edges: list[int] = []
    specs = []
    natives = []  # (array_start, valid_len, source_path) for padded tails
    pos = 0
    for path, loc, ma in zip(paths, locations, marrs):
        chunk = ma.chunks[0]
        n_full, rem = divmod(int(ma.shape[0]), chunk)
        m = ma.manifest
        for i in range(n_full):  # full chunks -> virtual refs
            specs.append(
                _VChunk(
                    index=(len(edges),),
                    location=loc or str(m._paths.ravel()[i]),
                    offset=int(m._offsets.ravel()[i]),
                    length=int(m._lengths.ravel()[i]),
                )
            )
            edges.append(chunk)
            pos += chunk
        if rem:  # padded tail chunk -> materialize valid part as native
            natives.append((pos, rem, path))
            edges.append(rem)
            pos += rem

    # ManifestArray codecs = (array->bytes serializer, *compressors); split them
    # so the new array decodes the referenced source bytes identically.
    serializer, *compressors = first.metadata.codecs
    arr = zarr.open_group(store, mode="a").create_array(  # create root if absent
        name=f"{MATRIX_GROUP}/{name}",
        shape=(pos,),
        dtype=first.dtype,
        chunks=[edges],  # rectilinear: explicit per-chunk edges
        serializer=serializer,
        compressors=compressors,
        fill_value=first.metadata.fill_value,
    )
    store.set_virtual_refs(array_path=arr.path, chunks=specs, validate_containers=True)
    for start, valid, path in natives:
        with _open_h5(path) as f:
            arr[start : start + valid] = f[f"{MATRIX_GROUP}/{name}"][-valid:]
    return {"virtual": len(specs), "native": len(natives)}


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


def virtualize_matrix(
    source,
    repo_path: str,
    overwrite: bool = True,
    download: bool = True,
    cache_dir: str | None = None,
) -> str:
    """Reference a collection's X as ONE virtual CSR matrix in a new Icechunk repo.

    `source` may be a list of files/URLs, a glob (``"dir/*.h5ad"``), or a directory.
    `data`/`indices` become single rectilinear virtual arrays spanning all files;
    `indptr` and obs/var are written into the repo. Returns the snapshot id.

    For remote (`http(s)://`) sources there are two indexing strategies:
      - `download=True` (default): fetch each file once (parallel range GETs),
        index the local copy. Far faster and kinder to the server than walking the
        HDF5 chunk index over thousands of HTTP range requests.
      - `download=False`: parse directly over HTTP without downloading — slower,
        but needs no local scratch space.
    `cache_dir` keeps the downloaded copies there (reused on re-run, not deleted) —
    handy for verification; otherwise downloads go to a temp dir that is removed.
    Either way the repo references the remote URLs, so downstream reads stream only
    the cells touched. `overwrite` (default True) makes the call safe to re-run.
    """
    import shutil
    import tempfile

    import icechunk as ic

    if overwrite and os.path.isdir(repo_path):
        shutil.rmtree(repo_path)
    published = _resolve_paths(source)  # the URLs/paths the repo will reference
    remote = _is_remote(published[0])

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
                {base: SplittingReadableStore(HTTPStore.from_url(base))}
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
                ic.http_store(),
            )
        else:
            root = os.path.commonpath([os.path.dirname(p) for p in read_paths])
            prefix, container_store = (
                f"file://{root}/",
                ic.local_filesystem_store(f"{root}/"),
            )

        config = ic.RepositoryConfig.default()
        config.set_virtual_chunk_container(
            ic.VirtualChunkContainer(prefix, container_store)
        )
        repo = ic.Repository.create(
            storage=ic.local_filesystem_storage(os.path.abspath(repo_path)),
            config=config,
            authorize_virtual_chunk_access=ic.containers_credentials({prefix: None}),
        )
        session = repo.writable_session("main")
        store = session.store

        parsed = [
            manifest_arrays(u, ("data", "indices"), registry, reader_factory)
            for u in parse_urls
        ]
        for name in ("data", "indices"):
            write_concat_virtual_array(
                store, name, [pa[name] for pa in parsed], read_paths, published
            )

        indptr, n_var = _combined_indptr(read_paths)
        g = zarr.open_group(store, mode="r+")
        g.create_array(
            name=f"{MATRIX_GROUP}/indptr",
            shape=indptr.shape,
            dtype=indptr.dtype,
            chunks=indptr.shape,
        )[:] = indptr
        g[MATRIX_GROUP].attrs.update(
            {
                "encoding-type": "csr_matrix",
                "encoding-version": "0.1.0",
                "shape": [int(indptr.size - 1), n_var],
            }
        )
        _write_obs_var(
            g, read_paths
        )  # copy small metadata natively (anndata loadable_variables)
        g.attrs["source_files"] = published
        return session.commit(f"one virtual X over {len(published)} files")
    finally:
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)


def _write_obs_var(group, paths: list[str]) -> None:
    """Copy obs (concatenated) and var (shared) into the repo as native arrays.

    obs/var are tiny next to X and dominated by strings/categoricals (not
    byte-range virtualizable), so we store them outright — VirtualiZarr's
    `loadable_variables` idea. Makes the repo self-contained: opening it later
    needs no access to the source files for metadata.
    """
    import anndata.io as aio
    import pandas as pd

    obs = pd.concat([_read_elem(p, "obs") for p in paths])
    aio.write_elem(group, "obs", obs)
    aio.write_elem(group, "var", _read_elem(paths[0], "var"))


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


def _open_root(repo_path: str):
    """The repo's root Zarr group, read-only (X lives at `MATRIX_GROUP`)."""
    return zarr.open_group(
        open_repo(repo_path).readonly_session("main").store, mode="r"
    )


def _csr(x):
    """Build a scipy CSR matrix from an X group (reads the referenced bytes)."""
    from scipy import sparse

    return sparse.csr_matrix(
        (x["data"][:], x["indices"][:], x["indptr"][:]), shape=tuple(x.attrs["shape"])
    )


def open_matrix(repo_path: str):
    """Open the repo and return X as one scipy CSR matrix (rows read on demand).

    No flags or source paths to set — the module enables rectilinear chunks on
    import and `open_repo` self-authorizes from the repo's own config.
    """
    return _csr(_open_root(repo_path)[MATRIX_GROUP])


def select_cells(repo_path: str, obs_indices):
    """Read an arbitrary set of cells (rows) as a CSR submatrix, lazily.

    Because the whole collection is one logical `X`, a selection that mixes cells
    from different source files is just a row gather: `obs_indices` can span file
    boundaries freely. Only the chunks covering the requested rows are fetched
    from the source files — the rest of the matrix is never read.
    """
    from scipy import sparse

    x = _open_root(repo_path)[MATRIX_GROUP]
    indptr = x["indptr"][:]  # small (one int per cell); the bulk stays on disk
    data, indices = x["data"], x["indices"]
    rows_d, rows_i, new_indptr = [], [], [0]
    for r in obs_indices:
        s, e = int(indptr[r]), int(indptr[r + 1])
        rows_d.append(data[s:e])
        rows_i.append(indices[s:e])
        new_indptr.append(new_indptr[-1] + (e - s))
    n_var = int(x.attrs["shape"][1])
    return sparse.csr_matrix(
        (np.concatenate(rows_d), np.concatenate(rows_i), np.array(new_indptr)),
        shape=(len(obs_indices), n_var),
    )


def open_anndata(repo_path: str):
    """Open the repo as an AnnData with a lazy, virtual X over the whole collection.

    X is a backed sparse dataset: opening reads no expression data, and slicing
    (``adata[rows]``) fetches only those rows' chunks from the source files. obs
    and var were copied into the repo at build time, so opening touches no source
    files at all — it is fully local. Call ``adata.to_memory()`` to materialize X.
    """
    import anndata as ad
    import anndata.io as aio

    root = _open_root(repo_path)
    return ad.AnnData(
        X=aio.sparse_dataset(root[MATRIX_GROUP]),
        obs=aio.read_elem(root["obs"]),
        var=aio.read_elem(root["var"]),
    )
