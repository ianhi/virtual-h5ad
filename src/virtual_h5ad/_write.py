"""Writing utils: build a virtual repo from a collection, and compute+store QC.

`virtualize_matrix` references each file's numeric arrays (sparse X, dense obsm)
as one rectilinear virtual matrix and writes obs/var/indptr natively.
`add_qc_metrics` shows the read-virtual -> compute -> write-native loop.
"""

from __future__ import annotations

import os.path  # commonpath/commonprefix: prefix-math, no pathlib equivalent
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import zarr
from obspec_utils.registry import ObjectStoreRegistry

from ._manifest import (
    MATRIX_GROUP,
    check_stackable,
    combined_indptr,
    inline_obsm,
    manifest_array,
    shared_obsm,
    trailing_consistent,
    write_concat_virtual_array,
)
from ._open import open_root
from ._sources import (
    download_file,
    is_remote,
    group_path,
    open_h5,
    read_h5_elem,
    local_registry,
    resolve_paths,
)


def _make_repo(repo_path, prefix: str, container_store, overwrite: bool, repo=None):
    """Get the Icechunk repo with a virtual container available for `prefix`.

    Three cases:

    - **Injected `repo`** (e.g. Arraylake): install the container **transiently** —
      re-open the same storage with the container in the open-time config and
      **never `save_config`**. Arraylake forbids persisting an `http(s)/file/memory`
      container (its `/open` 403s a repo whose config declares one), and the in-memory
      container is all `set_virtual_refs` needs to validate; readers re-supply it the
      same way (see `_ensure_virtual_container`). The refs themselves are ordinary
      committed data, so they persist fine.
    - **Fresh local repo** (`overwrite` recreates): create with the container in config.
    - **Existing local repo**: merge the container into config, reopen, and persist it
      (local repos may keep the container in config — Icechunk allows `file://`).
    """
    import shutil

    import icechunk as ic

    vcc = ic.VirtualChunkContainer(prefix, container_store)
    if repo is not None:
        cfg = repo.config  # a copy; safe to mutate — we do NOT save_config it
        cfg.set_virtual_chunk_container(vcc)
        return ic.Repository.open(
            repo.storage,
            config=cfg,
            authorize_virtual_chunk_access=ic.containers_credentials({prefix: None}),
        )
    if overwrite and Path(repo_path).is_dir():
        shutil.rmtree(repo_path)
    storage = ic.local_filesystem_storage(str(Path(repo_path).resolve()))
    if Path(repo_path).is_dir():  # append a group: merge the new container into config
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


@contextmanager
def _indexed_sources(source, download: bool, cache_dir, injected_repo: bool):
    """Resolve `source` to indexable paths + a parsing registry + a container spec.

    Yields a namespace with everything the manifest builders need: `read_paths` (what
    we index locally), `parse_urls` (``file://``-prefixed for the parser), `published`
    (the URLs/paths the repo will *reference*), `registry`, `reader_factory`, and the
    virtual-chunk-container `prefix`/`container_store`. Downloaded copies live in
    `cache_dir` (kept) or a temp dir (removed on exit). `injected_repo` requires a
    remote source so the repo stays portable. Shared by the full build and the
    additive top-up so both index sources exactly the same way.
    """
    import shutil
    import tempfile

    import icechunk as ic

    published = resolve_paths(source)  # the URLs/paths the repo will reference
    remote = is_remote(published[0])
    if injected_repo and not remote:
        raise ValueError("an injected repo needs remote source URLs to stay portable")

    # downloads go to cache_dir (kept) or a temp dir (deleted on exit)
    scratch = None
    if remote and download:
        dl_dir = cache_dir or (scratch := tempfile.mkdtemp(prefix="vh5ad_"))
        Path(dl_dir).mkdir(parents=True, exist_ok=True)
    reader_factory = None
    try:
        if remote and download:  # fetch once, index locally
            read_paths = [download_file(u, dl_dir) for u in published]
            parse_urls = [f"file://{p}" for p in read_paths]
            registry = local_registry(
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
            registry = local_registry(
                os.path.commonpath([os.path.dirname(p) for p in read_paths])
            )
            published = parse_urls

        check_stackable(read_paths)

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

        yield SimpleNamespace(
            read_paths=read_paths,
            parse_urls=parse_urls,
            published=published,
            registry=registry,
            reader_factory=reader_factory,
            prefix=prefix,
            container_store=container_store,
        )
    finally:
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)


def _matrix_helpers(store, group: str | None, src):
    """The three build primitives, bound to one repo `store` + source set `src`.

    The contract is "drop nothing": every source element is either virtualized (big
    per-cell matrices — byte-range refs) or copied native (everything else).

    - `_virt_array` — virtualize one dense/1-D array, concatenated along axis 0 (cells).
    - `_virt_sparse` — virtualize a CSR matrix: data/indices virtual, indptr native + attrs.
    - `_native` — copy a source element wholesale as native (from the first file).

    Shared by the full build (`virtualize_matrix`) and the additive top-up
    (`add_missing_elements`) so both reference/copy elements identically.
    """
    import anndata.io as aio

    def _virt_array(src_key: str, dest_key: str) -> None:
        grp, name = src_key.rsplit("/", 1)
        marrs = [
            manifest_array(u, grp, name, src.registry, src.reader_factory)
            for u in src.parse_urls
        ]
        write_concat_virtual_array(
            store, group_path(group, *dest_key.split("/")), src_key,
            marrs, src.read_paths, src.published,
        )

    def _virt_sparse(src_key: str, dest_key: str) -> None:
        for comp in ("data", "indices"):
            _virt_array(f"{src_key}/{comp}", f"{dest_key}/{comp}")
        indptr, n_var = combined_indptr(src.read_paths, matrix_key=src_key)
        gg = zarr.open_group(store, mode="r+")
        gg.create_array(
            name=group_path(group, *f"{dest_key}/indptr".split("/")),
            shape=indptr.shape, dtype=indptr.dtype, chunks=indptr.shape,
        )[:] = indptr
        gg[group_path(group, *dest_key.split("/"))].attrs.update(
            {"encoding-type": "csr_matrix", "encoding-version": "0.1.0",
             "shape": [int(indptr.size - 1), n_var]}
        )

    def _native(src_key: str, parent_rel: str, name: str) -> None:
        base0 = zarr.open_group(store, mode="r+")
        base = base0[group] if group else base0
        parent = base.require_group(parent_rel) if parent_rel else base
        aio.write_elem(parent, name, read_h5_elem(src.read_paths[0], src_key))

    return _virt_array, _virt_sparse, _native


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
    then reference the **remote** source URLs so the repo is portable. For an injected
    repo the source's virtual-chunk container is installed **transiently** (never
    persisted into the repo config), so nothing you couldn't persist (e.g. an
    Arraylake-forbidden `http(s)` prefix) lands on disk; readers re-supply it at open.

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
    if (repo_path is None) == (repo is None):
        raise ValueError("pass exactly one of repo_path (local) or repo (open repo)")

    with _indexed_sources(source, download, cache_dir, repo is not None) as src:
        read_paths, parse_urls = src.read_paths, src.parse_urls
        repo = _make_repo(
            repo_path, src.prefix, src.container_store, overwrite, repo=repo
        )
        session = repo.writable_session("main")
        store = session.store

        # Idempotent / re-runnable: if this group was already committed by a prior
        # run, drop it and rewrite cleanly (otherwise create_array hits "array
        # already exists"). A crash *before* commit leaves nothing — Icechunk
        # sessions are atomic — so this only matters when re-running after success.
        root0 = zarr.open_group(store, mode="a")
        if group is not None and group in root0:
            del root0[group]

        import h5py

        # The three build primitives (shared with the additive top-up). The contract
        # is "drop nothing": every source element is either virtualized (big per-cell
        # matrices — byte-range refs) or copied native (everything else). A
        # completeness check at the end fails if any top-level element was neither.
        _virt_array, _virt_sparse, _native = _matrix_helpers(store, group, src)

        handled: set[str] = set()  # top-level source keys we accounted for

        # X (always present, CSR)
        _virt_sparse(MATRIX_GROUP, MATRIX_GROUP)
        handled.add(MATRIX_GROUP)

        # dense obsm embeddings: virtualize as byte refs when the per-file grids line
        # up (always for a single file — a big embedding shouldn't be copied); inline
        # when files chunk the trailing axis differently (obsm is tiny next to X).
        obsm_names = shared_obsm(read_paths)
        for name in obsm_names:
            marrs = [
                manifest_array(u, "obsm", name, src.registry, src.reader_factory)
                for u in parse_urls
            ]
            if trailing_consistent(marrs):
                _virt_array(f"obsm/{name}", f"obsm/{name}")
            else:
                inline_obsm(store, group, name, read_paths)

        g = zarr.open_group(store, mode="r+")
        _write_obs_var(g, group, read_paths)  # native obs (concat) + var (shared)
        handled.update({"obs", "var", "obsm"})

        # Everything else — faithful, from the first file. `raw/X` and any `layers`
        # are big per-cell matrices (virtualize); `varm`/`obsp`/`varp`/`uns` are small
        # or global (copy native). Multi-file concat is defined for per-cell arrays
        # only; pairwise matrices across files would be ill-defined, so refuse loudly.
        multi = len(read_paths) > 1
        with open_h5(read_paths[0]) as f0:
            top = list(f0.keys())

            def _dict_attrs(container: str) -> None:
                dst = (g[group] if group else g)
                if container in dst:
                    dst[container].attrs.setdefault("encoding-type", "dict")
                    dst[container].attrs.setdefault("encoding-version", "0.1.0")

            if "obsm" in top and obsm_names:
                _dict_attrs("obsm")

            if "raw" in top:
                _virt_sparse("raw/X", "raw/X")
                _native("raw/var", "raw", "var")
                if "varm" in f0["raw"]:
                    _native("raw/varm", "raw", "varm")
                rawg = (g[group] if group else g)["raw"]
                rawg.attrs.update({"encoding-type": "raw", "encoding-version": "0.1.0"})
                handled.add("raw")

            if "layers" in top:
                lg = f0["layers"]
                for name in lg:
                    if isinstance(lg[name], h5py.Group) and lg[name].attrs.get(
                        "encoding-type"
                    ) in ("csr_matrix", "csc_matrix"):
                        _virt_sparse(f"layers/{name}", f"layers/{name}")
                    else:
                        _native(f"layers/{name}", "layers", name)
                _dict_attrs("layers")
                handled.add("layers")

            if "varm" in top:
                for name in f0["varm"]:
                    _native(f"varm/{name}", "varm", name)
                _dict_attrs("varm")
                handled.add("varm")

            for container in ("obsp", "varp"):
                if container in top:
                    if multi and len(list(f0[container])):
                        raise NotImplementedError(
                            f"{container} across multiple files isn't supported "
                            "(pairwise matrices don't concatenate along cells)."
                        )
                    for name in f0[container]:
                        _native(f"{container}/{name}", container, name)
                    _dict_attrs(container)
                    handled.add(container)

            if "uns" in top:
                _native("uns", "", "uns")  # global metadata, taken from the first file
                handled.add("uns")

            # Completeness guard: never silently drop a source element again.
            dropped = [k for k in top if k not in handled]
            if dropped:
                raise NotImplementedError(
                    f"virtualize_matrix would drop source elements {dropped}; "
                    "add handling for them (the goal is a lossless mirror)."
                )

        dsg = g[group] if group else g
        dsg.attrs["source_files"] = src.published
        dsg.attrs["obsm_keys"] = obsm_names
        extras = [k for k in ("raw", "layers", "varm", "obsp", "varp", "uns") if k in handled]
        return session.commit(
            f"virtual X{' + obsm' if obsm_names else ''}"
            + (f" + {'/'.join(extras)}" if extras else "")
            + f" over {len(src.published)} files"
            + (f" [{group}]" if group else "")
        )


def add_missing_elements(
    source,
    group: str | None = None,
    repo_path: str | None = None,
    repo=None,
    elements: tuple[str, ...] = ("raw", "uns"),
    download: bool = True,
    cache_dir: str | None = None,
) -> str | None:
    """Additively commit source elements an existing repo group is missing.

    `virtualize_matrix` rebuilds a group from scratch (it `del`s the group first), so
    it can't extend a repo you don't want to re-virtualize. This opens the *existing*
    `group` writable and commits only the requested `elements` that the source has but
    the repo lacks — every already-committed ref is left untouched (Icechunk is
    versioned; this is one commit on top). Use it to backfill `raw`/`uns` onto a group
    built by an older, lossy builder without re-referencing gigabytes of X/obsm.

    `elements` names top-level source keys to consider (default `raw` + `uns`); an
    element already present in the repo group is skipped, so the call is idempotent.
    Write target and source rules match `virtualize_matrix`: pass exactly one of
    `repo_path` (local) or `repo` (an open, e.g. Arraylake, repo — which needs remote
    source URLs). The source is indexed once (same download/registry path) to build the
    new byte-range refs. Returns the snapshot id, or ``None`` if nothing was missing.
    """
    if (repo_path is None) == (repo is None):
        raise ValueError("pass exactly one of repo_path (local) or repo (open repo)")

    with _indexed_sources(source, download, cache_dir, repo is not None) as src:
        # overwrite=False: reuse the existing repo/container, never recreate it.
        repo = _make_repo(
            repo_path, src.prefix, src.container_store, overwrite=False, repo=repo
        )
        session = repo.writable_session("main")
        store = session.store

        _virt_array, _virt_sparse, _native = _matrix_helpers(store, group, src)
        g = zarr.open_group(store, mode="r+")
        dst = g[group] if group else g

        added: list[str] = []
        with open_h5(src.read_paths[0]) as f0:
            if "raw" in elements and "raw" in f0 and "raw" not in dst:
                # big per-cell matrix → byte-range refs; var/varm copied native
                _virt_sparse("raw/X", "raw/X")
                _native("raw/var", "raw", "var")
                if "varm" in f0["raw"]:
                    _native("raw/varm", "raw", "varm")
                dst["raw"].attrs.update(
                    {"encoding-type": "raw", "encoding-version": "0.1.0"}
                )
                added.append("raw")
            if "uns" in elements and "uns" in f0 and "uns" not in dst:
                _native("uns", "", "uns")  # global metadata, from the first file
                added.append("uns")

        if not added:  # nothing missing — leave the repo untouched (no empty commit)
            return None
        return session.commit(
            f"additively add {'/'.join(added)}" + (f" [{group}]" if group else "")
        )


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
    obs = pd.concat([read_h5_elem(p, "obs") for p in paths])
    aio.write_elem(dest, "obs", obs)
    aio.write_elem(dest, "var", read_h5_elem(paths[0], "var"))


def add_qc_metrics(
    repo_path, group: str | None = None, rows=None
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
    import numpy as np

    node, session = open_root(repo_path, group, writable=True)
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
