"""Writing utils: build a virtual repo from a collection, and compute+store QC.

`virtualize_matrix` references each file's numeric arrays (sparse X, dense obsm)
as one rectilinear virtual matrix and writes obs/var/indptr natively.
`add_qc_metrics` shows the read-virtual -> compute -> write-native loop.
"""

from __future__ import annotations

import os.path  # commonpath/commonprefix: prefix-math, no pathlib equivalent
from pathlib import Path

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
    import shutil
    import tempfile

    import icechunk as ic

    if (repo_path is None) == (repo is None):
        raise ValueError("pass exactly one of repo_path (local) or repo (open repo)")
    published = resolve_paths(source)  # the URLs/paths the repo will reference
    remote = is_remote(published[0])
    if repo is not None and not remote:
        raise ValueError("an injected repo needs remote source URLs to stay portable")

    # downloads go to cache_dir (kept) or a temp dir (deleted in finally)
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
                manifest_array(u, MATRIX_GROUP, name, registry, reader_factory)
                for u in parse_urls
            ]
            write_concat_virtual_array(
                store,
                group_path(group, MATRIX_GROUP, name),
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
        obsm_names = shared_obsm(read_paths)
        for name in obsm_names:
            marrs = [
                manifest_array(u, "obsm", name, registry, reader_factory)
                for u in parse_urls
            ]
            if trailing_consistent(marrs):
                write_concat_virtual_array(
                    store,
                    group_path(group, "obsm", name),
                    f"obsm/{name}",
                    marrs,
                    read_paths,
                    published,
                )
            else:
                inline_obsm(store, group, name, read_paths)

        indptr, n_var = combined_indptr(read_paths)
        g = zarr.open_group(store, mode="r+")
        g.create_array(
            name=group_path(group, MATRIX_GROUP, "indptr"),
            shape=indptr.shape,
            dtype=indptr.dtype,
            chunks=indptr.shape,
        )[:] = indptr
        xg = g[group_path(group, MATRIX_GROUP)]
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
