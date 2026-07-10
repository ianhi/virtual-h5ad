"""Opening utils: turn a repo (local path or an injected Icechunk repo) into a
CSR matrix or a lazy AnnData, resolving virtual chunks along the way.

Virtual reads need the source's virtual-chunk container available. Local repos
persist their container in config and self-authorize (`open_repo`). Arraylake
repos can't persist an `http(s)` container (its `/open` forbids the prefix), so
the container is installed **transiently** at read time — discovered from the
committed ``source_files`` attr — by `_ensure_virtual_container`. `obs` is read
concurrently (see `_read_obs`) since it's one native array per column.
"""

from __future__ import annotations

import os.path  # commonprefix: URL string-math, no pathlib equivalent
from pathlib import Path

import zarr

from ._sources import MATRIX_GROUP, is_remote

_OBS_READ_WORKERS = 16


def open_repo(repo_path: str):
    """Open a local Icechunk repo, self-authorizing every virtual container it declares.

    The container prefixes are stored in the repo config, so a reader needs no
    prior knowledge of where the source files live (local prefixes use no creds).
    """
    import icechunk as ic

    storage = ic.local_filesystem_storage(str(Path(repo_path).resolve()))
    prefixes = ic.Repository.open(storage=storage).config.virtual_chunk_containers or {}
    return ic.Repository.open(
        storage=storage,
        authorize_virtual_chunk_access=ic.containers_credentials(
            {p: None for p in prefixes}
        ),
    )


def _as_repo(repo_or_path):
    """An open Icechunk repo from either a local path or an already-open Repository.

    An injected Repository (e.g. from Arraylake) is used as-is; a path opens
    locally and self-authorizes from the repo config.
    """
    import icechunk as ic

    return (
        repo_or_path
        if isinstance(repo_or_path, ic.Repository)
        else open_repo(repo_or_path)
    )


def _ensure_virtual_container(repo, group: str | None, partial_values_concurrency=None):
    """Install the source's virtual-chunk container transiently if it isn't already.

    Arraylake forbids persisting `http(s)/file/memory` containers in a repo's config
    (its `/open` 403s), so those repos are built with a *clean* config and the
    container is supplied in-memory at open time. We discover the source prefix from
    the committed ``source_files`` attr (readable without any container — it's plain
    metadata), and, if it's a remote prefix not already authorized, re-open the same
    storage with the container installed transiently. Local repos already carry their
    container in config (`open_repo` authorized it), so this is a no-op for them.
    """
    import icechunk as ic

    root = zarr.open_group(repo.readonly_session("main").store, mode="r")
    node = root[group] if (group and group in root) else root
    srcs = node.attrs.get("source_files")
    if not srcs:
        return repo
    prefix = os.path.commonprefix([str(s) for s in srcs]).rsplit("/", 1)[0] + "/"
    if not is_remote(prefix):
        return repo  # local repos carry the container in config already
    if prefix in (repo.authorized_virtual_container_prefixes or set()):
        return repo  # already authorized (e.g. a self-authorized local http repo)
    cfg = repo.config  # a copy; safe to mutate — we do NOT save_config it
    cfg.set_virtual_chunk_container(
        ic.VirtualChunkContainer(prefix, ic.http_store({"allow_http": "true"}))
    )
    if partial_values_concurrency is not None:
        # how many chunk/coalesced-span fetches icechunk keeps in flight (default 10);
        # raise it so a big or multi-region selection fans out wider.
        cfg.get_partial_values_concurrency = partial_values_concurrency
    return ic.Repository.open(
        repo.storage,
        config=cfg,
        authorize_virtual_chunk_access=ic.containers_credentials({prefix: None}),
    )


def open_root(
    repo_path,
    group: str | None = None,
    writable: bool = False,
    coalesce: bool = True,
    max_gap: int = 64 * 1024,
    partial_values_concurrency: int | None = None,
):
    """The repo's dataset Zarr group (root, or the named `group`), refs resolvable.

    Reads use icechunk's native read **coalescing** by default (`coalesce=True`, via
    `coalescing_zarr`'s codec pipeline over the native icechunk store): a selection's many
    virtual-chunk fetches are merged, by real byte offset, into a few large range GETs.
    A contiguous selection (e.g. one donor, whose CSR bytes are laid out in order)
    collapses from thousands of latency-bound requests to a handful. `max_gap` is the
    largest unread gap (bytes) bridged into one GET — the 64 KiB default costs a little
    over-read for far fewer round-trips; `0` merges only strictly-adjacent chunks (zero
    over-read). If the installed icechunk lacks native coalescing, this **degrades
    gracefully** to plain reads. Pass `coalesce=False` to force the plain path.

    `partial_values_concurrency` raises icechunk's in-flight fetch count (default 10) so
    a big or multi-region selection (e.g. many donors at once) fans its coalesced spans
    out wider.
    """
    repo = _ensure_virtual_container(
        _as_repo(repo_path), group, partial_values_concurrency
    )
    if writable:
        session = repo.writable_session("main")
        store = session.store
    else:
        session = repo.readonly_session("main")
        store = session.store
        if coalesce and hasattr(store, "get_many_chunks"):
            try:
                from coalescing_zarr import register_pipeline
                from coalescing_zarr.pipeline import MAX_GAP_KEY

                # New coalescing-zarr API: the codec pipeline reads through the
                # native icechunk store's get_many_chunks — no store wrapper. The
                # pipeline snapshots max_gap from zarr config when it builds the
                # array's pipeline, so set it before open_group below.
                zarr.config.set({MAX_GAP_KEY: max_gap})
                register_pipeline()  # global; degrades to stock for non-native stores
            except ImportError:
                pass  # coalescing-zarr not installed → plain reads on session.store
    root = zarr.open_group(store, mode="r+" if writable else "r")
    node = root[group] if group else root
    return (node, session) if writable else node


def open_matrix(repo_path, group: str | None = None):
    """Open the repo and return X as one scipy CSR matrix (reads the referenced bytes)."""
    from scipy import sparse

    x = open_root(repo_path, group)[MATRIX_GROUP]
    return sparse.csr_matrix(
        (x["data"][:], x["indices"][:], x["indptr"][:]), shape=tuple(x.attrs["shape"])
    )


def open_anndata(
    repo_path,
    group: str | None = None,
    obs_columns: list[str] | None = None,
    obsm_keys: list[str] | None = None,
    coalesce: bool = True,
    max_gap: int = 64 * 1024,
    partial_values_concurrency: int | None = None,
):
    """Open the repo as an AnnData with a lazy, virtual X over the whole collection.

    X is a backed sparse dataset: opening reads no expression data, and slicing
    (``adata[rows]``) fetches only those rows' chunks from the source files. obs
    and var were copied into the repo at build time, so opening touches no source
    files for metadata. Dense `obsm` embeddings, if virtualized, are exposed lazily
    (read on access). Call ``adata.to_memory()`` to materialize X.

    By default the **full** `obs` is loaded, reading its columns concurrently (see
    `_read_obs`) so the many small reads overlap. Pass `obs_columns` (a list of
    names) only when you deliberately want a subset plus the row index. Likewise
    `obsm_keys` selects which embeddings to attach (default: all present) — pass a
    subset (e.g. ``["X_umap"]``) to avoid streaming embeddings you won't use, since
    `to_memory` materializes every attached `obsm`.

    Reads coalesce by default (see `open_root`) — a big speedup for contiguous
    selections like a donor or a small cohort of donors.
    """
    import anndata as ad
    import anndata.io as aio

    root = open_root(
        repo_path,
        group,
        coalesce=coalesce,
        max_gap=max_gap,
        partial_values_concurrency=partial_values_concurrency,
    )

    def _lazy_or_read(node):
        """Backed (lazy) sparse dataset for a CSR/CSC group; eager read otherwise."""
        if node.attrs.get("encoding-type") in ("csr_matrix", "csc_matrix"):
            return aio.sparse_dataset(node)  # keeps big matrices out of memory
        return aio.read_elem(node)

    def _mapping(name):
        return (
            {k: _lazy_or_read(root[name][k]) for k in root[name].keys()}
            if name in root
            else {}
        )

    obsm = {}
    if "obsm" in root:
        keys = obsm_keys if obsm_keys is not None else list(root["obsm"].keys())
        obsm = {k: _lazy_or_read(root["obsm"][k]) for k in keys}

    adata = ad.AnnData(
        X=aio.sparse_dataset(root[MATRIX_GROUP]),
        obs=_read_obs(root["obs"], obs_columns),
        var=aio.read_elem(root["var"]),
        obsm=obsm or None,
        layers=_mapping("layers") or None,
        varm=_mapping("varm") or None,
        obsp=_mapping("obsp") or None,
        varp=_mapping("varp") or None,
        uns=aio.read_elem(root["uns"]) if "uns" in root else None,
    )
    if "raw" in root:
        rawg = root["raw"]
        adata.raw = ad.AnnData(
            X=aio.sparse_dataset(rawg["X"]),
            var=aio.read_elem(rawg["var"]),
            varm=_mapping("raw/varm") if "varm" in rawg else None,
            obs=adata.obs,  # raw shares the cell axis; Raw keeps only X/var/varm
        )
    return adata


def _read_obs(obs_group, columns: list[str] | None = None):
    """Read `obs` as a DataFrame, reading its columns **concurrently**.

    `obs` is one native array per column (categoricals as a codes + categories pair),
    and ``anndata.read_elem`` reads them one after another — over a remote repo that
    is dozens of latency-bound round-trips on open, the dominant cost of opening. The
    reads are independent, so we resolve the child nodes (cheap; from the already-
    loaded group metadata) and pull them across a thread pool, overlapping the
    round-trips: wall time becomes ~the slowest column, not the sum. `columns`
    optionally narrows to a subset (plus the row index); default reads the whole frame.
    """
    from concurrent.futures import ThreadPoolExecutor

    import anndata.io as aio
    import pandas as pd

    attrs = dict(obs_group.attrs)
    index_key = attrs.get("_index", "_index")
    if columns is None:
        columns = [str(c) for c in attrs.get("column-order", [])] or [
            k for k in obs_group.keys() if k != index_key
        ]
    nodes = [obs_group[k] for k in (index_key, *columns)]  # resolve nodes (metadata)
    with ThreadPoolExecutor(max_workers=_OBS_READ_WORKERS) as ex:
        index, *cols = list(ex.map(aio.read_elem, nodes))
    df = pd.DataFrame(dict(zip(columns, cols)), index=pd.Index(index))
    if index_key != "_index":
        df.index.name = index_key
    return df[list(columns)]

