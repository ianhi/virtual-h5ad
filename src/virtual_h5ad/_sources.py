"""Source inputs and small IO helpers.

Resolve h5ad inputs (files / URLs / globs / dirs), fetch remote files once for
local indexing, and do the small metadata reads (X attrs, obs/var, indptr) that
the rest of the engine needs. Nothing here knows about Icechunk or manifests.
"""

from __future__ import annotations

from pathlib import Path

from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

MATRIX_GROUP = "X"
CSR_ARRAYS = ("data", "indices", "indptr")


def is_remote(path) -> bool:
    return str(path).startswith(("http://", "https://"))


def group_path(group: str | None, *parts: str) -> str:
    """Zarr path for `parts`, under `group` if a dataset group is in use."""
    return "/".join([group, *parts]) if group else "/".join(parts)


def url_to_path(url: str) -> str:
    """Strip a ``file://`` scheme so h5py/os can open the path; leave URLs alone."""
    return url[len("file://") :] if str(url).startswith("file://") else url


def local_registry(root: str) -> ObjectStoreRegistry:
    """Local object-store registry for parsing files under `root`."""
    return ObjectStoreRegistry({f"file://{root}": LocalStore(prefix=root)})


def open_h5(path: str):
    """Open an h5ad for small metadata reads (X attrs, indptr, obs/var).

    Local files open directly; remote URLs are read with HTTP range requests via
    fsspec (used by the no-download streaming path).
    """
    import h5py

    if is_remote(path):
        import fsspec

        return h5py.File(fsspec.open(path).open())
    return h5py.File(path)


def download_file(
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

    name = Path(url).name
    # resolve to absolute: a relative path makes file://<rel> parse its first segment
    # as a host, breaking registry prefix-stripping (path doubling) downstream.
    dest = Path(dest_dir).resolve() / name
    if dest.exists():
        return str(dest)  # reuse a cached copy
    base = url.rsplit("/", 1)[0] + "/"
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
    return str(dest)


def resolve_paths(source) -> list[str]:
    """Accept HTTP URL(s), a list of files/URLs, a glob string, or a directory."""
    import glob

    if isinstance(source, (list, tuple)):
        paths = list(source)
    elif is_remote(source):
        paths = [source]
    elif Path(source).is_dir():
        paths = sorted(str(p) for p in Path(source).glob("*.h5ad"))
    else:
        paths = sorted(glob.glob(source))
    if not paths:
        raise FileNotFoundError(f"no .h5ad files matched: {source!r}")
    if is_remote(paths[0]):
        return paths
    return [str(Path(p).resolve()) for p in paths]


def read_h5_elem(path: str, key: str):
    import anndata.io as aio

    with open_h5(path) as f:
        return aio.read_elem(f[key])
