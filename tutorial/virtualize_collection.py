"""Virtualize a collection of h5ad files into one Icechunk repo.

Given many AnnData .h5ad files that share genes (var) but hold different cells
(obs) — the usual one-file-per-sample layout — this references every file's
sparse X into a single Icechunk repository without copying the array bytes.
Each sample lands at `sample_<i>/X` as a virtual CSR matrix.

Usage:
    python virtualize_collection.py SAMPLE.h5ad [SAMPLE.h5ad ...] --repo PATH

See TUTORIAL.md for the walkthrough.
"""

from __future__ import annotations

import argparse
import os

import h5py
import icechunk as ic
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser

CSR_ARRAYS = ["data", "indices", "indptr"]


def _registry(root: str) -> ObjectStoreRegistry:
    """Authorize VirtualiZarr to read source bytes under `root`."""
    return ObjectStoreRegistry({f"file://{root}": LocalStore(prefix=root)})


def virtual_csr(path: str) -> xr.Dataset:
    """Reference one file's CSR X (data/indices/indptr) as a virtual dataset.

    The three arrays have unrelated lengths, so we open each on its own and give
    it a distinct dimension name; xarray would otherwise force them onto one
    shared 'phony_dim_0' and reject the conflicting sizes.
    """
    root = os.path.dirname(os.path.abspath(path))
    url = f"file://{os.path.abspath(path)}"
    parts = []
    for name in CSR_ARRAYS:
        parser = HDFParser(
            group="X", drop_variables=[n for n in CSR_ARRAYS if n != name]
        )
        vds = open_virtual_dataset(url=url, registry=_registry(root), parser=parser)
        parts.append(vds.rename({d: f"{name}_dim" for d in vds.dims}))
    return xr.merge(parts)


def csr_attrs(path: str) -> dict:
    """The AnnData encoding tags a reader needs to rebuild a CSR matrix."""
    with h5py.File(path) as f:
        a = dict(f["X"].attrs)
    return {
        "encoding-type": str(a["encoding-type"]),
        "encoding-version": str(a.get("encoding-version", "")),
        "shape": [int(v) for v in a["shape"]],
    }


def source_root(paths: list[str]) -> str:
    """The directory prefix that the virtual-chunk container must authorize."""
    return os.path.commonpath([os.path.dirname(os.path.abspath(p)) for p in paths])


def virtualize_collection(paths: list[str], repo_path: str) -> str:
    """Write all samples' virtual X into one Icechunk repo. Returns snapshot id."""
    import zarr

    # Every source file lives under this root; one container authorizes them all.
    root = source_root(paths)
    config = ic.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        ic.VirtualChunkContainer(
            f"file://{root}/", ic.local_filesystem_store(f"{root}/")
        )
    )
    creds = ic.containers_credentials({f"file://{root}/": None})
    repo = ic.Repository.create(
        storage=ic.local_filesystem_storage(repo_path),
        config=config,
        authorize_virtual_chunk_access=creds,
    )

    session = repo.writable_session("main")
    for i, path in enumerate(paths):
        group = f"sample_{i}/X"
        virtual_csr(path).vz.to_icechunk(session.store, group=group)
        zarr.open_group(session.store, mode="r+")[group].attrs.update(csr_attrs(path))
    return session.commit(f"virtual collection: {len(paths)} samples")


def read_collection(repo_path: str, root: str):
    """Open the repo read-only, authorizing the source files under `root`.

    Returns the root Zarr group; each sample is at `sample_<i>/X`.
    """
    import zarr

    prefix = f"file://{os.path.abspath(root)}/"
    repo = ic.Repository.open(
        storage=ic.local_filesystem_storage(os.path.abspath(repo_path)),
        authorize_virtual_chunk_access=ic.containers_credentials({prefix: None}),
    )
    return zarr.open_group(repo.readonly_session("main").store, mode="r")


def stack_csr(group):
    """Vertically stack every sample's virtual CSR X into one scipy matrix."""
    from scipy import sparse

    mats = []
    for s in sorted(group.group_keys()):
        x = group[f"{s}/X"]
        mats.append(
            sparse.csr_matrix(
                (x["data"][:], x["indices"][:], x["indptr"][:]),
                shape=tuple(x.attrs["shape"]),
            )
        )
    return sparse.vstack(mats).tocsr()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="h5ad files to virtualize")
    ap.add_argument("--repo", required=True, help="Icechunk repo path to create")
    args = ap.parse_args()
    snap = virtualize_collection(args.paths, args.repo)
    print(f"committed {len(args.paths)} samples -> {args.repo} @ {snap}")


if __name__ == "__main__":
    main()
