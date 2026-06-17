"""Experiment 03 — virtualize a *collection* of h5ad files into one Icechunk repo.

Two questions:
  1. Can we concat the per-file CSR arrays into ONE virtual array? (No — shown.)
  2. What collection layout actually works, and does it reconstruct the full
     dataset? (One repo, a group per sample, all arrays virtual — yes.)

The collection is data/collection/shard_*.h5ad (built by 02_make_collection.py):
same genes (var), disjoint cells (obs) — the canonical multi-sample layout.
"""

import glob
import os

import h5py
import icechunk as ic
import xarray as xr
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from scipy import sparse
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser

DATA = os.path.abspath("data")
SHARDS = sorted(glob.glob(f"{DATA}/collection/shard_*.h5ad"))
CSR = ["data", "indices", "indptr"]


def registry():
    return ObjectStoreRegistry({f"file://{DATA}": LocalStore(prefix=DATA)})


def virtual_X(path):
    """One virtual dataset holding a file's CSR arrays, dims renamed to not collide."""
    url = f"file://{path}"
    parts = []
    for name in CSR:
        parser = HDFParser(group="X", drop_variables=[n for n in CSR if n != name])
        vds = open_virtual_dataset(url=url, registry=registry(), parser=parser)
        parts.append(vds.rename({d: f"{name}_dim" for d in vds.dims}))
    return xr.merge(parts)


def x_attrs(path):
    with h5py.File(path) as f:
        a = dict(f["X"].attrs)
    return {
        "encoding-type": str(a["encoding-type"]),
        "encoding-version": str(a.get("encoding-version", "")),
        "shape": [int(v) for v in a["shape"]],
    }


def show_concat_fails():
    print("== Q1: can we concat the CSR data arrays into one virtual array? ==")
    datas = []
    for p in SHARDS:
        parser = HDFParser(group="X", drop_variables=["indices", "indptr"])
        datas.append(
            open_virtual_dataset(url=f"file://{p}", registry=registry(), parser=parser)
        )
    try:
        xr.concat(datas, dim="phony_dim_0")
        print("  concat succeeded (unexpected)")
    except Exception as e:
        print(f"  [expected FAIL] {type(e).__name__}: {str(e)[:130]}")
        print(
            "  -> ragged/heterogeneous chunk grids can't form one regular Zarr array."
        )


def build_repo():
    print("\n== Q2: write one repo with a group per sample (all virtual) ==")
    repo_path = os.path.abspath("repo/exp03")
    config = ic.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        ic.VirtualChunkContainer(
            f"file://{DATA}/", ic.local_filesystem_store(f"{DATA}/")
        )
    )
    creds = ic.containers_credentials({f"file://{DATA}/": None})
    repo = ic.Repository.create(
        storage=ic.local_filesystem_storage(repo_path),
        config=config,
        authorize_virtual_chunk_access=creds,
    )
    session = repo.writable_session("main")
    for i, path in enumerate(SHARDS):
        grp = f"sample_{i}/X"
        virtual_X(path).vz.to_icechunk(session.store, group=grp)
        zarr.open_group(session.store, mode="r+")[grp].attrs.update(x_attrs(path))
    snap = session.commit(f"virtual collection: {len(SHARDS)} samples")
    print(f"  committed {snap}  ({len(SHARDS)} samples)")
    return repo, repo_path


def verify(repo, repo_path):
    print("\n== read back: stack all samples, compare to original pbmc3k ==")
    root = zarr.open_group(repo.readonly_session("main").store, mode="r")
    mats = []
    for i in range(len(SHARDS)):
        xg = root[f"sample_{i}/X"]
        mats.append(
            sparse.csr_matrix(
                (xg["data"][:], xg["indices"][:], xg["indptr"][:]),
                shape=tuple(xg.attrs["shape"]),
            )
        )
    big = sparse.vstack(mats).tocsr()
    import anndata as ad

    ref = ad.read_h5ad(f"{DATA}/pbmc3k_modern.h5ad").X
    print(f"  stacked shape={big.shape} nnz={big.nnz}")
    print(f"  equals original AnnData.X: {(big != ref).nnz == 0}")
    src = sum(os.path.getsize(p) for p in SHARDS)
    repo_sz = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, fs in os.walk(repo_path)
        for f in fs
    )
    print(
        f"  source files: {src / 1e6:.1f} MB   repo: {repo_sz / 1e3:.0f} KB (refs only)"
    )


def main():
    show_concat_fails()
    repo, repo_path = build_repo()
    verify(repo, repo_path)


if __name__ == "__main__":
    main()
