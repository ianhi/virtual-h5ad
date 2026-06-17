"""Experiment 01 — virtualize the sparse `X` of a modern h5ad into Icechunk.

Findings this probes (against a real pbmc3k file):
  - The HDF parser cannot open the whole file or even the `X` group as one
    xarray Dataset: the three CSR arrays (data/indices/indptr) get auto-named
    `phony_dim_0` and xarray refuses the conflicting lengths.
  - But each array is an independent 1-D byte buffer. Writing them one at a
    time into a shared Icechunk group sidesteps xarray's dimension coalescing,
    because Zarr dimension names are per-array with no group-level length rule.

So: virtualize X/{data,indices,indptr} as three single-variable datasets, write
all three into Icechunk group "X", then prove the byte-range refs roundtrip.
"""

import os

import h5py
import icechunk as ic
import numpy as np
import xarray as xr
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser

DATA = os.path.abspath("data")
SRC = f"{DATA}/pbmc3k_modern.h5ad"
URL = f"file://{SRC}"
SPARSE_ARRAYS = ["data", "indices", "indptr"]


def registry():
    return ObjectStoreRegistry({f"file://{DATA}": LocalStore(prefix=DATA)})


def virtual_array(group, name, all_names):
    """Virtual dataset holding exactly one array from `group`, dim renamed to it."""
    drop = [n for n in all_names if n != name]
    parser = HDFParser(group=group, drop_variables=drop)
    vds = open_virtual_dataset(url=URL, registry=registry(), parser=parser)
    # rename auto 'phony_dim_0' to a distinct (non-coordinate) dim per array so the
    # three can merge into one Dataset without colliding or becoming coordinates
    return vds.rename({d: f"{name}_dim" for d in vds.dims})


def main():
    print("== build per-array virtual datasets, merge into one X dataset ==")
    vdss = [virtual_array("X", n, SPARSE_ARRAYS) for n in SPARSE_ARRAYS]
    vds_X = xr.merge(vdss)  # distinct dim names => no phony_dim conflict
    for n in SPARSE_ARRAYS:
        v = vds_X[n]
        print(f"  X/{n:8s} shape={v.shape} dtype={v.dtype} dims={v.dims}")

    print("\n== write virtual refs into Icechunk group 'X' ==")
    repo_path = os.path.abspath("repo/exp01")
    config = ic.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        ic.VirtualChunkContainer(f"file://{DATA}/", ic.local_filesystem_store(f"{DATA}/"))
    )
    creds = ic.containers_credentials({f"file://{DATA}/": None})
    repo = ic.Repository.create(
        storage=ic.local_filesystem_storage(repo_path),
        config=config,
        authorize_virtual_chunk_access=creds,
    )
    session = repo.writable_session("main")
    vds_X.vz.to_icechunk(session.store, group="X")
    # carry the AnnData encoding so a reader knows this group is a CSR matrix
    with h5py.File(SRC) as f:
        x_attrs = dict(f["X"].attrs)
    zarr.open_group(session.store, mode="r+")["X"].attrs.update(
        {
            "encoding-type": str(x_attrs["encoding-type"]),
            "encoding-version": str(x_attrs.get("encoding-version", "")),
            "shape": [int(v) for v in x_attrs["shape"]],
        }
    )
    snap = session.commit("virtual CSR X refs from pbmc3k_modern.h5ad")
    print(f"  committed {snap}")

    print("\n== read back through Icechunk; compare to source bytes ==")
    rstore = repo.readonly_session("main").store
    xg = zarr.open_group(rstore, mode="r")["X"]
    print(f"  X attrs: {dict(xg.attrs)}")
    with h5py.File(SRC) as f:
        for n in SPARSE_ARRAYS:
            got, want = xg[n][:], f[f"X/{n}"][:]
            print(f"  X/{n:8s} match={np.array_equal(got, want)} (n={got.size}, {got.dtype})")


if __name__ == "__main__":
    main()
