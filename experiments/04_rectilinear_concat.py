"""Experiment 04 — ONE logical X over the whole collection, via rectilinear chunks.

Stacking CSR matrices vertically (same genes, different cells):
  - data    -> concatenate bytes  (rectilinear virtual array)
  - indices -> concatenate bytes  (rectilinear virtual array; col indices unchanged)
  - indptr  -> recompute with per-file offsets (tiny -> write native)

The result is a single Icechunk group `X` that reads as one CSR matrix spanning
all files, with the big arrays still pointing at the original h5ad bytes.
"""

import glob
import os

import anndata as ad
import icechunk as ic
import numpy as np
import zarr
from scipy import sparse

from virtual_h5ad._concat import (
    _registry,
    manifest_arrays,
    write_concat_virtual_array,
)

DATA = os.path.abspath("data")
SHARDS = sorted(glob.glob(f"{DATA}/collection/shard_*.h5ad"))
ROOT = f"{DATA}/collection"


def combined_indptr(paths):
    """Vertically-stacked CSR indptr: per-file row pointers shifted by cumulative nnz."""
    out = [np.array([0], dtype="int64")]
    nnz = 0
    for p in paths:
        with __import__("h5py").File(p) as f:
            ip = f["X/indptr"][:]
            n_var = int(f["X"].attrs["shape"][1])
        out.append(ip[1:].astype("int64") + nnz)
        nnz += int(ip[-1])
    return np.concatenate(out), n_var, nnz


def main():
    repo_path = os.path.abspath("repo/exp04")
    config = ic.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        ic.VirtualChunkContainer(
            f"file://{ROOT}/", ic.local_filesystem_store(f"{ROOT}/")
        )
    )
    creds = ic.containers_credentials({f"file://{ROOT}/": None})
    repo = ic.Repository.create(
        storage=ic.local_filesystem_storage(repo_path),
        config=config,
        authorize_virtual_chunk_access=creds,
    )
    session = repo.writable_session("main")
    store = session.store

    print("== virtually concatenate data + indices into one rectilinear array each ==")
    reg = _registry(ROOT)
    parsed = [manifest_arrays(f"file://{p}", ("data", "indices"), reg) for p in SHARDS]
    for name in ["data", "indices"]:
        counts = write_concat_virtual_array(
            store, name, [pa[name] for pa in parsed], SHARDS
        )
        a = zarr.open_group(store, mode="r")[f"X/{name}"]
        print(
            f"  X/{name:8s} shape={a.shape} dtype={a.dtype} "
            f"virtual_chunks={counts['virtual']} native_chunks={counts['native']}"
        )

    print("\n== recompute indptr with offsets, write native ==")
    indptr, n_var, nnz = combined_indptr(SHARDS)
    n_obs = indptr.size - 1
    g = zarr.open_group(store, mode="r+")
    ip = g.create_array(
        name="X/indptr", shape=indptr.shape, dtype=indptr.dtype, chunks=indptr.shape
    )
    ip[:] = indptr
    g["X"].attrs.update(
        {
            "encoding-type": "csr_matrix",
            "encoding-version": "0.1.0",
            "shape": [n_obs, n_var],
        }
    )
    print(f"  indptr n={indptr.size} n_obs={n_obs} n_var={n_var} nnz={nnz}")

    snap = session.commit("one logical X over collection (rectilinear virtual concat)")
    print(f"  committed {snap}")

    print("\n== read the single X back as one CSR matrix ==")
    rstore = repo.readonly_session("main").store
    xg = zarr.open_group(rstore, mode="r")["X"]
    X = sparse.csr_matrix(
        (xg["data"][:], xg["indices"][:], xg["indptr"][:]),
        shape=tuple(xg.attrs["shape"]),
    )
    ref = ad.read_h5ad(f"{DATA}/pbmc3k_modern.h5ad").X
    print(f"  X shape={X.shape} nnz={X.nnz}")
    print(f"  == original pbmc3k.X: {(X != ref).nnz == 0}")


if __name__ == "__main__":
    main()
