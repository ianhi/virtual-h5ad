"""Experiment 02a — build a realistic *collection* of h5ad files.

Real-world multi-file h5ad = many samples sharing the same genes (var) but with
different cells (obs). We synthesize that by sharding pbmc3k along the obs axis
into N files. Each shard is a valid modern-format h5ad with its own CSR X.

We then inspect the per-shard chunking of X/data, because cross-file virtual
concatenation hinges on whether those chunk grids line up.
"""

import os

import anndata as ad
import h5py
import numpy as np

SRC = "data/pbmc3k_modern.h5ad"
OUT = "data/collection"
N_SHARDS = 3


def main():
    os.makedirs(OUT, exist_ok=True)
    a = ad.read_h5ad(SRC)
    n = a.n_obs
    bounds = np.linspace(0, n, N_SHARDS + 1).astype(int)
    print(f"sharding {n} cells -> {N_SHARDS} files at {list(bounds)}")
    for i in range(N_SHARDS):
        lo, hi = bounds[i], bounds[i + 1]
        shard = a[lo:hi].copy()
        path = f"{OUT}/shard_{i}.h5ad"
        shard.write_h5ad(path, compression="gzip")
        with h5py.File(path) as f:
            d = f["X/data"]
            print(
                f"  {path}: n_obs={shard.n_obs} nnz={d.shape[0]} "
                f"chunks={d.chunks} (n_chunks={-(-d.shape[0] // d.chunks[0])})"
            )


if __name__ == "__main__":
    main()
