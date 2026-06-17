"""Experiment 05 — virtualize a real Allen Institute collection.

The Allen Institute for Immunology "Human Immune Health Atlas" (CZ CELLxGENE
collection 77f9d7e9-5675-49c3-abed-ce02f39eef1b) ships per-cell-type h5ad files
that share the same 32,357-gene panel (var) with disjoint cells (obs) — exactly
the stackable shape our tooling targets. Modern CELLxGENE format: CSR X with
int64 indices (pbmc3k used int32), gzip-chunked.

Download the two smallest by hand first (one request each; ~406 MB total):
  curl -sL https://datasets.cellxgene.cziscience.com/70dd5115-cec8-49ad-a2fb-f985067b193f.h5ad -o data/allen/other_cells.h5ad
  curl -sL https://datasets.cellxgene.cziscience.com/4f925e42-3e42-4e2d-8394-4e88bf9a0ff2.h5ad -o data/allen/dcs.h5ad

Then run this. It virtualizes the collection as one X and checks every source
file's rows reconstruct byte-for-byte.
"""

import glob
import os

import anndata as ad

from tutorial.rectilinear_concat import open_matrix, virtualize_matrix

ALLEN = os.path.abspath("data/allen")


def dir_bytes(p):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)


def main():
    paths = sorted(glob.glob(f"{ALLEN}/*.h5ad"))
    if not paths:
        raise SystemExit("no files in data/allen — see download commands in this file's docstring")
    print("collection:", [os.path.basename(p) for p in paths])

    virtualize_matrix(ALLEN, "repo/allen")
    src, repo = dir_bytes(ALLEN), dir_bytes(os.path.abspath("repo/allen"))
    print(f"  {src/1e6:.0f} MB sources -> {repo/1e3:.0f} KB repo ({src//repo}x smaller)")

    X = open_matrix("repo/allen")
    print(f"  combined X: {X.shape} nnz={X.nnz}")
    row = 0
    for p in paths:  # same (sorted) order virtualize_matrix used
        sx = ad.read_h5ad(p).X
        block = X[row : row + sx.shape[0]]
        print(f"  {os.path.basename(p):22s} rows {row}..{row + sx.shape[0]:<7} match={(block != sx).nnz == 0}")
        row += sx.shape[0]


if __name__ == "__main__":
    main()
