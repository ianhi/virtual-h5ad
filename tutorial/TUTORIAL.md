# Virtualize a collection of h5ad files into one big matrix

You have a folder of `.h5ad` files — one per sample, all sharing the same genes.
Together they're a single dataset of millions of cells. This tutorial references
them into **one logical expression matrix `X`** inside an
[Icechunk](https://icechunk.io) repo, **without copying the array bytes**: the
repo stores byte-range pointers into the original files, and a row slice reads
only the bytes it needs.

In the example below, **11 MB of source files become a 100 KB repo**, and ~99%
of the matrix stays as references — only a handful of chunks are materialized.

> **Scope.** We virtualize the sparse `X` (the bulk of any `.h5ad`).
> Variable-length strings (`obs`/`var` indices, categoricals) live in HDF5's
> global heap and aren't byte-range addressable — out of scope here. See
> [FINDINGS.md](../FINDINGS.md).

## Setup

```bash
uv add "virtualizarr[hdf,icechunk]" icechunk anndata scipy zarr
# follow along with a real collection:
bash experiments/00_get_data.sh                  # fetch pbmc3k
uv run python experiments/02_make_collection.py  # -> data/collection/shard_*.h5ad
```

The files must share `var` (genes) and differ in `obs` (cells) — the standard
one-file-per-sample layout.

## The obstacle, and the way through it

A sparse `X` is three 1-D arrays — `data`, `indices`, `indptr`
([CSR format](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))).
Stacking samples vertically just concatenates `data` and `indices` (the column
indices don't change — same genes). The catch is chunking: every file chunks
`data` differently and ends on a partial chunk, so the naive concat fails —

```
ValueError: Cannot concatenate arrays with inconsistent chunk shapes:
(5973,) vs (5939,). Requires ZEP003 (Variable-length Chunks).
```

That used to be the end of the road. It no longer is. Zarr now supports
**rectilinear (variable-length) chunk grids**, and Icechunk stores chunk
references by coordinate regardless of grid shape. So we build the combined array
ourselves: its chunk grid is every file's HDF5 chunk edges laid end to end, and
each grid cell points at one source chunk's bytes.

One wrinkle: HDF5 pads each array's *final* chunk to full size (e.g. 5939 stored,
5918 valid). That padding can't be allowed to leak across a file boundary, so
each file's tail chunk is materialized as one small **native** chunk holding only
its valid elements. Everything else stays virtual. This is the part VirtualiZarr
doesn't do yet; it lives in the
[`virtual_h5ad`](../src/virtual_h5ad/_concat.py) package.

## Step 1 — Reference the collection as one matrix

```python
import glob
from virtual_h5ad import virtualize_matrix

paths = sorted(glob.glob("data/collection/shard_*.h5ad"))
virtualize_matrix(paths, repo_path="repo/single")
```

`virtualize_matrix` writes `data` and `indices` as single rectilinear virtual
arrays spanning every file, recomputes `indptr` with per-file row offsets (it's
tiny — written native), and tags the group as a `csr_matrix`. The one
security-relevant detail is handled for you: Icechunk is told it may read bytes
from the source directory (the *virtual chunk container*).

```bash
$ du -sh data/collection repo/single
11M   data/collection
100K  repo/single          # references + 3 tail chunks/array — no bulk copy
```

## Step 2 — Open it as one big CSR matrix

Reading rectilinear chunks is behind an experimental flag, and the reader must
authorize the same source directory the repo was built against:

```python
import zarr
zarr.config.set({"array.rectilinear_chunks": True})   # enable rectilinear grids

from virtual_h5ad import open_matrix
X = open_matrix("repo/single")
print(X.shape, X.nnz)      # (2700, 32738) 2286884
```

`X` is a `scipy.sparse.csr_matrix` identical to the original pbmc3k matrix. Slice
a row and only that row's chunks are fetched from the original `.h5ad` files —
nothing was duplicated to build the repo.

## Where to go next

- **Remote sources.** Swap the local container for `ic.s3_store(...)` and point
  source URLs at `s3://…`; the same references work over object storage. See
  [Icechunk's virtual datasets guide](https://icechunk.io/en/latest/virtual/).
- **Simpler layout.** If you don't need a single array, a group per sample
  (`experiments/03_virtual_collection.py`) avoids the rectilinear flag entirely
  and stacks at read time.
- **Carrying `obs`/`var`.** Strings aren't virtualizable, but you can load and
  write them natively into the same repo so the collection is self-describing.
