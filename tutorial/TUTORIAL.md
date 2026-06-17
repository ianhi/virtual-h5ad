# Virtualize a collection of h5ad files into one dataset

You have a folder of `.h5ad` files — one per sample, all sharing the same genes.
Together they're a single logical dataset of millions of cells, but they're
scattered across files and you'd rather not duplicate tens of gigabytes to query
them as one.

This tutorial references every file's expression matrix into a single
[Icechunk](https://icechunk.io) repository **without copying the array bytes**.
The repo stores byte-range pointers into the original files; reading goes
straight to the source data. In the example below, **11 MB of source files
become a 60 KB repo.**

> **Scope.** We virtualize the sparse expression matrix `X` — the bulk of any
> `.h5ad`. Variable-length strings (the `obs`/`var` indices and categorical
> columns) live in HDF5's global heap and aren't byte-range addressable, so
> they're out of scope here. See [FINDINGS.md](../FINDINGS.md) for why.

## What you need

```bash
uv add "virtualizarr[hdf,icechunk]" icechunk anndata scipy
```

A collection of modern-format `.h5ad` files that share `var` (genes) and differ
in `obs` (cells). If you want to follow along exactly, build one by sharding
pbmc3k:

```bash
bash experiments/00_get_data.sh           # fetch pbmc3k
uv run python experiments/02_make_collection.py   # -> data/collection/shard_*.h5ad
```

## Why not just concatenate everything into one array?

The obvious wish is a single virtual `X` of shape `(all_cells, genes)`. It
doesn't work, and the reason is worth understanding before we pick a layout.

A sparse `X` is stored as three 1-D arrays — `data`, `indices`, `indptr` (the
[CSR format](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))).
To virtually concatenate two files, VirtualiZarr would stitch their `data`
arrays into one Zarr array. But each file chunks `data` differently and ends on
a partial chunk:

```
shard_0  nnz=760171  chunks=(5939,)
shard_1  nnz=764451  chunks=(5973,)
shard_2  nnz=762262  chunks=(5956,)
```

Zarr requires a *regular* chunk grid — every chunk the same size but the last.
Stitching these ragged arrays needs variable-length chunks
([ZEP003](https://zarr.dev/zeps/draft/ZEP0003.html)), which isn't available yet.
VirtualiZarr says so directly:

```
ValueError: Cannot concatenate arrays with inconsistent chunk shapes:
(5973,) vs (5939,). Requires ZEP003 (Variable-length Chunks).
```

So we keep each sample as its own group and stack at read time. The matrix bytes
stay virtual; only the tiny stacking happens in memory.

## Step 1 — Reference each file's `X`

Opening a CSR `X` has one wrinkle: its three arrays have unrelated lengths
(`data` is per-nonzero, `indptr` is per-cell), but VirtualiZarr funnels each file
through an xarray `Dataset`, which tries to put them on one shared dimension and
errors. We sidestep it by opening each array on its own with a distinct
dimension name, then merging:

```python
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser
import xarray as xr

parts = []
for name in ["data", "indices", "indptr"]:
    parser = HDFParser(group="X", drop_variables=[n for n in ["data","indices","indptr"] if n != name])
    vds = open_virtual_dataset(url=f"file://{path}", registry=registry, parser=parser)
    parts.append(vds.rename({d: f"{name}_dim" for d in vds.dims}))
virtual = xr.merge(parts)
```

[`tutorial/virtualize_collection.py`](virtualize_collection.py) wraps this as
`virtual_csr(path)`.

## Step 2 — Write the whole collection to one repo

Each sample is written under its own group, `sample_<i>/X`. We copy the AnnData
`encoding-type` / `shape` attributes across so a reader knows it's a CSR matrix.
The one security-relevant step: Icechunk must be told it's allowed to read bytes
from the source directory — that's the *virtual chunk container*.

```bash
uv run python tutorial/virtualize_collection.py \
    data/collection/shard_*.h5ad --repo repo/collection
# committed 3 samples -> repo/collection @ QQTX80GYYVH5KX9H6ETG
```

```bash
$ du -sh data/collection repo/collection
11M   data/collection
60K   repo/collection          # references + metadata — no copied array bytes
```

## Step 3 — Read it back as one big matrix

The reader must authorize the same source directory it was built against (this is
Icechunk's guarantee that you can't silently read from unexpected locations):

```python
from tutorial.virtualize_collection import read_collection, stack_csr

group = read_collection("repo/collection", source_root="data/collection")
X = stack_csr(group)          # vstack every sample's virtual CSR X
print(X.shape, X.nnz)         # (2700, 32738) 2286884
```

`X[i]` pulls only that row's byte ranges from the original `.h5ad` on demand —
nothing was duplicated to build the repo. Stacking the three samples reproduces
the original pbmc3k matrix exactly.

## Where to go next

- **Remote sources.** Swap the local container for `ic.s3_store(...)` and point
  the source URLs at `s3://…`; the same virtual references work over object
  storage. See [Icechunk's virtual datasets guide](https://icechunk.io/en/latest/virtual/).
- **Carrying `obs`/`var`.** Strings aren't virtualizable, but you can load and
  write them natively into the same repo so the collection is self-describing.
- **Scale.** This pattern is `O(files)` metadata and reads nothing until you
  index, so a collection of thousands of files commits in seconds.
