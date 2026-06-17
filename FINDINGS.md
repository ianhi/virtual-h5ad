# Virtualizing h5ad into Icechunk — findings

Running log of what works and what fights us when referencing AnnData `.h5ad`
files into [Icechunk](https://icechunk.io) via [VirtualiZarr](https://virtualizarr.readthedocs.io),
without copying the array bytes. For contributors continuing this work.

**Stack (verified installed):** virtualizarr 2.6.2 · icechunk 2.0.6 · zarr 3.2.1
· obstore 0.10.1 · anndata 0.12 · Python 3.12.

## The goal, concretely

An `.h5ad` is an HDF5 file. Its numeric arrays are already chunked byte ranges on
disk. We want an Icechunk repo that *points at* those byte ranges so the data
reads as Zarr/Xarray with no second copy — the 7.4 MB pbmc3k file should become a
~36 KB repo of references, not 7.4 MB of duplicated data.

## What works (experiment 01)

Virtualizing the sparse `X` (a `csr_matrix` group of `data` / `indices` /
`indptr`) end-to-end. See [`experiments/01_virtualize_modern.py`](experiments/01_virtualize_modern.py).

Proven on `data/pbmc3k_modern.h5ad`:

- **7.4 MB source → 36 KB Icechunk repo** — references only, no byte copy.
- **Byte-identical roundtrip** — each array read back through Icechunk equals the
  source HDF5 bytes.
- **Reconstructs to a real matrix** — `scipy.sparse.csr_matrix` rebuilt from the
  three virtual arrays equals the source `AnnData.X` exactly.

## The core friction: xarray's dimension model vs. sparse arrays

The HDF parser happily builds a per-array `ManifestStore` for the `X` group — as
a raw Zarr store the three arrays look fine:

```
/
├── data    (2286884,) float32   chunks=(8934,)
├── indices (2286884,) int32     chunks=(8934,)
└── indptr  (2701,)    int32     chunks=(2701,)
```

But `open_virtual_dataset` (and every Icechunk writer) funnels through an xarray
`Dataset`, which auto-names each array's first axis `phony_dim_0` and then refuses
the conflicting lengths (2286884 vs 2701). The three CSR arrays are independent
1-D buffers — they have no business sharing a dimension.

**Workaround that holds:** open each array as a *single-variable* dataset, rename
its `phony_dim_0` to a distinct name (`data_dim`, …), `xr.merge` the three, and
write once with `vds.vz.to_icechunk(store, group="X")`. Notes:

- Don't name the dim after the variable (`data`/`data`) — xarray then treats it as
  a coordinate and `merge` becomes ambiguous. Use a suffix.
- `to_icechunk(group=...)` *creates* the group, so you can't call it per-array
  into the same group — merge first, write once.
- Zarr dimension names are per-array metadata with no group-level length rule, so
  the distinct-length arrays coexist in the Icechunk group `X` without complaint.

## What does not virtualize

- **Whole-file / `obs` / `var` parse fails** with `'bytes' object has no
  attribute 'item'` — the parser chokes on variable-length UTF-8 string datasets
  (the dataframe index and string columns). Strings live in the HDF5 global heap,
  not a fixed-stride buffer, so they aren't byte-range references anyway.
- **Implication for the tool:** walk the file, virtualize the numeric leaf arrays
  (X components, `obsm`/`varm`, numeric obs/var columns), and handle
  strings/categoricals/scalars separately (load into memory, or write natively
  into Icechunk). AnnData semantics live in `encoding-type` attrs, which we copy
  onto the Zarr group (done for `X` → `csr_matrix`).

## Collections: many files → one repo (experiments 02/03)

The multi-sample case (same `var`, disjoint `obs`) works as **one repo with a
group per sample**, all arrays virtual. Tutorial:
[`tutorial/TUTORIAL.md`](tutorial/TUTORIAL.md). On a 3-shard pbmc3k: 11 MB
sources → 60 KB repo; stacking the samples reproduces the original `X` exactly.

**You cannot concatenate the CSR arrays into one virtual array.** Each file
chunks `data`/`indices` differently and ends on a partial chunk; a single Zarr
array needs a regular chunk grid. VirtualiZarr is explicit:

```
ValueError: Cannot concatenate arrays with inconsistent chunk shapes:
(5973,) vs (5939,). Requires ZEP003 (Variable-length Chunks).
```

So the collection stays group-per-sample and stacks at read time (the stack is
tiny; the matrix bytes stay virtual).

**Container prefix must match on read.** Icechunk gates virtual reads: the
reader must `authorize_virtual_chunk_access` for the same source-directory prefix
the repo was built against, or chunk fetches raise `IcechunkError`.

## Open questions / next

- Carry string + categorical columns into the repo (load-and-write-native path).
- Generalize the per-array merge into a recursive walk of the whole h5ad.
- Remote (`s3://`) sources via `ic.s3_store` containers.
