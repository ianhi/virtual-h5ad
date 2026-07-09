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

### Two layouts

**(a) Group per sample** (experiment 03) — simplest; stack at read time.
VirtualiZarr's own concat refuses to merge the CSR arrays:

```
ValueError: Cannot concatenate arrays with inconsistent chunk shapes:
(5973,) vs (5939,). Requires ZEP003 (Variable-length Chunks).
```

**(b) One logical `X` via rectilinear chunks** (experiment 04) — the better
result. That ZEP003 limit is *VirtualiZarr's regular-grid assumption, not a real
wall*: Zarr now ships rectilinear (variable-length) chunk grids
([flag](https://github.com/zarr-developers/zarr-python): `array.rectilinear_chunks`),
and Icechunk stores refs by chunk coordinate regardless of grid. So we bypass
VirtualiZarr's concat and assemble the array ourselves
([`src/virtual_h5ad/_concat.py`](src/virtual_h5ad/_concat.py)):

- Combined chunk grid = every file's HDF5 chunk edges laid end to end.
- Each full chunk → a virtual reference to the source bytes.
- HDF5 pads every array's **final** chunk to full size (5939 stored, 5918
  valid). That padding would leak across a file boundary, so each file's tail
  chunk is materialized as one **native** chunk holding only its valid elements.
- `data`/`indices` concatenate by bytes (column indices are unchanged — same
  genes); `indptr` is recomputed with per-file offsets and written native.

Result on 3-shard pbmc3k: a single `X` (2700×32738), **381 virtual + 3 native
chunks** per array (~99% of bytes stay referenced), 11 MB → 100 KB repo,
reconstructs the original matrix exactly, and a lazy row slice reads only the
bytes it needs. **Readers must enable** `zarr.config.set({'array.rectilinear_chunks': True})`.

**Container prefix must match on read.** Icechunk gates virtual reads: the
reader must `authorize_virtual_chunk_access` for the same source-directory prefix
the repo was built against, or chunk fetches raise `IcechunkError`.

## Real data: Allen Institute Human Immune Health Atlas (experiment 05)

Validated on actual CELLxGENE files (collection
`77f9d7e9-5675-49c3-abed-ce02f39eef1b`), not just sharded pbmc3k. Two cell-type
files (dcs + other_cells) sharing the 32,357-gene panel:

- **406 MB sources → ~3.5 MB repo (~116× smaller)**, one logical `X`
  (34,224 × 32,357, 79M nonzeros), virtualized in ~1 s. (The repo is larger than
  the ~410 KB of the pure-`X` era because `obs`/`var` — the full CELLxGENE cell and
  gene metadata — are now carried natively so the repo is self-contained; `X`
  itself is still ~99% references.)
- Every source file's rows reconstruct **byte-identically**.
- These files use **int64** `indices`/`indptr` (pbmc3k used int32) and gzip
  (`deflate` level 4) — the tool reads dtype/codecs/chunk sizes per array, so it
  adapts with no changes. Files have different `data` chunk sizes (8270 vs
  14213); the rectilinear grid stitches them anyway.

Gotcha: `virtualize_matrix` orders samples by sorted path; verify against the
same order (sorted glob put `dcs` before `other_cells`).

## Dense obsm embeddings (X_pca / X_umap / X_scVI)

The concat engine was generalized from 1-D (CSR components) to **N-D**, so dense
`obsm` embeddings virtualize the same way `X` does. The rule that keeps roundtrips
byte-identical: a source chunk becomes a virtual reference only if it is a full,
non-padded chunk on **every** axis; a chunk that is a padded tail on *any* axis
(HDF5 pads each array's final chunk per axis) is materialized as one native chunk
of just its valid elements. For a 2-D embedding that means the trailing-column tail
of each row-block is native, plus the usual axis-0 file tails.

Caveat: the rectilinear grid is a *product* of per-axis edge lists, so every file
must chunk the trailing axes identically. A single file always qualifies (this is
where it matters — ocular's 98 MB `X_scVI` shouldn't be copied into the repo). When
multiple files disagree (HDF5 picks the chunk shape from each file's size — the two
Allen files chunk `X_pca` differently on axis 1), we **inline** the embedding
instead; it is tiny next to `X`. See `_trailing_consistent` / `_inline_obsm`.

## Many datasets in one repo (groups)

`virtualize_matrix(..., group="allen")` writes a dataset under a named subgroup;
`overwrite=False` on the next call **appends** another group to the same repo.
Byte-range containers are per-source-prefix and live in the repo config, so
appending a group with a new source location means adding its container to the
config and **persisting it** — `save_config()` saves the config the repo was
*opened with*, so you must reopen with the merged config, then save (otherwise the
new container is silently dropped and reads fail with "no virtual chunk container
can handle the chunk location"). `open_anndata(repo, group="allen")` opens one.

Validated: Allen (2 files) as `/allen` + the 10 GB ocular atlas as `/ocular` in a
single **50 MB** repo (10.7 GB of sources, ~213× smaller); both open as ordinary
AnnData, X and obsm byte-identical.

## Computing values and storing them native

`add_qc_metrics` demonstrates the read-virtual → compute → write-native loop.
`n_genes` (nonzero genes per cell) comes straight from the CSR `indptr`, so it is
computed for every cell **without reading a single expression value**;
`total_counts` reads through the virtual X (optionally on a row subset). Both are
written back as native `obs` columns, so a reader just sees ordinary AnnData
columns — no bespoke API.

## Toward a cloud repo (e.g. Arraylake)

Two locations, only one of which needs to change:

- **Source bytes** (what the virtual refs point at). CELLxGENE serves its files
  over HTTPS (`https://datasets.cellxgene.cziscience.com/<id>.h5ad`), and the HTTP
  virtual container already resolves those from anywhere. So a repo built against
  the HTTPS URLs is portable as-is — no `s3://` source container needed. (Pass the
  URLs to `virtualize_matrix`; `download=True` fetches them once to index, but the
  repo still *references* the URLs.)
- **Repo storage** (the refs + native chunks + metadata). This is the only piece
  that changes for a cloud store: swap `ic.local_filesystem_storage` for the
  cloud/Arraylake-backed storage. The source references are unaffected.

## Read amplification: HDFParser reads the whole file just to index it

Indexing an h5ad meant one `open_virtual_dataset` + `HDFParser` call per array,
which reads the HDF5 file through obstore's `BlockStoreReader` in 1 MiB blocks.
Building a virtual manifest only needs the chunk B-tree (tens of MB), but the
B-tree's entries are scattered through the file, so each tiny read is amplified to
a full 1 MiB block. For chunk-dense single-cell CSR arrays this is catastrophic:
the 815k-cell ocular file has `X/data` ≈ 107k chunks and `X/indices` ≈ 215k chunks,
so the walk reads ~100% of the file — 10.3 GB to index a 10 GB file (62 GB on a
40 GB file) — which dominated the whole pipeline's cost.

Fix: for a **local** source, build the `ManifestArray` natively via VirtualiZarr's
own `_construct_manifest_array` on an h5py-opened dataset, bypassing obstore
entirely. It reads only the B-tree and yields a **byte-identical** ManifestArray
(same shape/chunks/dtype/codecs/fill-value/offsets). Measured on the ocular file's
`X` (fresh process): **33.7 MB** for `data` and **67 MB** for `indices`, vs ~5 GB
each through the parser. The remote streaming path (`download=False`) keeps the
`HDFParser` route, which uses a buffered reader for its HTTP requests. Only local
files (including the `download=True` fetch-then-index case) take the native path.
Full analysis: `notes/virtualizarr-single-cell-frictions.md`.

## Open questions / next

- Point repo storage at Arraylake (a small change to `_make_repo` /
  `virtualize_matrix`: accept an Icechunk storage or Arraylake repo instead of a
  local path). Build the demo repo against the HTTPS source URLs so it is
  location-independent.
- Generalize the per-array merge into a recursive walk of the whole h5ad
  (`layers`, `varm`, `obsp` — empty in the CELLxGENE files tested so far).
