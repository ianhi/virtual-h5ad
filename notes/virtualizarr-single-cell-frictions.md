# VirtualiZarr frictions on large single-cell HDF5 (`.h5ad`)

*Draft for a VirtualiZarr issue. Measurements are reproducible; see the inline
scripts. Stack: virtualizarr 2.6.2 · obstore 0.10.1 · obspec-utils 0.9.0 ·
h5py 3.16 · icechunk 2.0.6 · zarr 3.2.1 · Python 3.12.*

## Context: a different shape of file than the design center

VirtualiZarr grew up in the geoscience world: **many small-to-medium NetCDF/HDF5
files**, each a handful of variables on a **regular chunk grid**, stacked along a
new (usually time) axis. The cost model is tuned for that — e.g. the default HDF
reader uses 1 MiB blocks "tuned for cloud object stores where HTTP request
overhead is significant."

Single-cell genomics (`.h5ad`, an HDF5 file) pushes the opposite regime:

- **One large, singular file** (10–40+ GB), not a stack of small ones.
- The main array is a **sparse CSR matrix** stored as three independent 1-D
  datasets (`X/data`, `X/indices`, `X/indptr`), **not** a rectangular N-D grid.
- The datasets are **chunk-dense**: `X/data` for an 815k-cell atlas has ~107k
  chunks; `X/indices` ~215k. A 3.76M-cell atlas has millions.
- Concatenating several atlases means stitching **ragged chunk grids** (each file
  chunks `X/data` differently), not appending along a clean new axis.

Two concrete frictions fall out of this. The first is a serious performance
cliff; the second is a modeling gap we currently work around by hand.

---

## Friction 1 — Building a *virtual* manifest reads ~the entire file

### Symptom

Building a virtual reference to a chunk-dense HDF5 dataset — an operation whose
entire purpose is to **not** read the data — reads essentially the whole file.
On a real 815k-cell atlas (`X/data`, 107,508 chunks; 10.3 GB file), the default
`HDFParser` fetches **8.17 GB** from the store to produce the manifest. On a
40 GB atlas the build read was measured at **62 GB** — larger than the file — and
at gp3-EBS bandwidth (~125 MB/s) that is ~8 minutes of pure indexing per file,
the dominant cost of the whole pipeline.

### It is not the parser logic — it is the read granularity

`_dataset_chunk_manifest` correctly uses `dsid.chunk_iter` / `get_chunk_info`,
which are **metadata-only** HDF5 calls: they return each chunk's `byte_offset` and
`size` from the dataset's B-tree without touching chunk data. Filter/codec
extraction is metadata-only too.

The amplification is entirely in the **I/O layer**. h5py reads the file through
obspec's `BlockStoreReader` (default **1 MiB blocks, 64-block/64 MiB LRU**). A
B-tree walk is thousands of **tiny (~2 KiB), scattered, non-sequential** reads
spread across the whole file; each lands in a distinct 1 MiB block, and the 64 MiB
cache thrashes against a 10 GB span, so nearly every data byte is fetched to
recover a few MB of offsets.

### Measured: read granularity vs. what the walk actually needs

Native h5py (HDF5's own metadata cache + byte-precise reads) walking the same
B-tree, on the real file:

| array | chunks | reads (requests) | bytes read | read size | **B-tree entries / read** |
|---|---:|---:|---:|---:|---:|
| `X/data` | 107,508 | 3,855 | **8.1 MB** | uniform ~2.0 KB | **27.9** |
| `X/indices` | 215,004 | 7,698 | **16.1 MB** | uniform ~2.0 KB | **27.9** |

Each read is a single ~2 KiB HDF5 v1 B-tree node packing ~28 chunk records. The
walk *revisits* internal nodes (every leaf lookup re-traverses the root); HDF5's
metadata cache serves revisits for free, so it needs only ~3.8k actual reads.

`HDFParser` through `BlockStoreReader`, same array (`X/data`), sweeping block size
(bytes counted at the store, cache-independent):

| reader | requests | bytes fetched | @125 MB/s |
|---|---:|---:|---:|
| **native h5py `chunk_iter`** | **3,855** | **8.1 MB** | ~0.1 s |
| `BlockStoreReader(block=8 KiB)` | 11,999 | 98 MB | ~0.8 s |
| `BlockStoreReader(block=64 KiB)` | 9,750 | 639 MB | ~5 s |
| `BlockStoreReader(block=1 MiB)` *(default)* | 7,793 | **8.17 GB** | ~65 s |
| `EagerStoreReader` | 1 | whole file (10.3 GB) | ~80 s |

Two things to note:

1. Native h5py is **below the trade-off curve on both axes** — fewer requests
   *and* far fewer bytes — because it doesn't re-fetch revisited internal nodes
   and doesn't quantize 2 KiB reads up to blocks. Tuning `block_size` only trades
   bytes for *more* requests; it never reaches native.
2. This is largely a **default-configuration pitfall** (1 MiB is wrong for a
   scattered metadata walk), but there is **no single good block size**: request
   count stays ~8–12k regardless, and the walk is a **serial, dependent**
   traversal (you can't fetch node *N+1* until node *N* tells you where it is), so
   no generic reader can parallelize it.

### Minimal reproducer

Self-contained (PEP 723 inline deps), synthesizes an 8 MB file:
[`vz_hdf_manifest_read_amplification.py`](./vz_hdf_manifest_read_amplification.py).

```
file: 8.1 MB, dataset x: 4000 chunks
manifest metadata (offsets+lengths): 64.0 KB for 4000 chunks
HDFParser fetched from store: 8.1 MB in 8 get_ranges calls
  -> 100% of the 8.1 MB file, to extract 64.0 KB of offsets
  -> 126x the metadata it needed
  manifest matches native chunk_iter: True
```

### The fix that works, and what it implies for remote

Build the manifest from a **natively-opened** h5py dataset instead of through the
obstore reader. VirtualiZarr's own `_construct_manifest_array(filepath, dataset,
group)` already does the right thing when handed a dataset from a native
`h5py.File` — it produced a **byte-identical** `ManifestArray` (same shape,
chunks, dtype, codecs `[BytesCodec, Zlib]`, fill value, offsets) reading
**26.6 MB / 56.3 MB** for data / indices in ~0.9 s, vs 10.3 GB / ~6 s.

For **remote** sources the B-tree walk is intrinsically ~11.5k serial 2 KiB reads
per file. Over HTTP that is latency-bound (~11.5k × RTT) *whatever* reader is used
— fsspec's default 5 MiB read-ahead just re-introduces the block amplification.
So the robust pattern is **download-to-local once** (bandwidth-bound, fully
parallel) **then walk natively** — not because streaming reads too many bytes, but
because the walk can't be parallelized.

### Possible upstream directions

- A **smaller / HDF-aware default `block_size`** for the HDF path, or auto-tuning
  from chunk density; at minimum a doc warning for chunk-dense files.
- An **HDF-metadata-aware reader** that reads B-tree pages at HDF5's own
  granularity (essentially what native h5py does), or a fast path that builds the
  `ChunkManifest` via `chunk_iter` on a locally-available file without routing
  metadata reads through the block reader.
- Optionally, a two-phase B-tree fetch (read internal nodes, then batch leaf nodes
  concurrently) to break the serial-round-trip wall for the remote case.

---

## Friction 2 — Ragged / rectilinear concat isn't expressible, and independent 1-D arrays fight the dimension model

Stitching several atlases into one logical `X` runs into two related walls.

### 2a. VirtualiZarr's concat requires a uniform chunk grid

Each atlas chunks `X/data` differently (HDF5 picks the chunk shape from each
file's size — e.g. 8270 vs 14213 elements). Concatenating them:

```
ValueError: Cannot concatenate arrays with inconsistent chunk shapes:
(5973,) vs (5939,). Requires ZEP003 (Variable-length Chunks).
```

But this is a *regular-grid assumption inside VirtualiZarr*, not a real wall:
zarr-python ships rectilinear (variable-length) chunk grids
(`zarr.config.set({"array.rectilinear_chunks": True})`), and Icechunk stores refs
by chunk coordinate regardless of grid. We bypass the concat and assemble the
combined array by hand: the merged grid is every file's HDF5 chunk edges laid end
to end, each full chunk becomes a virtual ref, and each file's HDF5-padded **tail**
chunk is materialized as one native chunk of just its valid elements (otherwise
padding leaks across the file boundary). This works and keeps ~99% of bytes
referenced, but it's ~200 lines of manifest surgery that arguably belongs in
VirtualiZarr as first-class **ragged concat** support.

### 2b. The xarray dimension model rejects independent 1-D arrays

The three CSR arrays are independent 1-D buffers of different lengths
(`data`/`indices` = nnz, `indptr` = n_obs+1). `open_virtual_dataset` funnels
through an xarray `Dataset`, which auto-names each array's first axis
`phony_dim_0` and then refuses the conflicting lengths:

```
data    (2286884,)  phony_dim_0
indices (2286884,)  phony_dim_0
indptr  (2701,)     phony_dim_0   <- length conflict
```

The same thing blocks reading sibling `obsm` embeddings of different width
(`X_pca` k=50 vs `X_umap` k=2) in one pass. We work around it by opening **one
array at a time** with the siblings dropped, then assembling — which also happens
to be why Friction 1 bites once per array. A CSR-/dataframe-aware path (or simply
not forcing shared dimension names on independent 1-D arrays) would remove this.

---

## Why these matter together

Both frictions are the same story: assumptions that are free in the
many-small-regular-files regime (route all reads through one block reader; every
array shares a grid and a dimension) become expensive or blocking when the file is
a single large, chunk-dense, sparse-encoded HDF5. Single-cell atlases (`.h5ad`,
CELLxGENE, 10⁵–10⁷ cells) are a large and growing corpus that would virtualize
cleanly if these were addressed; today it takes a hand-written manifest layer
(`src/virtual_h5ad/_concat.py`) to get there.
```
