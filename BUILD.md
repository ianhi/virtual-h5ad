# What we had to build

Reference for contributors: the line between what existing libraries gave us and
what this repo adds. The short version — **VirtualiZarr virtualizes one HDF5 file;
we built the layer that turns a collection of them into a single logical sparse
matrix that any ordinary AnnData read can slice.** All custom code lives in
[`src/virtual_h5ad/_concat.py`](src/virtual_h5ad/_concat.py).

See [FINDINGS.md](FINDINGS.md) for the experiment-by-experiment log that got us
here; this doc is the consolidated build-vs-buy picture.

## Bought (the heavy lifting we did *not* write)

| Capability | Library | How we use it |
|---|---|---|
| HDF5 chunk B-tree → byte offsets/lengths + codecs | **VirtualiZarr** `HDFParser` / `open_virtual_dataset` | one parse per file → a `ManifestArray` per CSR array (`manifest_arrays`, [`_concat.py:105`](src/virtual_h5ad/_concat.py#L105)) |
| Virtual-chunk storage & gated reads | **Icechunk** `VirtualChunkContainer` / `VirtualChunkSpec` / `set_virtual_refs` | we hand it the per-chunk references; it stores and serves them |
| Variable-length chunk grids | **zarr 3** `array.rectilinear_chunks` | the grid type that lets one array hold differently-sized chunks |
| Parallel range GETs; HTTP/local stores | **obstore** | bulk download of source files for local indexing (`_download`) |
| CSR decode, `obs`/`var` IO, backed sparse | **scipy** / **anndata** | reconstruct the matrix; `sparse_dataset` backs the lazy `X` |

The hard, format-specific part — walking HDF5's on-disk chunk index to get exact
byte ranges — is entirely VirtualiZarr's. We never touch HDF5 internals.

## Built (because nothing does it)

Everything below exists because VirtualiZarr virtualizes and concatenates along a
**regular** grid, and refuses ragged CSR arrays:

```
ValueError: Cannot concatenate arrays with inconsistent chunk shapes:
(5973,) vs (5939,). Requires ZEP003 (Variable-length Chunks).
```

That limit is VirtualiZarr's regular-grid assumption, not a storage wall — Icechunk
stores refs by chunk coordinate regardless of grid. So we bypass its concat and
assemble the array ourselves.

1. **Cross-file rectilinear CSR concatenation** —
   `write_concat_virtual_array` ([`:125`](src/virtual_h5ad/_concat.py#L125)).
   Lay every file's HDF5 chunk edges end to end as one rectilinear grid; emit each
   full source chunk as a `VirtualChunkSpec` pointing at the original bytes. The
   result is one logical `X/data` (and `X/indices`) spanning the whole collection,
   ~99% still referenced.

2. **Padded-tail materialization** — same function. HDF5 pads each array's *final*
   chunk to full size (e.g. 5939 stored, 5918 valid). Concatenated, that padding
   would leak across a file boundary, corrupting the next file's rows. We write
   each file's tail chunk as one **native** chunk holding only its valid elements.
   This is the subtle correctness fix that makes byte-identical roundtrips hold.

3. **Codec split** — same function. A `ManifestArray`'s codec pipeline is
   `(serializer, *compressors)`; we split it to recreate the destination array so
   it decodes the referenced source bytes identically (dtype/filters/fill value
   carried per array, so int32-vs-int64 and gzip-vs-none mix freely across files).

4. **CSR `indptr` recompute** — `_combined_indptr` ([`:217`](src/virtual_h5ad/_concat.py#L217)).
   Row pointers are offset by each file's cumulative nnz and written native (tiny).
   Column `indices` need no remap — the files share a gene panel, guarded by
   `_check_stackable` ([`:193`](src/virtual_h5ad/_concat.py#L193)).

5. **`obs`/`var` carried natively** — `_write_obs_var` ([`:377`](src/virtual_h5ad/_concat.py#L377)).
   String/categorical columns live in HDF5's global heap, not fixed-stride buffers,
   so they aren't byte-range references (VirtualiZarr's `loadable_variables` idea).
   We concatenate `obs`, copy the shared `var`, and `write_elem` them into the repo
   — making it self-contained: metadata reads touch no source file.

6. **Download-then-index** — `_download` ([`:70`](src/virtual_h5ad/_concat.py#L70)).
   Indexing HDF5 means a serial walk of scattered metadata; doing that over HTTP is
   thousands of latency-bound requests. Pulling the file once in concurrent
   range-GET blocks, then parsing locally, is far faster and kinder to the server.
   The repo still references the remote URL. (Streaming-parse path kept as an
   option.)

7. **Self-authorizing open** — `open_repo` ([`:400`](src/virtual_h5ad/_concat.py#L400)).
   Icechunk gates virtual reads on the source-prefix the repo was built against. We
   read those prefixes back from the repo config so a reader needs no prior
   knowledge of where the data lives.

8. **N-D concat for dense obsm** — same `write_concat_virtual_array`, generalized
   from 1-D to N dimensions. Dense embeddings (`X_pca`, `X_umap`, `X_scVI`)
   virtualize like `X`; a chunk is a virtual ref only if it is full on *every*
   axis, else native (padded tails on any axis). Falls back to inlining when files
   chunk the trailing axes differently (`_trailing_consistent` / `_inline_obsm`).

9. **Datasets as groups in one repo** — `group=` / `overwrite=False` append
   ([`_make_repo`](src/virtual_h5ad/_concat.py)). The subtlety is config: virtual
   containers are per-source-prefix and must be *persisted*, and `save_config()`
   saves the config the repo was opened with — so appending a group means reopening
   with the merged config, then saving, or the new container is dropped.

10. **Compute → write-native loop** — `add_qc_metrics`. Read through the virtual X
    (or, for `n_genes`, straight from `indptr` with no matrix read), then write the
    result back as native `obs` columns. Shows the repo is not read-only scaffolding
    — you can store derived values in it.

Selective reads need no custom code: because the collection is one logical CSR `X`,
`adata[mask].to_memory()` slices it natively, and zarr's codec pipeline fans the
range-GETs out concurrently (bounded by `async.concurrency`, default 10 — raise it
for scattered remote reads). An earlier hand-rolled thread-pool reader (`select_cells`)
was dropped once the latency benchmark showed it only matched tuning that knob; the
thread-pool baseline now lives in `latency_benchmark.py`.

## The net

Single-file virtualization is bought. The product — *one or more collections as
self-contained, sliceable sparse matrices in a single repo, read and written with
the ordinary AnnData idiom* — is the pieces above, ~500 lines. The public surface
is three functions: `virtualize_matrix`, `open_anndata`, and `add_qc_metrics`.

Items 1–3 (and the N-D generalization, 8) are the natural upstream candidates
(ragged/CSR/dense concat in VirtualiZarr); the rest are AnnData- and
access-pattern-specific glue.
