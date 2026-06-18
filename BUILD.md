# What we had to build

Reference for contributors: the line between what existing libraries gave us and
what this repo adds. The short version — **VirtualiZarr virtualizes one HDF5 file;
we built the layer that turns a collection of them into a single logical sparse
matrix, plus the read path that makes selective access fast.** All custom code
lives in [`tutorial/rectilinear_concat.py`](tutorial/rectilinear_concat.py).

See [FINDINGS.md](FINDINGS.md) for the experiment-by-experiment log that got us
here; this doc is the consolidated build-vs-buy picture.

## Bought (the heavy lifting we did *not* write)

| Capability | Library | How we use it |
|---|---|---|
| HDF5 chunk B-tree → byte offsets/lengths + codecs | **VirtualiZarr** `HDFParser` / `open_virtual_dataset` | one parse per file → a `ManifestArray` per CSR array (`manifest_arrays`, [`rectilinear_concat.py:99`](tutorial/rectilinear_concat.py#L99)) |
| Virtual-chunk storage & gated reads | **Icechunk** `VirtualChunkContainer` / `VirtualChunkSpec` / `set_virtual_refs` | we hand it the per-chunk references; it stores and serves them |
| Variable-length chunk grids | **zarr 3** `array.rectilinear_chunks` | the grid type that lets one array hold differently-sized chunks |
| Parallel range GETs; HTTP/local stores | **obstore** | bulk download (`_download`) and the concurrent reader |
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
   `write_concat_virtual_array` ([`:119`](tutorial/rectilinear_concat.py#L119)).
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

4. **CSR `indptr` recompute** — `_combined_indptr` ([`:211`](tutorial/rectilinear_concat.py#L211)).
   Row pointers are offset by each file's cumulative nnz and written native (tiny).
   Column `indices` need no remap — the files share a gene panel, guarded by
   `_check_stackable` ([`:187`](tutorial/rectilinear_concat.py#L187)).

5. **`obs`/`var` carried natively** — `_write_obs_var` ([`:369`](tutorial/rectilinear_concat.py#L369)).
   String/categorical columns live in HDF5's global heap, not fixed-stride buffers,
   so they aren't byte-range references (VirtualiZarr's `loadable_variables` idea).
   We concatenate `obs`, copy the shared `var`, and `write_elem` them into the repo
   — making it self-contained: metadata reads touch no source file.

6. **Download-then-index** — `_download` ([`:66`](tutorial/rectilinear_concat.py#L66)).
   Indexing HDF5 means a serial walk of scattered metadata; doing that over HTTP is
   thousands of latency-bound requests. Pulling the file once in concurrent
   range-GET blocks, then parsing locally, is far faster and kinder to the server.
   The repo still references the remote URL. (Streaming-parse path kept as an
   option.)

7. **Concurrent run-coalesced reader** — `select_cells` ([`:435`](tutorial/rectilinear_concat.py#L435)).
   A row gather over CSR is scattered byte ranges. We sort requested rows into
   contiguous runs (one range request each) and fetch them across a thread pool
   (reads release the GIL in Rust), restoring the caller's order. Streaming 3,462
   scattered cells from the 10 GB file: **~160 s serial → ~9 s**.

8. **Self-authorizing open** — `open_repo` ([`:392`](tutorial/rectilinear_concat.py#L392)).
   Icechunk gates virtual reads on the source-prefix the repo was built against. We
   read those prefixes back from the repo config so a reader needs no prior
   knowledge of where the data lives.

## The net

Single-file virtualization is bought. The product — *a collection as one queryable
sparse matrix, self-contained, with fast selective reads* — is the eight pieces
above, ~250 lines. The public surface is three functions: `virtualize_matrix`,
`open_anndata`, `select_cells`.

Items 1–3 are the natural upstream candidates (ragged/CSR concat in VirtualiZarr);
the rest are AnnData- and access-pattern-specific glue.
