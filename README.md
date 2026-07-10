# virtual-h5ad

Virtually ingest AnnData `.h5ad` files into [Icechunk](https://icechunk.io) using
[VirtualiZarr](https://virtualizarr.readthedocs.io) — referencing the HDF5 byte
ranges as Zarr instead of copying the data. A whole collection becomes **one
logical sparse matrix** you can open as an ordinary `AnnData` and slice, streaming
only the cells you select.

**Note — tech demo.** Virtualizing these is absolutely possible, but getting good
performance surfaced improvement opportunities up and down the stack, so this
currently leans on a few branches I plan to upstream:

1. Local optimizations in virtualizarr/zarr to virtualize the sparse layout cleanly.
2. A Zarr codec pipeline for coalescing reads — <https://github.com/ianhi/coalescing-zarr>.
3. An icechunk branch that coalesces range reads in the Rust layer —
   <https://github.com/earth-mover/icechunk/tree/ian/more-specific-vritual>.

A fresh `uv sync` installs all three automatically (the right per-platform fork
wheel is pinned in `[tool.uv.sources]`; no Rust build). Initial results are
promising: fairly high-performance selections of subsets without downloading the
whole files.

## Quick start

```bash
uv sync
```

Then run the two notebooks (launch Jupyter from this directory so the kernel uses
the project `.venv`):

- **[build_virtual_omics.ipynb](build_virtual_omics.ipynb)** — build the repo:
  reference three published CELLxGENE studies as virtual byte ranges (copying no
  expression data) into an Icechunk repo.
- **[virtual_omics_demo.ipynb](virtual_omics_demo.ipynb)** — the payoff:
  reproduce Fig. 3 of Gong/Gustafson et al. (*Nature* 2025) against
  `earthmover-public/virtual-omics` — an obs-only composition analysis over all
  108 donors with zero expression streamed, a targeted expression pull to validate
  it, and a read-coalescing benchmark.

## Public API — `src/virtual_h5ad/`

```python
from virtual_h5ad import virtualize_matrix, open_anndata, add_qc_metrics

virtualize_matrix(["a.h5ad", "b.h5ad"], "repo/atlas")  # build (references only)
adata = open_anndata("repo/atlas")                      # lazy, backed by the repo
sub   = adata[mask].to_memory()                         # streams only those cells
add_qc_metrics("repo/atlas")                            # compute over virtual X, store native
```

The engine is split across small modules: `_sources` (inputs/IO), `_manifest`
(the rectilinear virtual-concat engine), `_write` (build + QC), `_open` (read).
Reads **coalesce by default** — a contiguous selection (e.g. one donor) collapses
from thousands of latency-bound chunk fetches into a handful of large range GETs.
See [`notes/coiled-setup.md`](notes/coiled-setup.md) to run the demo on Coiled.

## Status

Working:

- Virtualizing sparse CSR `X` — single file, a multi-sample collection, and **one
  logical matrix across the collection** via rectilinear chunk grids (~99%
  virtual, byte-identical roundtrips).
- Dense `obsm` embeddings (`X_pca`, `X_umap`, `X_scVI`) — virtualized as byte-range
  refs when the per-file grids line up (always for a single file), inlined
  otherwise (they are tiny next to `X`).
- `obs`/`var` (strings, categoricals) written natively, so a repo is
  self-contained. Several datasets in one repo as named `group`s.
- Computing values over the virtual matrix and writing them **back as native
  chunks** (`add_qc_metrics`).

Local files and remote `http(s)://` sources both work — and because CELLxGENE
serves its files over HTTPS, a repo built against those URLs is portable to a
cloud store as-is (only the *repo's* storage backend changes; the source
references don't), which is how `earthmover-public/virtual-omics` is hosted on
Arraylake.
