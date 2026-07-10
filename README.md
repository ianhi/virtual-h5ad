# virtual-h5ad

Virtually ingest AnnData `.h5ad` files into [Icechunk](https://icechunk.io) using
[VirtualiZarr](https://virtualizarr.readthedocs.io) — referencing the HDF5 byte
ranges as Zarr instead of copying the data.

**Note - Tech Demo**

This is very much in tech demo status. Virtualizing these is absolutely possible, but in getting there I found several performance improvement opportunities up and down the stack. So this uses some branchs of virtualizarr, zarr, and icechunk. ALl of which I plan to upstream so this work well by default.

That said, intial results are extremely promising I can make fairly high performance selections of subsets without downloading the entire files!

## Quick start

```bash
uv sync
bash experiments/00_get_data.sh                  # fetch a real pbmc3k h5ad
uv run python experiments/02_make_collection.py  # shard it into a collection
uv run python tutorial/virtualize_collection.py data/collection/shard_*.h5ad --repo repo/collection
```

## Layout

- **[atlas_demo.ipynb](atlas_demo.ipynb)** — the capstone demo: reference **two**
  published CELLxGENE studies (Allen immune atlas + a 10 GB ocular atlas) into
  **one** repo as separate groups, open each as ordinary `AnnData`, then compute
  QC metrics over the virtual matrix and write them **back into the repo as native
  chunks**. Start here.
- **[virtualize_h5ad_collection.ipynb](virtualize_h5ad_collection.ipynb)** — the
  minimal walkthrough on synthetic shards.
- **[virtualize_allen_collection.ipynb](virtualize_allen_collection.ipynb)** — the
  builder workflow on the Allen study alone.
- **[query_cloud_atlas.ipynb](query_cloud_atlas.ipynb)** — the reader's payoff:
  query a 10 GB CELLxGENE atlas through a small repo, reading only the cells you
  select (free metadata, targeted streaming).
- **[tutorial/TUTORIAL.md](tutorial/TUTORIAL.md)** — the same walkthrough as prose.
- **[src/virtual_h5ad/](src/virtual_h5ad/)** — the package. The public API is
  `virtualize_matrix` (build the repo, one dataset per `group`), `open_anndata`
  (open a group as a normal, lazily-backed `AnnData`), and `add_qc_metrics`
  (compute over the virtual X, store native); the engine lives in
  [`_concat.py`](src/virtual_h5ad/_concat.py).
- **[tutorial/virtualize_collection.py](tutorial/virtualize_collection.py)** —
  the simpler group-per-sample layout (stack at read time).
- **[BUILD.md](BUILD.md)** — what we leaned on existing libraries for vs. what we
  had to build (the build-vs-buy picture).
- **[FINDINGS.md](FINDINGS.md)** — running log of what virtualizes cleanly, what
  doesn't, and why.
- **`experiments/`** — the probes that established the above (numbered).

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
references don't). Not yet done: pointing repo storage at a cloud store like
Arraylake — see FINDINGS.md.
