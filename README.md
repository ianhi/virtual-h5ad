# virtual-h5ad

Virtually ingest AnnData `.h5ad` files into [Icechunk](https://icechunk.io) using
[VirtualiZarr](https://virtualizarr.readthedocs.io) — referencing the HDF5 byte
ranges as Zarr instead of copying the data.

## Quick start

```bash
uv sync
bash experiments/00_get_data.sh                  # fetch a real pbmc3k h5ad
uv run python experiments/02_make_collection.py  # shard it into a collection
uv run python tutorial/virtualize_collection.py data/collection/shard_*.h5ad --repo repo/collection
```

## Layout

- **[virtualize_h5ad_collection.ipynb](virtualize_h5ad_collection.ipynb)** — the
  runnable notebook walkthrough (start here).
- **[virtualize_allen_collection.ipynb](virtualize_allen_collection.ipynb)** — the
  same workflow on a real study (Allen Institute Human Immune Health Atlas).
- **[query_cloud_atlas.ipynb](query_cloud_atlas.ipynb)** — the reader's payoff:
  query a 10 GB CELLxGENE atlas through a 31 MB repo, reading only the cells you
  select (free metadata, targeted streaming).
- **[tutorial/TUTORIAL.md](tutorial/TUTORIAL.md)** — the same walkthrough as prose.
- **[tutorial/rectilinear_concat.py](tutorial/rectilinear_concat.py)** —
  `virtualize_matrix` / `open_matrix`: the whole collection as one logical `X`.
- **[tutorial/virtualize_collection.py](tutorial/virtualize_collection.py)** —
  the simpler group-per-sample layout (stack at read time).
- **[BUILD.md](BUILD.md)** — what we leaned on existing libraries for vs. what we
  had to build (the build-vs-buy picture).
- **[FINDINGS.md](FINDINGS.md)** — running log of what virtualizes cleanly, what
  doesn't, and why.
- **`experiments/`** — the probes that established the above (numbered).

## Status

Working: virtualizing sparse CSR `X` — single file, a multi-sample collection,
and **one logical matrix across the collection** via rectilinear chunk grids
(~99% virtual, byte-identical roundtrips). Not yet: variable-length strings
(`obs`/`var` indices, categoricals) — see FINDINGS.md.
