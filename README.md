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

- **[tutorial/TUTORIAL.md](tutorial/TUTORIAL.md)** — virtualize a *collection* of
  h5ad files into one big matrix (start here).
- **[tutorial/rectilinear_concat.py](tutorial/rectilinear_concat.py)** —
  `virtualize_matrix` / `open_matrix`: the whole collection as one logical `X`.
- **[tutorial/virtualize_collection.py](tutorial/virtualize_collection.py)** —
  the simpler group-per-sample layout (stack at read time).
- **[FINDINGS.md](FINDINGS.md)** — running log of what virtualizes cleanly, what
  doesn't, and why.
- **`experiments/`** — the probes that established the above (numbered).

## Status

Working: virtualizing sparse CSR `X` — single file, a multi-sample collection,
and **one logical matrix across the collection** via rectilinear chunk grids
(~99% virtual, byte-identical roundtrips). Not yet: variable-length strings
(`obs`/`var` indices, categoricals) — see FINDINGS.md.
