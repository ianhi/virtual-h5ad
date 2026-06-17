#!/usr/bin/env bash
# Fetch + build the test h5ad files used by the experiments.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
# Classic pbmc3k tutorial dataset (hosted by the scanpy/anndata author), LEGACY format.
curl -sL "https://falexwolf.de/data/pbmc3k_raw.h5ad" -o data/pbmc3k_raw.h5ad
# Round-trip through current anndata to get a MODERN-format file.
uv run python -c "import anndata as ad; ad.read_h5ad('data/pbmc3k_raw.h5ad').write_h5ad('data/pbmc3k_modern.h5ad', compression='gzip')"
echo "done -> data/pbmc3k_raw.h5ad (legacy), data/pbmc3k_modern.h5ad (modern)"
