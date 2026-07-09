"""Build the atlas demo into an Arraylake-backed Icechunk repo (Coiled-scale).

Rebuilds three published CZ CELLxGENE studies as *virtual references* — copying no
expression data — into one repo, as groups:

    allen/immune_health_atlas        Human Immune Health Atlas       1.82M cells
    allen/immunobiology_of_aging     Immunobiology of Aging Cohort   3.76M cells
    ocular_surface_atlas             ocular surface atlas             815k cells

The two Allen studies (Gustafson et al., Nature 2025) are *authored by* the Allen
Institute for Immunology but referenced here from their **CELLxGENE-hosted** copies,
because CELLxGENE serves HTTP range requests (Allen's own portal does not) — so the
refs resolve from anywhere and the repo stays tiny. The ocular atlas is a separate
(non-Allen) CELLxGENE study, hence its own top-level group.

Basic virtualization only — no computation over the matrices. Virtualizing walks
each file's HDF5 chunk index and copies the small metadata (obs/var/indptr); it
never reads expression values. The heavy step is a one-time download used only for
indexing each ~20-40 GB file (cached and reused on restart). Intended for Coiled:

    uv run --with arraylake python push_to_arraylake.py

Note: `obs` is written natively, so a node needs enough RAM to hold the largest
frame (~3.76M rows for the aging cohort). Requires an authenticated Arraylake
session (``arraylake auth login`` or a token in the environment). Derived values
(QC, a normalized layer) are a separate Coiled job.
"""

from __future__ import annotations

import os
import time

CELLXGENE = "https://datasets.cellxgene.cziscience.com/"
REPO_NAME = "earthmover-public/virtual-omics-demo"

# (group, single full-atlas CELLxGENE file, label). Each is one range-servable file;
# the per-cell-type partitions are just the same cells split, so we take the whole.
DATASETS = [
    (
        "allen/immune_health_atlas",
        CELLXGENE + "c1ce03ce-4b71-4b8d-b695-c3cec99cdfc2.h5ad",
        "Human Immune Health Atlas (1.82M cells)",
    ),
    (
        "allen/immunobiology_of_aging",
        CELLXGENE + "924554e2-d893-4015-be86-017e96fab55b.h5ad",
        "Immunobiology of Aging (3.76M cells)",
    ),
    (
        "ocular_surface_atlas",
        CELLXGENE + "30127f02-d0b9-4b56-bf4c-e5b77e4989bd.h5ad",
        "Ocular surface atlas (815k cells)",
    ),
]


def main() -> None:
    from arraylake import Client

    from virtual_h5ad import virtualize_matrix
    from virtual_h5ad._open import open_root
    from virtual_h5ad._sources import MATRIX_GROUP

    # Arraylake hands back an icechunk.Repository. We do NOT pass
    # authorize_virtual_chunk_access (that maps prefixes to Arraylake S3 bucket
    # nicknames, for private sources): CELLxGENE is public HTTPS, and this script
    # only writes refs + reads native metadata, neither of which needs virtual-read
    # authorization. virtualize_matrix configures + persists the HTTP container.
    client = Client()
    repo = client.get_or_create_repo(REPO_NAME)

    # Indexing downloads are cached here and NOT deleted, so a restart reuses them.
    # A local ocular copy under data/ocular_cache is reused if present.
    cache = "data/_source_cache"
    os.makedirs(cache, exist_ok=True)

    for i, (group, url, label) in enumerate(DATASETS, 1):
        cdir = (
            "data/ocular_cache"
            if group == "ocular_surface_atlas" and os.path.isdir("data/ocular_cache")
            else cache
        )
        print(
            f"[{i}/{len(DATASETS)}] virtualizing {label} → {REPO_NAME} /{group} "
            "(downloading to index if not cached)…",
            flush=True,
        )
        t = time.time()
        virtualize_matrix(url, group=group, repo=repo, cache_dir=cdir)
        print(f"        ✓ {group} committed in {time.time() - t:.0f}s", flush=True)

    # Verify by reopening each group — metadata only (X attrs + obsm keys), so this
    # does not read the matrices or load the full obs frames.
    print("verifying (reopening each group from Arraylake)…", flush=True)
    for group, _url, _label in DATASETS:
        root = open_root(repo, group)
        shape = tuple(int(x) for x in root[MATRIX_GROUP].attrs["shape"])
        obsm = list(root["obsm"].array_keys()) if "obsm" in root else []
        print(f"  {group:30} {shape}  obsm={obsm}", flush=True)
    print(f"\ndone → {REPO_NAME}  (derived values: run QC / normalization on Coiled)")


if __name__ == "__main__":
    main()
