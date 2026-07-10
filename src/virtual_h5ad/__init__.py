"""Virtualize a collection of h5ad files as one sparse matrix in Icechunk.

Reference the HDF5 byte ranges of many ``.h5ad`` files as a single logical Zarr
matrix — no copying:

    from virtual_h5ad import virtualize_matrix, open_anndata

    virtualize_matrix(["a.h5ad", "b.h5ad"], "repo/atlas")  # build (references)
    adata = open_anndata("repo/atlas")                      # lazy, backed by repo
    sub = adata[mask].to_memory()                           # streams only those cells
    add_qc_metrics("repo/atlas")                            # compute + store native

The engine is split across small modules: ``_sources`` (inputs/IO), ``_manifest``
(the rectilinear virtual-concat engine), ``_write`` (build + QC), ``_open`` (read).
"""

from __future__ import annotations

import os
import warnings

import zarr

# Icechunk's Rust backend logs a LocalFileSystem warning on every commit; quiet it
# so notebooks stay readable (set before icechunk is imported, lazily, below).
os.environ.setdefault("ICECHUNK_LOG", "off")

# zarr 3 warns once per array-create that autosharding will change; we create many
# arrays per repo, which floods notebooks. The default is fine for us — quiet it.
warnings.filterwarnings("ignore", message=".*autosharding.*", category=UserWarning)

# Our concatenated arrays use rectilinear (variable-length) chunk grids; readers
# and writers both need this enabled.
zarr.config.set({"array.rectilinear_chunks": True})

from ._open import open_anndata, open_matrix, open_repo  # noqa: E402
from ._write import (  # noqa: E402
    add_missing_elements,
    add_qc_metrics,
    virtualize_matrix,
)

__all__ = [
    "virtualize_matrix",
    "add_missing_elements",
    "open_anndata",
    "open_matrix",
    "open_repo",
    "add_qc_metrics",
]
