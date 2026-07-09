"""Virtualize a collection of h5ad files as one sparse matrix in Icechunk.

Reference the HDF5 byte ranges of many ``.h5ad`` files as a single logical Zarr
matrix — no copying. See the module docstring of :mod:`virtual_h5ad._concat` for
how it works.

    from virtual_h5ad import virtualize_matrix, open_anndata

    virtualize_matrix(["a.h5ad", "b.h5ad"], "repo/atlas")  # build (references)
    adata = open_anndata("repo/atlas")                      # lazy, backed by repo
    sub = adata[mask].to_memory()                           # streams only those cells
    add_qc_metrics("repo/atlas")                            # compute + store native
"""

from ._concat import (
    add_qc_metrics,
    open_anndata,
    open_matrix,
    open_repo,
    virtualize_matrix,
)

__all__ = [
    "virtualize_matrix",
    "open_anndata",
    "open_matrix",
    "open_repo",
    "add_qc_metrics",
]
