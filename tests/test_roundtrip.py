"""Faithful-virtualization roundtrip: `open_anndata(virtualize(a))` must equal `a`.

The core contract of this library is that virtualizing an AnnData leaves *nothing*
behind — every element (X, raw, layers, obsm, varm, obsp, varp, uns, obs, var)
survives a build→open roundtrip. `make_synthetic()` builds a small AnnData that
exercises every element type; the test virtualizes it to a local Icechunk repo and
asserts the reopened object is deep-equal to the original (anndata's own
`assert_equal`). Big numeric arrays are referenced (virtual), the rest is copied
native — but the reader must reconstruct all of it either way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import anndata as ad
from anndata.tests.helpers import assert_equal

from virtual_h5ad import open_anndata, virtualize_matrix


def make_synthetic(n: int = 40, n_var: int = 12, n_raw_var: int = 30) -> ad.AnnData:
    """A small AnnData touching every on-disk element type.

    `raw` deliberately has a *different* (larger) gene set than the main matrix, so
    the roundtrip exercises raw carrying its own `var`. Kept deterministic (seeded).
    """
    rng = np.random.default_rng(0)

    def csr(r, c, seed):
        return sp.random(r, c, density=0.3, format="csr", dtype="float32", random_state=seed)

    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(["A", "B", "C", "D"] * (n // 4)),
            "score": rng.standard_normal(n).astype("float64"),
        },
        index=[f"cell{i}" for i in range(n)],
    )
    var = pd.DataFrame(
        {"highly_variable": rng.integers(0, 2, n_var).astype(bool)},
        index=[f"gene{i}" for i in range(n_var)],
    )
    adata = ad.AnnData(X=csr(n, n_var, 0), obs=obs, var=var)

    raw_var = pd.DataFrame(index=[f"gene{i}" for i in range(n_raw_var)])
    adata.raw = ad.AnnData(X=csr(n, n_raw_var, 1), var=raw_var)

    adata.layers["counts"] = csr(n, n_var, 2)
    adata.obsm["X_pca"] = rng.standard_normal((n, 5)).astype("float32")
    adata.obsm["X_umap"] = rng.standard_normal((n, 2)).astype("float32")
    adata.varm["PCs"] = rng.standard_normal((n_var, 5)).astype("float32")
    adata.obsp["connectivities"] = csr(n, n, 3)
    adata.varp["cor"] = csr(n_var, n_var, 4)
    adata.uns = {
        "title": "synthetic",
        "n_cells": 42,
        "colors": np.array(["#ffffff", "#000000"]),
        "params": {"method": "umap", "k": 15},
    }
    return adata


@pytest.fixture
def synthetic() -> ad.AnnData:
    return make_synthetic()


def test_virtualize_roundtrip_is_faithful(synthetic, tmp_path):
    """Every element survives build→open, byte-identically for the referenced arrays."""
    src = tmp_path / "synth.h5ad"
    synthetic.write_h5ad(src)

    repo = str(tmp_path / "repo")
    virtualize_matrix(str(src), repo_path=repo, overwrite=True)

    roundtripped = open_anndata(repo).to_memory()
    assert_equal(roundtripped, synthetic)
