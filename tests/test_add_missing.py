"""`add_missing_elements` backfills raw/uns onto a group built without them.

Simulates the situation the published atlas is in: an older, lossy build captured
X/obsm/obs/var but dropped `raw` and `uns`. We reproduce that by virtualizing a
*stripped* copy (no raw/uns), then call `add_missing_elements` against the full
source to commit the missing pieces additively, and assert the reopened object now
carries a faithful `raw` and `uns` (and that the existing X is untouched).
"""

from __future__ import annotations

from anndata.tests.helpers import assert_equal

from virtual_h5ad import add_missing_elements, open_anndata, virtualize_matrix
from virtual_h5ad._open import open_root

from test_roundtrip import make_synthetic


def test_add_missing_elements_backfills_raw_and_uns(tmp_path):
    full = make_synthetic()
    stripped = full.copy()
    del stripped.raw  # the old, lossy builder left `raw` out

    full_src = tmp_path / "full.h5ad"
    stripped_src = tmp_path / "stripped.h5ad"
    full.write_h5ad(full_src)
    stripped.write_h5ad(stripped_src)

    repo = str(tmp_path / "repo")
    virtualize_matrix(str(stripped_src), repo_path=repo, overwrite=True)
    # anndata always writes a `uns` group (even when empty), so the stripped build
    # still captures one — drop it from the repo to mimic the old builder, which
    # never wrote `uns` at all (the state the published atlas is actually in).
    node, session = open_root(repo, writable=True)
    del node["uns"]
    session.commit("simulate old lossy build: drop uns")

    # the lossy repo really is missing raw/uns
    before = open_anndata(repo).to_memory()
    assert before.raw is None
    assert not before.uns

    snap = add_missing_elements(str(full_src), repo_path=repo)
    assert snap is not None  # something was committed

    after = open_anndata(repo).to_memory()
    assert_equal(after.raw.X, full.raw.X)  # referenced from the full source
    assert_equal(after.raw.var, full.raw.var)
    assert_equal(after.uns, full.uns)
    assert_equal(after.X, before.X)  # existing refs untouched

    # idempotent: a second call finds nothing missing and makes no commit
    assert add_missing_elements(str(full_src), repo_path=repo) is None
