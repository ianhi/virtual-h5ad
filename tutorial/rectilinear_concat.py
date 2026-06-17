"""Virtualize a collection of h5ad files as ONE sparse matrix in Icechunk.

This module hides the virtualization details. The user-facing surface is three
functions:

    virtualize_matrix(source, repo_path)  -> build the repo (references, no copy)
    open_anndata(repo_path)               -> a normal AnnData backed by the repo
    select_cells(repo_path, obs_indices)  -> a CSR submatrix, read lazily

How it works (the part you can ignore): the files' sparse `X` arrays are stitched
into one logical array using Zarr *rectilinear* (variable-length) chunk grids, so
each chunk can point straight at a byte range in the original file. VirtualiZarr
won't concatenate ragged chunk grids itself, so we assemble the references here.
See FINDINGS.md for the full story.
"""

from __future__ import annotations

import os

# Icechunk's Rust backend logs a LocalFileSystem warning on every commit; quiet it
# so notebooks stay readable. Set before icechunk is imported; respects an override.
os.environ.setdefault("ICECHUNK_LOG", "off")

import numpy as np
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser

zarr.config.set({"array.rectilinear_chunks": True})

MATRIX_GROUP = "X"
CSR_ARRAYS = ("data", "indices", "indptr")


def _registry(root: str) -> ObjectStoreRegistry:
    """Authorize VirtualiZarr to read source bytes under `root`."""
    return ObjectStoreRegistry({f"file://{root}": LocalStore(prefix=root)})


def manifest_arrays(path: str, names, registry: ObjectStoreRegistry) -> dict:
    """The VirtualiZarr ManifestArrays for the named CSR components of one file.

    One parse per file: `data` and `indices` share a length (nnz), so they open
    together cleanly. (`indptr` has a different length and would collide on the
    shared phony dimension, so it is never parsed alongside them.)
    """
    drop = [a for a in CSR_ARRAYS if a not in names]
    vds = open_virtual_dataset(
        url=f"file://{os.path.abspath(path)}",
        registry=registry,
        parser=HDFParser(group=MATRIX_GROUP, drop_variables=drop),
    )
    return {n: vds[n].data for n in names}


def write_concat_virtual_array(store, name: str, marrs: list, paths: list[str]) -> dict:
    """Write `X/<name>` = the per-file arrays concatenated on axis 0 (rectilinear).

    Full chunks become virtual references to the source bytes; each file's padded
    final chunk (HDF5 pads it to full size) is materialized as one native chunk
    holding only its valid elements, so no padding leaks across a file boundary.
    `marrs` is one ManifestArray per file. Returns virtual/native chunk counts.
    """
    import h5py

    first = marrs[0]
    edges: list[int] = []
    specs = []
    natives = []  # (array_start, valid_len, source_path) for padded tails
    pos = 0
    for path, ma in zip(paths, marrs):
        chunk = ma.chunks[0]
        n_full, rem = divmod(int(ma.shape[0]), chunk)
        m = ma.manifest
        for i in range(n_full):  # full chunks -> virtual refs
            specs.append(
                _VChunk(
                    index=(len(edges),),
                    location=str(m._paths.ravel()[i]),
                    offset=int(m._offsets.ravel()[i]),
                    length=int(m._lengths.ravel()[i]),
                )
            )
            edges.append(chunk)
            pos += chunk
        if rem:  # padded tail chunk -> materialize valid part as native
            natives.append((pos, rem, path))
            edges.append(rem)
            pos += rem

    # ManifestArray codecs = (array->bytes serializer, *compressors); split them
    # so the new array decodes the referenced source bytes identically.
    serializer, *compressors = first.metadata.codecs
    arr = zarr.open_group(store, mode="a").create_array(  # create root if absent
        name=f"{MATRIX_GROUP}/{name}",
        shape=(pos,),
        dtype=first.dtype,
        chunks=[edges],  # rectilinear: explicit per-chunk edges
        serializer=serializer,
        compressors=compressors,
        fill_value=first.metadata.fill_value,
    )
    store.set_virtual_refs(array_path=arr.path, chunks=specs, validate_containers=True)
    for start, valid, path in natives:
        with h5py.File(path) as f:
            arr[start : start + valid] = f[f"{MATRIX_GROUP}/{name}"][-valid:]
    return {"virtual": len(specs), "native": len(natives)}


def _VChunk(index, location, offset, length):
    # Icechunk's VirtualChunkSpec is a positional-args pyo3 class; wrap for clarity.
    import icechunk as ic

    return ic.VirtualChunkSpec(
        index=list(index), location=location, offset=offset, length=length
    )


def _check_stackable(paths: list[str]) -> None:
    """Fail loudly unless every file is a CSR matrix over the same genes (var).

    Concatenating CSR `indices`/`data` by bytes assumes identical columns and CSR
    layout; otherwise the stacked result is silently wrong.
    """
    import anndata as ad

    ref = ad.read_h5ad(paths[0], backed="r").var_names
    for p in paths:
        a = ad.read_h5ad(p, backed="r")
        if getattr(a.X, "format", None) != "csr":
            raise ValueError(f"{os.path.basename(p)}: X is not a CSR matrix.")
        if len(a.var_names) != len(ref) or not (a.var_names == ref).all():
            raise ValueError(
                f"{os.path.basename(p)} has a different gene set than "
                f"{os.path.basename(paths[0])} — files must share genes to stack."
            )


def _combined_indptr(paths: list[str]):
    """Vertically-stacked CSR indptr: each file's row pointers shifted by prior nnz."""
    import h5py

    out = [np.array([0], dtype="int64")]
    nnz, n_var = 0, 0
    for p in paths:
        with h5py.File(p) as f:
            ip = f[f"{MATRIX_GROUP}/indptr"][:]
            n_var = int(f[MATRIX_GROUP].attrs["shape"][1])
        out.append(ip[1:].astype("int64") + nnz)
        nnz += int(ip[-1])
    return np.concatenate(out), n_var


def _resolve_paths(source) -> list[str]:
    """Accept a list of files, a glob string, or a directory of .h5ad files."""
    import glob

    if isinstance(source, (list, tuple)):
        paths = list(source)
    elif os.path.isdir(source):
        paths = sorted(glob.glob(os.path.join(source, "*.h5ad")))
    else:
        paths = sorted(glob.glob(source))
    if not paths:
        raise FileNotFoundError(f"no .h5ad files matched: {source!r}")
    return [os.path.abspath(p) for p in paths]


def virtualize_matrix(source, repo_path: str, overwrite: bool = True) -> str:
    """Reference a collection's X as ONE virtual CSR matrix in a new Icechunk repo.

    `source` may be a list of files, a glob (``"dir/*.h5ad"``), or a directory.
    `data`/`indices` become single rectilinear virtual arrays spanning all files;
    `indptr` is recomputed and written native. Returns the snapshot id.

    `overwrite` (default True) replaces any existing repo at `repo_path` so the
    call is safe to re-run; set False to require a clean path.
    """
    import shutil

    import icechunk as ic

    if overwrite and os.path.isdir(repo_path):
        shutil.rmtree(repo_path)
    paths = _resolve_paths(source)
    _check_stackable(paths)
    root = os.path.commonpath([os.path.dirname(p) for p in paths])
    config = ic.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        ic.VirtualChunkContainer(
            f"file://{root}/", ic.local_filesystem_store(f"{root}/")
        )
    )
    repo = ic.Repository.create(
        storage=ic.local_filesystem_storage(os.path.abspath(repo_path)),
        config=config,
        authorize_virtual_chunk_access=ic.containers_credentials(
            {f"file://{root}/": None}
        ),
    )
    session = repo.writable_session("main")

    registry = _registry(root)  # built once, reused for every file
    parsed = [
        manifest_arrays(p, ("data", "indices"), registry) for p in paths
    ]  # 1 parse/file
    for name in ("data", "indices"):
        write_concat_virtual_array(
            session.store, name, [pa[name] for pa in parsed], paths
        )

    indptr, n_var = _combined_indptr(paths)
    g = zarr.open_group(session.store, mode="r+")
    g.create_array(
        name=f"{MATRIX_GROUP}/indptr",
        shape=indptr.shape,
        dtype=indptr.dtype,
        chunks=indptr.shape,
    )[:] = indptr
    g[MATRIX_GROUP].attrs.update(
        {
            "encoding-type": "csr_matrix",
            "encoding-version": "0.1.0",
            "shape": [int(indptr.size - 1), n_var],
        }
    )
    # record sources so a reader can reattach obs/var without being told where they are
    g.attrs["source_files"] = paths
    return session.commit(f"one virtual X over {len(paths)} files")


def open_repo(repo_path: str):
    """Open an Icechunk repo, self-authorizing every virtual container it declares.

    The container prefixes are stored in the repo config, so a reader needs no
    prior knowledge of where the source files live (local prefixes use no creds).
    """
    import icechunk as ic

    storage = ic.local_filesystem_storage(os.path.abspath(repo_path))
    prefixes = ic.Repository.open(storage=storage).config.virtual_chunk_containers or {}
    return ic.Repository.open(
        storage=storage,
        authorize_virtual_chunk_access=ic.containers_credentials(
            {p: None for p in prefixes}
        ),
    )


def _open_root(repo_path: str):
    """The repo's root Zarr group, read-only (X lives at `MATRIX_GROUP`)."""
    return zarr.open_group(
        open_repo(repo_path).readonly_session("main").store, mode="r"
    )


def _csr(x):
    """Build a scipy CSR matrix from an X group (reads the referenced bytes)."""
    from scipy import sparse

    return sparse.csr_matrix(
        (x["data"][:], x["indices"][:], x["indptr"][:]), shape=tuple(x.attrs["shape"])
    )


def open_matrix(repo_path: str):
    """Open the repo and return X as one scipy CSR matrix (rows read on demand).

    No flags or source paths to set — the module enables rectilinear chunks on
    import and `open_repo` self-authorizes from the repo's own config.
    """
    return _csr(_open_root(repo_path)[MATRIX_GROUP])


def select_cells(repo_path: str, obs_indices):
    """Read an arbitrary set of cells (rows) as a CSR submatrix, lazily.

    Because the whole collection is one logical `X`, a selection that mixes cells
    from different source files is just a row gather: `obs_indices` can span file
    boundaries freely. Only the chunks covering the requested rows are fetched
    from the source files — the rest of the matrix is never read.
    """
    from scipy import sparse

    x = _open_root(repo_path)[MATRIX_GROUP]
    indptr = x["indptr"][:]  # small (one int per cell); the bulk stays on disk
    data, indices = x["data"], x["indices"]
    rows_d, rows_i, new_indptr = [], [], [0]
    for r in obs_indices:
        s, e = int(indptr[r]), int(indptr[r + 1])
        rows_d.append(data[s:e])
        rows_i.append(indices[s:e])
        new_indptr.append(new_indptr[-1] + (e - s))
    n_var = int(x.attrs["shape"][1])
    return sparse.csr_matrix(
        (np.concatenate(rows_d), np.concatenate(rows_i), np.array(new_indptr)),
        shape=(len(obs_indices), n_var),
    )


def open_anndata(repo_path: str):
    """Open the repo as an AnnData with a lazy, virtual X over the whole collection.

    X is a backed sparse dataset: opening reads no expression data, and slicing
    (``adata[rows]``) fetches only those rows' chunks from the original files.
    obs (concatenated) and var (shared) are loaded from those files so you get a
    normal, queryable AnnData. Call ``adata.to_memory()`` to materialize all of X.
    """
    import anndata as ad
    import anndata.io as aio
    import pandas as pd

    root = _open_root(repo_path)
    sources = list(root.attrs["source_files"])
    obs = pd.concat([ad.read_h5ad(p, backed="r").obs for p in sources])
    var = ad.read_h5ad(sources[0], backed="r").var
    return ad.AnnData(X=aio.sparse_dataset(root[MATRIX_GROUP]), obs=obs, var=var)
