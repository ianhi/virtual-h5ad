"""Rectilinear virtual-concat engine.

The part you can ignore when *using* the library. It builds a VirtualiZarr
``ManifestArray`` for one array in a source file, and stitches the per-file arrays
into ONE logical Zarr array on a *rectilinear* (variable-length) chunk grid, where
each full chunk becomes a byte-range reference to the source and only HDF5-padded
tail chunks are materialized. Plus the checks/helpers that decide what can be
concatenated (same genes, consistent obsm grids) and the combined CSR ``indptr``.
"""

from __future__ import annotations

import itertools
import os

import numpy as np
import zarr
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser

from ._sources import MATRIX_GROUP, is_remote, open_h5, read_h5_elem, url_to_path


def manifest_array(url: str, group: str, name: str, registry, reader_factory=None):
    """The VirtualiZarr ManifestArray for a single named array in `group`.

    Local files take the native path: `_construct_manifest_array` reads only the
    HDF5 chunk B-tree (tens of MB) off an h5py-opened dataset. The `HDFParser` route
    reads through obstore's 1 MiB-block reader, which amplifies the B-tree's many
    scattered reads into a near-full-file scan — catastrophic for chunk-dense CSR
    arrays (see notes/virtualizarr-single-cell-frictions.md). Both yield a
    byte-identical ManifestArray. `filepath=url` keeps ref paths consistent (and
    `write_concat_virtual_array` overrides the location with `published` anyway).

    Remote streaming (`download=False`, `reader_factory` passed) keeps the parser
    path: it opens the array in isolation (drop its siblings) because
    `open_virtual_dataset` funnels through an xarray Dataset that auto-names each
    array's leading axis `phony_dim_0` and then refuses siblings whose other axes
    disagree (e.g. `X_pca` k=50 vs `X_umap` k=2), and its `reader_factory` buffers
    reads into fewer, larger requests. The native path needs neither trick —
    `_construct_manifest_array` takes a single dataset directly.
    """
    import h5py

    if not is_remote(url):
        from virtualizarr.parsers.hdf.hdf import _construct_manifest_array

        with h5py.File(url_to_path(url)) as f:
            return _construct_manifest_array(
                filepath=url, dataset=f[f"{group}/{name}"], group=group
            )

    with open_h5(url_to_path(url)) as f:
        siblings = [
            k for k in f[group] if k != name and isinstance(f[group][k], h5py.Dataset)
        ]
    kw = {"reader_factory": reader_factory} if reader_factory is not None else {}
    vds = open_virtual_dataset(
        url=url,
        registry=registry,
        parser=HDFParser(group=group, drop_variables=siblings, **kw),
    )
    return vds[name].data


def write_concat_virtual_array(
    store, dest_path: str, src_key: str, marrs: list, read_paths: list[str], locations
) -> dict:
    """Write `dest_path` = the per-file arrays concatenated along axis 0.

    Works for 1-D CSR components (`X/data`, `X/indices`) and N-D dense arrays
    (`obsm/X_pca`). The concatenated array is a Zarr *rectilinear* grid: axis 0's
    chunk edges are every file's HDF5 edges laid end to end; trailing axes keep the
    (shared) source grid. A chunk that is a full, non-padded chunk on **every** axis
    becomes a virtual reference to the source bytes. Any chunk that is a padded tail
    on some axis (HDF5 pads each array's final chunk per axis) is materialized as a
    native chunk holding only its valid elements — otherwise that padding would leak
    across a file (axis 0) or column (trailing axes) boundary. `src_key` is the
    array's path inside each source file (``"X/data"``, ``"obsm/X_pca"``);
    `locations` is one URL per file for the references (defaults to the parsed path).
    Returns virtual/native chunk counts.
    """
    import icechunk as ic

    if locations is None:
        locations = [None] * len(read_paths)
    first = marrs[0]
    ndim = len(first.shape)
    # trailing axes (everything past the concat axis) must match across files
    for ma in marrs:
        if tuple(ma.shape[1:]) != tuple(first.shape[1:]) or tuple(
            ma.chunks[1:]
        ) != tuple(first.chunks[1:]):
            raise ValueError(f"{src_key}: inconsistent trailing dims across files")

    # axis 0: walk the files, emitting one dest block per source chunk
    blocks0 = []  # {file, local0, e0, tail0}
    for fi, ma in enumerate(marrs):
        c0 = ma.chunks[0]
        n_full, rem = divmod(int(ma.shape[0]), c0)
        for j in range(n_full):
            blocks0.append({"file": fi, "local0": j, "e0": c0, "tail0": False})
        if rem:
            blocks0.append({"file": fi, "local0": n_full, "e0": rem, "tail0": True})
    edges = [[b["e0"] for b in blocks0]]

    # trailing axes: same grid for every file (checked above)
    tail_idx = {}  # axis -> index of the padded tail chunk, or None
    for a in range(1, ndim):
        ca, sa = first.chunks[a], int(first.shape[a])
        n_full, rem = divmod(sa, ca)
        edges.append([ca] * n_full + ([rem] if rem else []))
        tail_idx[a] = n_full if rem else None

    serializer, *compressors = first.metadata.codecs
    arr = zarr.open_group(store, mode="a").create_array(  # create parents if absent
        name=dest_path,
        shape=(sum(edges[0]), *[int(first.shape[a]) for a in range(1, ndim)]),
        dtype=first.dtype,
        chunks=edges,
        serializer=serializer,
        compressors=compressors,
        fill_value=first.metadata.fill_value,
    )

    # dest-offset lookup per axis (cumulative edge sums)
    offs = [np.concatenate([[0], np.cumsum(e)]) for e in edges]
    trailing_ranges = [range(len(edges[a])) for a in range(1, ndim)]
    specs, natives = [], []  # natives: (file, dest_index, src_index_in_file)
    for i0, b in enumerate(blocks0):
        ma = marrs[b["file"]]
        loc = locations[b["file"]]
        for tail in itertools.product(*trailing_ranges):
            dest_index = (i0, *tail)
            src_index = (b["local0"], *tail)
            is_tail = b["tail0"] or any(
                t == tail_idx[a + 1] for a, t in enumerate(tail)
            )
            if is_tail:
                natives.append((b["file"], dest_index, src_index))
            else:
                m = ma.manifest
                specs.append(
                    ic.VirtualChunkSpec(
                        index=list(dest_index),
                        location=loc or str(m._paths[src_index]),
                        offset=int(m._offsets[src_index]),
                        length=int(m._lengths[src_index]),
                    )
                )
    failed = store.set_virtual_refs(
        array_path=arr.path, chunks=specs, validate_containers=True
    )
    if failed:
        # validate_containers=True *drops* refs whose source location matches no
        # configured virtual chunk container and returns their indices (it does not
        # raise). Ignoring that silently commits an array with shape but no data,
        # which reads back as all-zeros with no error. Fail loudly instead: this
        # means the container was not installed on the repo (see _make_repo).
        raise RuntimeError(
            f"{dest_path}: {len(failed)} of {len(specs)} virtual refs were rejected — "
            "no configured virtual chunk container matches their source prefix, so they "
            "were dropped. The repo's virtual_chunk_container is missing/not installed; "
            "fix container setup before writing refs."
        )

    # materialize native (padded-tail) chunks: read each file's valid sub-block once
    by_file: dict[int, list] = {}
    for fi, dest_index, src_index in natives:
        by_file.setdefault(fi, []).append((dest_index, src_index))
    for fi, items in by_file.items():
        ma = marrs[fi]
        with open_h5(read_paths[fi]) as f:
            src = f[src_key]
            for dest_index, src_index in items:
                dslice = tuple(
                    slice(int(offs[a][dest_index[a]]), int(offs[a][dest_index[a] + 1]))
                    for a in range(ndim)
                )
                sslice = tuple(
                    slice(
                        src_index[a] * ma.chunks[a],
                        src_index[a] * ma.chunks[a]
                        + (dslice[a].stop - dslice[a].start),
                    )
                    for a in range(ndim)
                )
                arr[dslice] = src[sslice]
    return {"virtual": len(specs), "native": len(natives)}


def check_stackable(paths: list[str]) -> None:
    """Fail loudly unless every file is a CSR matrix over the same genes (var).

    Concatenating CSR `indices`/`data` by bytes assumes identical columns and CSR
    layout; otherwise the stacked result is silently wrong. Reads only the small
    X attrs + var index.
    """
    import anndata.io as aio

    ref = None
    for p in paths:
        with open_h5(p) as f:
            if f[MATRIX_GROUP].attrs.get("encoding-type") != "csr_matrix":
                raise ValueError(f"{os.path.basename(p)}: X is not a CSR matrix.")
            names = aio.read_elem(f["var"]).index
        if ref is None:
            ref = names
        elif len(names) != len(ref) or not (names == ref).all():
            raise ValueError(
                f"{os.path.basename(p)} has a different gene set than "
                f"{os.path.basename(paths[0])} — files must share genes to stack."
            )


def shared_obsm(paths: list[str]) -> list[str]:
    """Dense `obsm` arrays present (with matching width) in *every* file.

    These are byte-range-virtualizable embeddings (e.g. `X_pca`, `X_umap`). Groups
    (sparse/dataframe obsm) and arrays missing from some file are skipped.
    """
    import h5py

    per_file = []
    for p in paths:
        with open_h5(p) as f:
            if "obsm" not in f:
                return []
            g = f["obsm"]
            per_file.append(
                {k: g[k].shape[1:] for k in g if isinstance(g[k], h5py.Dataset)}
            )
    common = set(per_file[0])
    for d in per_file[1:]:
        common &= set(d)
    return sorted(k for k in common if all(d[k] == per_file[0][k] for d in per_file))


def trailing_consistent(marrs: list) -> bool:
    """Whether every file chunks the obsm array's trailing axes identically.

    The rectilinear product grid needs one shared trailing-axis edge list, so a
    virtual concat is only possible when the files agree; otherwise we inline.
    """
    first = marrs[0]
    return all(
        tuple(ma.shape[1:]) == tuple(first.shape[1:])
        and tuple(ma.chunks[1:]) == tuple(first.chunks[1:])
        for ma in marrs
    )


def inline_obsm(store, group: str | None, name: str, paths: list[str]) -> None:
    """Copy an obsm embedding into the repo as a native array (concatenated axis 0)."""
    import anndata.io as aio

    dsg = zarr.open_group(store, mode="r+")
    dest = dsg[group] if group else dsg
    if "obsm" not in dest:
        dest.create_group("obsm")
    arrs = [read_h5_elem(p, f"obsm/{name}") for p in paths]
    aio.write_elem(dest["obsm"], name, np.concatenate(arrs, axis=0))


def combined_indptr(paths: list[str]):
    """Vertically-stacked CSR indptr: each file's row pointers shifted by prior nnz."""
    out = [np.array([0], dtype="int64")]
    nnz, n_var = 0, 0
    for p in paths:
        with open_h5(p) as f:
            ip = f[f"{MATRIX_GROUP}/indptr"][:]
            n_var = int(f[MATRIX_GROUP].attrs["shape"][1])
        out.append(ip[1:].astype("int64") + nnz)
        nnz += int(ip[-1])
    return np.concatenate(out), n_var
