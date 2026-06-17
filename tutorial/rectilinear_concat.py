"""Concatenate the per-file CSR component arrays into ONE virtual array.

VirtualiZarr refuses to concatenate arrays with different chunk shapes — a
single Zarr array needs a regular grid, and ragged h5ad files don't have one.
But Zarr now supports *rectilinear* (variable-length) chunk grids, and Icechunk
stores virtual chunk references by chunk coordinate regardless of grid shape.

So we bypass VirtualiZarr's concat entirely and assemble the array ourselves:
the combined chunk grid is the concatenation of every file's HDF5 chunk edges
(each file's partial final chunk becomes its own grid cell), and every grid cell
points at exactly one source chunk's byte range.

This is the part VirtualiZarr does not do yet. Everything else (parsing each
file into a ManifestArray, the Icechunk container/auth) is reused.
"""

from __future__ import annotations

import os

import numpy as np
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser

zarr.config.set({"array.rectilinear_chunks": True})


def manifest_array(path: str, array: str, root: str):
    """The VirtualiZarr ManifestArray for one CSR component of one file."""
    others = [a for a in ("data", "indices", "indptr") if a != array]
    reg = ObjectStoreRegistry({f"file://{root}": LocalStore(prefix=root)})
    vds = open_virtual_dataset(
        url=f"file://{os.path.abspath(path)}",
        registry=reg,
        parser=HDFParser(group="X", drop_variables=others),
    )
    return vds[array].data


def write_concat_virtual_array(store, group: str, name: str, paths: list[str], root_dir: str) -> dict:
    """Write one array = the files' `X/<name>` arrays concatenated on axis 0.

    The combined chunk grid is rectilinear: every file's HDF5 chunk edges laid
    end to end. Full chunks become virtual references to the source bytes; each
    file's padded final chunk (HDF5 pads it to full size) is materialized as a
    native chunk holding only its valid elements, so no padding leaks across a
    file boundary. Returns counts of virtual vs native chunks.
    """
    marrs = [manifest_array(p, name, root_dir) for p in paths]
    first = marrs[0]

    edges: list[int] = []
    specs = []
    natives = []  # (array_start, valid_len, source_path) for partial tails
    array_pos = 0
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
            array_pos += chunk
        if rem:  # padded tail chunk -> materialize valid part as native
            natives.append((array_pos, rem, path))
            edges.append(rem)
            array_pos += rem

    root = zarr.open_group(store, mode="a")  # create root group if absent
    # ManifestArray codecs = (array->bytes serializer, *compressors); split them
    # so the new array decodes the referenced source bytes identically.
    serializer, *compressors = first.metadata.codecs
    arr = root.create_array(
        name=f"{group}/{name}",
        shape=(array_pos,),
        dtype=first.dtype,
        chunks=[edges],  # rectilinear: explicit per-chunk edges
        serializer=serializer,
        compressors=compressors,
        fill_value=first.metadata.fill_value,
    )
    store.set_virtual_refs(array_path=arr.path, chunks=specs, validate_containers=True)
    for start, valid, path in natives:  # native tails (read valid elements from source)
        import h5py

        with h5py.File(path) as f:
            arr[start : start + valid] = f[f"X/{name}"][-valid:]
    return {"virtual": len(specs), "native": len(natives)}


# Icechunk's VirtualChunkSpec is a positional-args pyo3 class; wrap for clarity.
def _VChunk(index, location, offset, length):
    import icechunk as ic

    return ic.VirtualChunkSpec(
        index=list(index), location=location, offset=offset, length=length
    )


def _combined_indptr(paths: list[str]):
    """Vertically-stacked CSR indptr: each file's row pointers shifted by prior nnz."""
    import h5py

    out = [np.array([0], dtype="int64")]
    nnz, n_var = 0, None
    for p in paths:
        with h5py.File(p) as f:
            ip = f["X/indptr"][:]
            n_var = int(f["X"].attrs["shape"][1])
        out.append(ip[1:].astype("int64") + nnz)
        nnz += int(ip[-1])
    return np.concatenate(out), n_var


def virtualize_matrix(paths: list[str], repo_path: str) -> str:
    """Reference a collection's X as ONE virtual CSR matrix in a new Icechunk repo.

    `data`/`indices` become single rectilinear virtual arrays spanning all files;
    `indptr` is recomputed and written native. Returns the snapshot id.
    """
    import icechunk as ic

    root = os.path.commonpath([os.path.dirname(os.path.abspath(p)) for p in paths])
    config = ic.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        ic.VirtualChunkContainer(f"file://{root}/", ic.local_filesystem_store(f"{root}/"))
    )
    repo = ic.Repository.create(
        storage=ic.local_filesystem_storage(os.path.abspath(repo_path)),
        config=config,
        authorize_virtual_chunk_access=ic.containers_credentials({f"file://{root}/": None}),
    )
    session = repo.writable_session("main")
    for name in ("data", "indices"):
        write_concat_virtual_array(session.store, "X", name, paths, root)
    indptr, n_var = _combined_indptr(paths)
    g = zarr.open_group(session.store, mode="r+")
    g.create_array(name="X/indptr", shape=indptr.shape, dtype=indptr.dtype, chunks=indptr.shape)[
        :
    ] = indptr
    g["X"].attrs.update(
        {"encoding-type": "csr_matrix", "encoding-version": "0.1.0",
         "shape": [int(indptr.size - 1), n_var]}
    )
    return session.commit(f"one virtual X over {len(paths)} files")


def open_matrix(repo_path: str, source_root: str):
    """Open the repo and return X as one scipy CSR matrix (rows read on demand)."""
    import icechunk as ic
    from scipy import sparse

    prefix = f"file://{os.path.abspath(source_root)}/"
    repo = ic.Repository.open(
        storage=ic.local_filesystem_storage(os.path.abspath(repo_path)),
        authorize_virtual_chunk_access=ic.containers_credentials({prefix: None}),
    )
    x = zarr.open_group(repo.readonly_session("main").store, mode="r")["X"]
    return sparse.csr_matrix(
        (x["data"][:], x["indices"][:], x["indptr"][:]), shape=tuple(x.attrs["shape"])
    )
