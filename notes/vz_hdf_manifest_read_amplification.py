# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "virtualizarr[hdf]==2.6.2",
#     "obstore==0.10.1",
#     "obspec-utils==0.9.0",
#     "h5py>=3.11",
#     "numpy>=2",
# ]
# ///
"""Minimal reproducer: VirtualiZarr's HDFParser reads ~the whole file to build a
*virtual* manifest of a chunk-dense HDF5 dataset.

Building a virtual manifest should touch only metadata: per chunk, the byte
offset + length in the file (HDF5 keeps these in the dataset's chunk index /
B-tree; for a filtered dataset the compressed size is stored there too). It
should NOT read chunk *data*. But HDFParser reads the file through obspec's
`BlockStoreReader` (1 MiB blocks, 64-block LRU). h5py's chunk-index walk issues
many tiny reads scattered across the whole file; each lands in a distinct 1 MiB
block and the small cache thrashes, so nearly every data byte is fetched to
extract a few KB of offsets.

We count bytes actually fetched from the store (cache-independent) by wrapping a
LocalStore in a tallying proxy, and compare against a native h5py `chunk_iter`
walk that produces the identical offsets/lengths.

Run:  uv run notes/vz_hdf_manifest_read_amplification.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser


def make_file(path: str, n: int = 4_000_000, chunk: int = 1_000) -> None:
    """A single 1-D gzip-compressed dataset with many chunks (dense B-tree)."""
    rng = np.random.default_rng(0)
    # mildly compressible so the file is a realistic fraction of raw size
    data = (rng.standard_normal(n) * 100).astype("int64")
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=data, chunks=(chunk,), compression="gzip")


class TallyStore:
    """Delegates to a real obstore store, tallying bytes returned by range reads.

    BlockStoreReader only needs `get_ranges` + a size lookup; we forward
    everything else. Byte tally is what the HDF read path actually fetched.
    """

    def __init__(self, inner: LocalStore) -> None:
        self._inner = inner
        self.bytes = 0
        self.calls = 0

    def get_ranges(self, path, starts, lengths):
        out = self._inner.get_ranges(path, starts=starts, lengths=lengths)
        self.bytes += sum(int(x) for x in lengths)
        self.calls += 1
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


def native_manifest(path: str, key: str):
    """Offsets/lengths straight from the chunk index — the metadata a manifest needs."""
    with h5py.File(path) as f:
        dsid = f[key].id
        n = dsid.get_num_chunks()
        offs = np.empty(n, "int64")
        lens = np.empty(n, "int64")
        i = [0]

        def cb(blob):
            k = i[0]
            offs[k], lens[k] = blob.byte_offset, blob.size
            i[0] += 1

        dsid.chunk_iter(cb)
    return offs, lens


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "dense.h5")
        make_file(path)
        size = Path(path).stat().st_size
        with h5py.File(path) as f:
            nchunks = f["x"].id.get_num_chunks()
        print(f"file: {size/1e6:.1f} MB, dataset x: {nchunks} chunks\n")

        # --- native chunk_iter: the metadata a manifest needs ---
        offs, lens = native_manifest(path, "x")
        meta_bytes = offs.nbytes + lens.nbytes
        print(f"manifest metadata (offsets+lengths): {meta_bytes/1e3:.1f} KB "
              f"for {len(offs)} chunks")

        # --- HDFParser via a tallying LocalStore ---
        tally = TallyStore(LocalStore(prefix=d))
        registry = ObjectStoreRegistry({f"file://{d}": tally})
        url = f"file://{path}"
        vds = open_virtual_dataset(url=url, registry=registry, parser=HDFParser())
        ma = vds["x"].data
        m = ma.manifest

        # correctness: HDFParser and native agree on the byte ranges
        same = (int(m._offsets[(0,)]) == offs[0]) and (int(m._lengths[(0,)]) == lens[0])

        print(f"\nHDFParser fetched from store: {tally.bytes/1e6:.1f} MB "
              f"in {tally.calls} get_ranges calls")
        print(f"  -> {tally.bytes/size:.0%} of the {size/1e6:.1f} MB file, "
              f"to extract {meta_bytes/1e3:.1f} KB of offsets")
        print(f"  -> {tally.bytes/meta_bytes:.0f}x the metadata it needed")
        print(f"  manifest matches native chunk_iter: {same}")


if __name__ == "__main__":
    main()
