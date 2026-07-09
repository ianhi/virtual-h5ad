"""Under real network latency, does anndata's native read already parallelize — and
does tuning zarr's concurrency make a hand-rolled thread-pool reader redundant?

We point an Icechunk repo's virtual chunks at a local HTTP server that injects real
per-request latency and counts GETs + peak concurrency. The server is `snailmail`'s
`HTTPRangeServer` (the shared benchmark server — not a hand-rolled one). Each read
runs in a FRESH process (icechunk caches fetched chunks process-globally, so a warm
process would read zero bytes), and the server reports the REAL peak concurrency.

We compare, sweeping the relevant concurrency knob for each:
  - adata[mask].to_memory()           at several zarr `async.concurrency` values
  - a thread-pool run-coalescing read  at several worker counts (`_threadpool_select`)

The honest question: anndata's native read fans out via zarr's codec pipeline, but
that is bounded by `async.concurrency` (default 10). If raising it closes the gap to
the thread pool, the thread pool is redundant (just tune zarr). Result: it does close
the gap — which is why `select_cells` was dropped from the package and only this
benchmark keeps the thread-pool reader, as the baseline it's measured against.

Requires a patched icechunk (the HTTP virtual-chunk port fix) and:
    uv run --no-sync python latency_benchmark.py
(`--no-sync` stops uv re-pinning icechunk to the locked PyPI build.)
"""

import multiprocessing as mp
import os
import time

import numpy as np

from snailmail import HTTPRangeServer, LogNormal

SRC = os.path.abspath("data/allen/other_cells.h5ad")
REPO = os.path.abspath("repo/latency_bench")
RTT_MODE_MS = 30.0
N_CELLS = 200


def _native(repo, idx, q, zarr_concurrency):
    import zarr

    zarr.config.set({"async.concurrency": zarr_concurrency})
    from virtual_h5ad import open_anndata

    adata = open_anndata(repo)
    mask = np.zeros(adata.n_obs, dtype=bool)
    mask[idx] = True
    t = time.perf_counter()
    sub = adata[mask].to_memory()
    q.put((time.perf_counter() - t, int(sub.X.nnz)))


def _threadpool_select(repo_path, obs_indices, workers):
    """Run-coalescing thread-pool row gather — the baseline the native read is measured
    against. Removed from the package (`adata[mask].to_memory()` + zarr concurrency
    matches it); kept here only to reproduce the comparison."""
    from concurrent.futures import ThreadPoolExecutor

    from scipy import sparse

    from virtual_h5ad._concat import MATRIX_GROUP, _open_root

    x = _open_root(repo_path)[MATRIX_GROUP]
    indptr = x["indptr"][:]  # small (one int per cell); the bulk stays remote
    data, indices = x["data"], x["indices"]

    order = np.argsort(np.asarray(obs_indices), kind="stable")
    rows = np.asarray(obs_indices)[order]
    breaks = np.flatnonzero(np.diff(rows) != 1)  # contiguous runs -> one read each
    run_bounds = zip(np.r_[0, breaks + 1], np.r_[breaks, rows.size - 1])

    def fetch(bounds):  # read a whole run's bytes in a single range request
        r0, r1 = int(rows[bounds[0]]), int(rows[bounds[1]])
        a, b = int(indptr[r0]), int(indptr[r1 + 1])
        return data[a:b], indices[a:b], indptr[r0 : r1 + 2] - a

    cells: list = [None] * rows.size  # per-cell (data, indices), in sorted order
    sorted_pos = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for d_run, i_run, ip in ex.map(fetch, run_bounds):
            for k in range(len(ip) - 1):
                s, e = int(ip[k]), int(ip[k + 1])
                cells[sorted_pos] = (d_run[s:e], i_run[s:e])
                sorted_pos += 1

    inv = np.empty(rows.size, dtype=np.intp)  # restore the caller's row order
    inv[order] = np.arange(rows.size)
    rows_d, rows_i, new_indptr = [], [], [0]
    for p in inv:
        d, i = cells[p]
        rows_d.append(d)
        rows_i.append(i)
        new_indptr.append(new_indptr[-1] + d.size)
    return sparse.csr_matrix(
        (np.concatenate(rows_d), np.concatenate(rows_i), np.array(new_indptr)),
        shape=(len(obs_indices), int(x.attrs["shape"][1])),
    )


def _custom(repo, idx, q, workers):
    t = time.perf_counter()
    x = _threadpool_select(repo, idx, workers)
    q.put((time.perf_counter() - t, int(x.nnz)))


def run_in_fresh_process(target, idx, server, knob):
    """Run a read in a cold-cache subprocess; return (wall, nnz) + server stats."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    server.reset_counts()
    p = ctx.Process(target=target, args=(REPO, idx, q, knob))
    p.start()
    wall, nnz = q.get()
    p.join()
    st = server.stats()
    return wall, nnz, st["n_gets"], st["max_in_flight"], st["total_bytes"]


def main():
    from virtual_h5ad import open_anndata, virtualize_matrix

    server = HTTPRangeServer.from_file(SRC).start()  # build at zero latency
    src_url = server.url(os.path.basename(SRC))
    try:
        print("server:", src_url)
        virtualize_matrix(src_url, REPO, download=True)

        adata = open_anndata(REPO)
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(adata.n_obs, size=N_CELLS, replace=False))
        runs = 1 + int(np.count_nonzero(np.diff(idx) != 1))
        del adata

        server.set_latency(LogNormal(mode_ms=RTT_MODE_MS))
        print(f"latency: {server.describe()}")
        print(f"selection: {N_CELLS} scattered cells in {runs} runs\n")

        # (label, target, knob) — knob is zarr async.concurrency for native, workers for custom
        cases = [
            ("native  zarr.conc=10", _native, 10),
            ("native  zarr.conc=40", _native, 40),
            ("native  zarr.conc=128", _native, 128),
            ("native  zarr.conc=256", _native, 256),
            ("threadpool  w=32", _custom, 32),
            ("threadpool  w=128", _custom, 128),
        ]
        for label, target, knob in cases:
            wall, nnz, gets, peak, nbytes = run_in_fresh_process(target, idx, server, knob)
            assert nnz == 155436, f"{label}: wrong nnz {nnz}"
            print(
                f"{label:22s} wall={wall:6.2f}s  GETs={gets:4d}  "
                f"peak_concurrency={peak:3d}  bytes={nbytes/1e6:5.2f}MB"
            )
    finally:
        server.stop()


if __name__ == "__main__":
    main()
