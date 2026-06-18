"""Under real network latency, does anndata's native read already parallelize?

We point an Icechunk repo's virtual chunks at a local HTTP server that injects
real per-request latency and counts GETs + peak concurrency
(benchmarks/http_latency_server.py). Then we compare, each in a FRESH process
(icechunk caches fetched chunks process-globally, so a warm process would read
zero bytes from the server):

  - adata[mask].to_memory()   # anndata's own coalescing backed reader
  - select_cells(repo, idx)   # our custom coalesce + thread pool

The server reports the REAL GET count and peak concurrency, so we can see whether
anndata already fans out concurrent GETs (peak > 1) and matches our wall-clock —
in which case select_cells is redundant.

Run from the project root:  uv run python latency_benchmark.py

STATUS: blocked by an Icechunk bug. Icechunk drops the port from virtual-chunk
location URLs (see experiments/icechunk_port_drop_repro.py), so a repo pointed at
this server's `http://127.0.0.1:<ephemeral>/...` is unreadable — no container
matches the port-less location. To run this end-to-end, either fix Icechunk or
serve on port 80 (needs root). Meanwhile the concurrency question is answered by
the read call stack: anndata's adata[mask] goes through zarr's codec pipeline
(codec_pipeline.read -> concurrent_map -> asyncio.gather), i.e. it already fans
out concurrent async chunk reads — so select_cells' thread pool is redundant.
"""

import multiprocessing as mp
import os
import time

import numpy as np

from benchmarks.http_latency_server import LatencyRangeServer

SRC = os.path.abspath("data/allen/other_cells.h5ad")
REPO = os.path.abspath("repo/latency_bench")
RTT_MODE_MS = 30.0
N_CELLS = 200


def _native(repo, idx, q):
    from tutorial.rectilinear_concat import open_anndata

    adata = open_anndata(repo)
    mask = np.zeros(adata.n_obs, dtype=bool)
    mask[idx] = True
    t = time.perf_counter()
    sub = adata[mask].to_memory()
    q.put((time.perf_counter() - t, int(sub.X.nnz)))


def _custom(repo, idx, q):
    from tutorial.rectilinear_concat import select_cells

    t = time.perf_counter()
    x = select_cells(repo, idx)
    q.put((time.perf_counter() - t, int(x.nnz)))


def run_in_fresh_process(target, idx, server):
    """Run a read in a cold-cache subprocess; return (wall, nnz) + server stats."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    server.reset_counts()
    p = ctx.Process(target=target, args=(REPO, idx, q))
    p.start()
    wall, nnz = q.get()
    p.join()
    return wall, nnz, server.n_gets, server.max_in_flight, server.total_bytes


def main():
    from tutorial.rectilinear_concat import open_anndata, virtualize_matrix

    server = LatencyRangeServer(SRC, latency_ms=0.0).start()  # build at zero latency
    try:
        print("server:", server.url)
        virtualize_matrix(server.url, REPO, download=True)

        adata = open_anndata(REPO)
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(adata.n_obs, size=N_CELLS, replace=False))
        runs = 1 + int(np.count_nonzero(np.diff(idx) != 1))
        del adata

        server.set_latency_ms(RTT_MODE_MS)
        print(f"latency: {server.latency.describe()}")
        print(f"realized: {server.latency.realized_percentiles()}")
        print(f"selection: {N_CELLS} scattered cells in {runs} runs\n")

        for label, target in [("adata[mask].to_memory()", _native), ("select_cells", _custom)]:
            wall, nnz, gets, peak, nbytes = run_in_fresh_process(target, idx, server)
            print(
                f"{label:26s} wall={wall:6.2f}s  nnz={nnz}  GETs={gets:4d}  "
                f"peak_concurrency={peak:3d}  bytes={nbytes/1e6:5.2f}MB"
            )
    finally:
        server.stop()


if __name__ == "__main__":
    main()
