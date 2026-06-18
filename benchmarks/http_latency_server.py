"""A local HTTP file server with HTTP Range support and RANDOM per-request latency.

Adapted from ``experiments/sharding-icechunk/range_server.py`` (which already does
ranged serving + GET counting). Additions for the end-to-end latency benchmark:

  * a RANDOM per-request delay drawn from a LOGNORMAL distribution (``LatencyModel``):
    object-store GET RTT is well-modelled by a lognormal (unimodal hump + long right
    tail). Parameterised by the PDF mode (peak, the ``--latency-ms`` knob) and a
    log-scale shape ``--sigma`` (default 0.5). A deterministic ``--fixed`` mode is
    kept for the zero-latency reference. Every obstore ranged GET pays this real,
    wall-clock cost over a real HTTP stack.
  * threaded serving (``ThreadingHTTPServer``) so the concurrent ranged GETs that
    zarr/obstore fan out with ``asyncio.gather`` don't deadlock a single-threaded
    server -- each concurrent GET is handled on its own thread and sleeps in
    parallel, matching how a real object store behaves.
  * a server-side request counter (true number of GETs received) AND a peak
    concurrency counter (max requests simultaneously in flight), so we can report
    both the REAL source-GET count and the REAL concurrency obstore achieves --
    the basis for honest wall-clock interpretation (not a serial n*RTT assumption).

LOCAL loopback only (binds 127.0.0.1 on an ephemeral port); nothing leaves the
machine. Can be run as a CLI (prints the chosen port) or used in-process via the
``LatencyRangeServer`` class.

CLI:
    uv run python benchmarks/http_latency_server.py --latency-ms 45            # lognormal, mode 45ms
    uv run python benchmarks/http_latency_server.py --latency-ms 45 --sigma 0.7
    uv run python benchmarks/http_latency_server.py --latency-ms 20 --fixed    # deterministic
"""

from __future__ import annotations

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np


class LatencyModel:
    """Per-request latency, drawn from a LOGNORMAL distribution (or fixed).

    Object-store GET RTT is well described by a lognormal: a unimodal hump with a
    long right tail (a few slow fetches), unlike a shifted-exponential whose peak
    sits at the floor. We parameterise by the PDF **mode** (peak, the most-likely
    latency we tune) and a log-scale **shape** ``sigma``:

        mu        = ln(mode_ms) + sigma**2          # so the PDF mode == mode_ms
        sleep_ms  = LogNormal(mean=mu, sigma=sigma)  # numpy rng.lognormal(mu, sigma)
        sleep_s   = sleep_ms / 1000

    Derived stats: median_ms = exp(mu) = mode * exp(sigma**2);
    mean_ms = exp(mu + sigma**2 / 2).

    A deterministic mode (``random=False``, or ``mode_ms == 0``) sleeps exactly
    ``mode_ms`` and is the zero-latency / reference path.

    Parameters
    ----------
    mode_ms:
        Target PDF mode (peak) per-request latency, ms. The knob we sweep.
    sigma:
        Lognormal log-scale shape (skew). Default 0.5 (moderate skew).
    random:
        If True, draw lognormal; else sleep exactly ``mode_ms`` (deterministic).
    seed:
        Optional RNG seed for reproducible draws.
    """

    def __init__(self, mode_ms=0.0, *, sigma=0.5, random=True, seed=None):
        self.mode_ms = float(mode_ms)
        self.sigma = float(sigma)
        self.random = bool(random)
        self._rng = np.random.default_rng(seed)
        if self.mode_ms > 0.0:
            self.mu = np.log(self.mode_ms) + self.sigma**2
            self.median_ms = float(np.exp(self.mu))
            self.mean_ms = float(np.exp(self.mu + self.sigma**2 / 2.0))
        else:
            self.mu = float("nan")
            self.median_ms = 0.0
            self.mean_ms = 0.0

    def describe(self) -> dict:
        if not self.random or self.mode_ms == 0.0:
            return dict(kind="fixed", mode_ms=self.mode_ms)
        return dict(
            kind="lognormal",
            mode_ms=round(self.mode_ms, 4),
            sigma=self.sigma,
            mu=round(self.mu, 6),
            median_ms=round(self.median_ms, 4),
            mean_ms=round(self.mean_ms, 4),
            formula="mu=ln(mode)+sigma^2; sleep_ms=LogNormal(mu,sigma)",
        )

    def draw_s(self) -> float:
        if self.mode_ms <= 0.0:
            return 0.0
        if not self.random:
            return self.mode_ms / 1e3
        return float(self._rng.lognormal(self.mu, self.sigma)) / 1e3

    def realized_percentiles(self, n=50000) -> dict:
        """p10/p50/p90/p99 (ms) of an actual draw sample (honest distribution)."""
        if self.mode_ms <= 0.0 or not self.random:
            return dict(p10_ms=self.mode_ms, p50_ms=self.mode_ms,
                        p90_ms=self.mode_ms, p99_ms=self.mode_ms, n=n)
        rng = np.random.default_rng(12345)  # separate stream, doesn't perturb serving
        draws = rng.lognormal(self.mu, self.sigma, size=n)
        p = np.percentile(draws, [10, 50, 90, 99])
        return dict(
            p10_ms=round(float(p[0]), 3), p50_ms=round(float(p[1]), 3),
            p90_ms=round(float(p[2]), 3), p99_ms=round(float(p[3]), 3), n=n,
        )


class SharedPipe:
    """A global, thread-safe FIFO bandwidth limiter modelling ONE shared client pipe.

    Models the client's finite downlink: every request's BYTE TRANSFER is metered
    through a single pipe with total bandwidth ``B`` bytes/s, so aggregate egress can
    never exceed ``B`` no matter how many requests transfer concurrently. The
    per-request RTT (latency sleep) is handled separately and stays parallel; only the
    byte transfer contends here.

    FIFO reservation (the core of the model)::

        with lock:
            start = max(now, pipe_free)   # wait behind any in-progress transfer
            pipe_free = start + nbytes / B
            finish = pipe_free
        sleep(finish - now)               # outside the lock: transfers OVERLAP in
                                          # wall-time but their pipe-time serializes

    With C concurrent GETs the RTTs overlap, but the byte transfers queue for the pipe
    and share ``B`` -- so total throughput is capped at ``B`` and any OVER-READ (extra
    bytes a coalesced fetch pulls) directly costs pipe time. ``B is None`` (or <= 0)
    disables metering (unlimited bandwidth -- the deterministic reference path).
    """

    def __init__(self, bandwidth_bytes_per_s: float | None):
        self.B = (
            float(bandwidth_bytes_per_s)
            if bandwidth_bytes_per_s and bandwidth_bytes_per_s > 0
            else None
        )
        self._lock = threading.Lock()
        self._pipe_free = 0.0  # perf_counter timestamp the pipe is next free

    def transfer(self, nbytes: int) -> None:
        """Charge ``nbytes`` of transfer time through the shared pipe (blocking)."""
        if self.B is None or nbytes <= 0:
            return
        with self._lock:
            # Read the clock INSIDE the lock so a thread that stalls between
            # entering transfer() and acquiring the lock can't under-charge the
            # reservation against a stale `now`.
            now = time.perf_counter()
            start = max(now, self._pipe_free)
            self._pipe_free = start + nbytes / self.B
            finish = self._pipe_free
        delay = finish - time.perf_counter()
        if delay > 0:
            time.sleep(delay)

    def reset(self) -> None:
        # Reset under the lock so it can't race a concurrent reservation in
        # transfer() (which also reads/writes _pipe_free under the lock).
        with self._lock:
            self._pipe_free = time.perf_counter()


class LatencyRangeServer:
    """Threaded localhost HTTP server: Range support + random per-request latency.

    Each request sleeps a draw from a :class:`LatencyModel` (random distribution by
    default; fixed mode available) before responding, and the server tracks the max
    number of requests it saw simultaneously in flight (the real concurrency obstore
    achieves) so wall-clock can be interpreted honestly.

    A :class:`SharedPipe` additionally meters the BYTE TRANSFER of every response
    through one global bandwidth-limited pipe (``bandwidth_mbs`` MB/s), so over-read is
    charged real transfer time. The RTT sleep stays per-request (parallel); only bytes
    contend for the pipe. ``bandwidth_mbs=None`` (or 0) disables metering.

    Parameters
    ----------
    file_path:
        File to serve (its basename becomes the URL path).
    latency_ms:
        Target lognormal PDF MODE (peak) per-request latency, ms. Mutable via
        :meth:`set_latency_ms`. (Named ``latency_ms`` for the sweep API; it is the
        distribution's mode, not its mean -- see :class:`LatencyModel`.)
    random_latency:
        If True (default), draw each delay from the lognormal; else sleep exactly
        the mode (deterministic reference).
    sigma, seed:
        Passed to :class:`LatencyModel`.
    bandwidth_mbs:
        Total shared-pipe bandwidth in MB/s (1 MB = 1e6 bytes). ``None`` or 0 ->
        unlimited (no transfer metering; the deterministic reference). Mutable via
        :meth:`set_bandwidth_mbs`.
    """

    def __init__(
        self,
        file_path: str | Path,
        latency_ms: float = 0.0,
        *,
        random_latency: bool = True,
        sigma: float = 0.5,
        seed: int | None = None,
        bandwidth_mbs: float | None = None,
    ):
        self.file_path = Path(file_path)
        self.data = self.file_path.read_bytes()
        self.size = len(self.data)
        self._random_latency = random_latency
        self._sigma = sigma
        self._seed = seed
        self.set_bandwidth_mbs(bandwidth_mbs)  # sets self.bandwidth_mbs + self.pipe
        self.latency = LatencyModel(
            latency_ms, sigma=sigma, random=random_latency, seed=seed
        )
        self.ranges: list[tuple[int, int]] = []  # (start, end_inclusive) per GET
        self.n_requests = 0  # all methods (GET + HEAD)
        self.n_gets = 0  # GET only (the reads we care about)
        self._in_flight = 0  # currently-active requests
        self.max_in_flight = 0  # peak concurrency observed since last reset
        self._lock = threading.Lock()
        server_self = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"  # keep-alive; obstore reuses connections

            def log_message(self, *a):  # silence
                pass

            def _enter(self):
                with server_self._lock:
                    server_self._in_flight += 1
                    if server_self._in_flight > server_self.max_in_flight:
                        server_self.max_in_flight = server_self._in_flight

            def _leave(self):
                with server_self._lock:
                    server_self._in_flight -= 1

            def _sleep_latency(self):
                d = server_self.latency.draw_s()
                if d > 0:
                    time.sleep(d)

            def _send_full(self, body: bool):
                # Serve the whole file as a 200 (no Range, or an unparseable
                # multi-range we decline to split). Counts as a full-size read.
                if body:
                    with server_self._lock:
                        server_self.ranges.append((0, server_self.size - 1))
                self.send_response(200)
                self.send_header("Content-Length", str(server_self.size))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                if body:
                    server_self.pipe.transfer(server_self.size)
                    self.wfile.write(server_self.data)

            def _serve(self, body: bool):
                # Sleep BEFORE responding so the artificial latency is paid per
                # request (this is the whole point of the server).
                self._sleep_latency()
                rng = self.headers.get("Range")
                if rng and rng.startswith("bytes="):
                    spec = rng[len("bytes=") :]
                    if "," in spec:
                        # multi-range request: don't crash on int(); fall back to
                        # serving the full body (a valid 200 response per RFC 7233).
                        self._send_full(body)
                        return
                    start_s, _, end_s = spec.partition("-")
                    if start_s == "":
                        # suffix range: bytes=-N
                        n = int(end_s)
                        start = max(0, server_self.size - n)
                        end = server_self.size - 1
                    else:
                        start = int(start_s)
                        end = int(end_s) if end_s else server_self.size - 1
                    end = min(end, server_self.size - 1)
                    # Validate the (now-clamped) range: an unsatisfiable range
                    # (start beyond EOF, or start > end) must be a 416, not a
                    # malformed 206 carrying 0 bytes.
                    if start >= server_self.size or start > end:
                        self.send_response(416)
                        self.send_header(
                            "Content-Range", f"bytes */{server_self.size}"
                        )
                        self.send_header("Content-Length", "0")
                        self.send_header("Accept-Ranges", "bytes")
                        self.end_headers()
                        return
                    if body:
                        with server_self._lock:
                            server_self.ranges.append((start, end))
                    chunk = server_self.data[start : end + 1]
                    self.send_response(206)
                    self.send_header(
                        "Content-Range",
                        f"bytes {start}-{end}/{server_self.size}",
                    )
                    self.send_header("Content-Length", str(len(chunk)))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    if body:
                        # Charge byte-transfer time through the shared client pipe
                        # BEFORE writing: over-read directly costs pipe time.
                        server_self.pipe.transfer(len(chunk))
                        self.wfile.write(chunk)
                else:
                    self._send_full(body)

            def do_GET(self):
                with server_self._lock:
                    server_self.n_requests += 1
                    server_self.n_gets += 1
                self._enter()
                try:
                    self._serve(body=True)
                finally:
                    self._leave()

            def do_HEAD(self):
                with server_self._lock:
                    server_self.n_requests += 1
                self._enter()
                try:
                    self._serve(body=False)
                finally:
                    self._leave()

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/{self.file_path.name}"

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def set_latency_ms(self, latency_ms: float) -> None:
        """Re-arm the lognormal latency model at a new target MODE (same sigma)."""
        self.latency = LatencyModel(
            latency_ms,
            sigma=self._sigma,
            random=self._random_latency,
            seed=self._seed,
        )

    def set_bandwidth_mbs(self, bandwidth_mbs: float | None) -> None:
        """Re-arm the shared-pipe limiter at a new total bandwidth (MB/s)."""
        self.bandwidth_mbs = (
            float(bandwidth_mbs) if bandwidth_mbs and bandwidth_mbs > 0 else None
        )
        self.pipe = SharedPipe(
            None if self.bandwidth_mbs is None else self.bandwidth_mbs * 1e6
        )

    def describe(self) -> dict:
        """Server metadata (latency model + shared-pipe bandwidth)."""
        return dict(
            file=str(self.file_path),
            size_bytes=self.size,
            url=self.url,
            port=self.port,
            latency=self.latency.describe(),
            bandwidth_mbs=self.bandwidth_mbs,
            bandwidth_model=(
                "unlimited"
                if self.bandwidth_mbs is None
                else "shared client pipe (FIFO; aggregate egress capped at B bytes/s)"
            ),
        )

    def start(self) -> "LatencyRangeServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def reset_counts(self) -> None:
        with self._lock:
            self.ranges = []
            self.n_requests = 0
            self.n_gets = 0
            # Restart the peak from the current occupancy. Between sequential
            # cells the server is quiescent (_in_flight == 0); we leave
            # _in_flight itself alone (it is incremented on request start and
            # decremented in a finally, so it can't leak) so any genuinely
            # in-flight request still decrements correctly.
            self.max_in_flight = self._in_flight
        # Reset the shared pipe so a stale pipe_free from a prior run doesn't charge
        # phantom wait time at the start of the next measured read.
        self.pipe.reset()

    @property
    def total_bytes(self) -> int:
        return sum(e - s + 1 for s, e in self.ranges)

    def __enter__(self) -> "LatencyRangeServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--file",
        default=str(Path(__file__).resolve().parent.parent / "CMU-1.tiff"),
        help="file to serve (default: repo-root CMU-1.tiff)",
    )
    ap.add_argument(
        "--latency-ms",
        type=float,
        default=0.0,
        help="target lognormal MODE (peak) per-request delay in milliseconds",
    )
    ap.add_argument(
        "--fixed",
        action="store_true",
        help="deterministic latency (sleep exactly --latency-ms) instead of lognormal",
    )
    ap.add_argument(
        "--sigma",
        type=float,
        default=0.5,
        help="lognormal log-scale shape / skew (default 0.5); random mode only",
    )
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for draws")
    ap.add_argument(
        "--bandwidth-mbs",
        type=float,
        default=None,
        help="shared-pipe total bandwidth in MB/s (1 MB = 1e6 bytes); "
        "omit or 0 for unlimited (no transfer metering)",
    )
    args = ap.parse_args()

    server = LatencyRangeServer(
        args.file,
        latency_ms=args.latency_ms,
        random_latency=not args.fixed,
        sigma=args.sigma,
        seed=args.seed,
        bandwidth_mbs=args.bandwidth_mbs,
    ).start()
    try:
        print(f"serving {server.file_path} ({server.size} bytes)")
        print(f"server  : {server.describe()}")
        print(f"latency : {server.latency.describe()}")
        print(f"realized: {server.latency.realized_percentiles()}")
        print(f"url     : {server.url}")
        print(f"port    : {server.port}")
        print("Ctrl-C to stop.")
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        print(
            f"\nstopped. served {server.n_gets} GETs, {server.total_bytes} bytes, "
            f"peak concurrency {server.max_in_flight}."
        )


if __name__ == "__main__":
    main()
