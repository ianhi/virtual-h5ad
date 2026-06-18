# Notes on http_latency_server.py (usability feedback)

Copied from `../virtual/benchmarks/http_latency_server.py` and used to benchmark
Icechunk virtual reads under injected latency. It worked well; the model
(lognormal latency mode + shared-pipe bandwidth + GET/concurrency counters) is
exactly what's needed for honest wall-clock interpretation. Things that would
have made it easier to use:

1. **Serves only from RAM.** `self.data = self.file_path.read_bytes()` loads the
   whole file, so it can't serve the 10 GB atlas — I had to fall back to a 56 MB
   file. Add an on-disk mode that `pread`s the requested range from the file.
   *Then subtract the real disk-read time from the injected latency* so the
   promised RTT stays honest (a range read of a cold 10 GB file isn't free).

2. **Ephemeral port is hardcoded** (`("127.0.0.1", 0)`). Expose a `port` arg on
   `LatencyRangeServer.__init__` and the CLI. Needed to (a) pin a stable URL and
   (b) work around consumers that mishandle non-default ports — e.g. Icechunk
   drops the port from virtual-chunk location URLs (see
   `experiments/icechunk_port_drop_repro.py`), so a virtual repo pointed at
   `http://127.0.0.1:<ephemeral>/...` is unreadable; only port 80 would match.

3. **plain-HTTP gotcha.** Consumers must opt into `http://` explicitly
   (obstore: `client_options={"allow_http": True}`; icechunk:
   `http_store({"allow_http": "true"})`). Worth a one-line note in the docstring,
   since the default failure (`reqwest BadScheme`) is opaque.

4. **In-process usage is the good path but under-documented.** The module
   docstring leads with the CLI; the `LatencyRangeServer` class (with
   `__enter__/__exit__`, `reset_counts()`, `set_latency_ms()`,
   `set_bandwidth_mbs()`, and the `n_gets`/`max_in_flight`/`total_bytes`
   counters) is what a benchmark actually wants. A short in-process example near
   the top would help.

5. **A `stats()` snapshot** returning `{n_gets, max_in_flight, total_bytes}` under
   one lock would be tidier than reading the fields individually (they can move
   between reads while requests are in flight).
