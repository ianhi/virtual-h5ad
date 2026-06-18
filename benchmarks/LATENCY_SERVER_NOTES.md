# http_latency_server.py — notes

Our copy is an **aiohttp-based rewrite** of `../virtual/benchmarks/http_latency_server.py`.
The original hand-rolled `BaseHTTPRequestHandler` reimplemented HTTP Range parsing,
206/416/suffix-range handling, and loaded the whole file into RAM. This version
keeps only the benchmark-specific instrumentation and lets a tested backend own the
HTTP correctness. Changes worth porting back upstream:

1. **aiohttp `web.FileResponse` owns Range/206/416/conditional + streaming.** No
   custom range code. Correctness comes from a mature library, not us.

2. **Serves from disk, not RAM.** `self.size = path.stat().st_size`; `FileResponse`
   streams the requested range off disk. The original `read_bytes()` couldn't serve
   the 10 GB atlas. The injected latency is *added on top* of the (sub-ms, local-SSD)
   range read, so the modelled RTT stays dominated by the knob — accurate enough that
   we don't subtract disk time, but worth revisiting for spinning disks / huge ranges.

3. **`port` argument** (default 0 = ephemeral). Lets callers pin a port — needed to
   bind 80 to dodge the icechunk port-drop bug, or for a stable URL across runs.

4. **Async concurrency model.** One event loop; per-request `await asyncio.sleep`
   for latency overlaps naturally, and an `AsyncSharedPipe` (asyncio.Lock + loop
   clock) meters bandwidth. Cleaner than the thread-per-request + blocking sleeps of
   the original, and the peak-concurrency counter still works (verified: 16
   concurrent GETs → peak 16).

Kept from the original: the `LatencyModel` (lognormal, mode-parameterised) and the
shared-pipe bandwidth concept; the counters (`n_gets`, `ranges`/`total_bytes`,
`max_in_flight`); `set_latency_ms`/`set_bandwidth_mbs`/`reset_counts`; the
`__enter__/__exit__` context-manager API.

Remaining nits:
- Range is parsed a second time for *accounting only* (counters + bandwidth bytes);
  serving still uses aiohttp. A post-response byte count would avoid the small
  duplication but aiohttp doesn't expose it cleanly before `prepare()`.
- A single `stats()` snapshot (`{n_gets, max_in_flight, total_bytes}` under one lock)
  would be tidier than reading the fields individually while requests are in flight.
- Consumers must opt into plain HTTP (obstore `client_options={"allow_http": True}`,
  icechunk `http_store({"allow_http": "true"})`); documented in the module docstring.
