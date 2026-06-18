# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk==2.0.6",
#   "zarr==3.2.1",
# ]
# ///
"""Minimal, self-contained repro: icechunk drops the port from virtual-chunk URLs.

A virtual chunk reference stored with location `http://127.0.0.1:PORT/data.bin`
reads back as `http://127.0.0.1/data.bin` (port dropped), so it no longer matches
its VirtualChunkContainer prefix (which keeps the port) and reads fail with
"no virtual chunk container can handle the chunk location".

HTTPS CDNs are unaffected (port 443 is the default, nothing to drop); this breaks
any HTTP(S) source on a non-default port — localhost dev servers, MinIO, custom
S3 endpoints. Discovered while benchmarking against a local latency HTTP server.

Run standalone (no project env needed):

    uv run experiments/icechunk_port_drop_repro.py
"""

import icechunk as ic
import zarr

PORT_URL = "http://127.0.0.1:53999/"
LOCATION = "http://127.0.0.1:53999/data.bin"  # explicit, non-default port

cfg = ic.RepositoryConfig.default()
cfg.set_virtual_chunk_container(
    ic.VirtualChunkContainer(PORT_URL, ic.http_store({"allow_http": "true"}))
)
repo = ic.Repository.create(
    storage=ic.local_filesystem_storage("/tmp/icechunk_port_drop"),
    config=cfg,
    authorize_virtual_chunk_access=ic.containers_credentials({PORT_URL: None}),
)
session = repo.writable_session("main")
store = session.store
arr = zarr.open_group(store, mode="a").create_array(
    name="x", shape=(10,), dtype="float32", chunks=(10,)
)
store.set_virtual_refs(
    array_path=arr.path,
    chunks=[ic.VirtualChunkSpec(index=[0], location=LOCATION, offset=0, length=40)],
    validate_containers=False,
)
session.commit("one virtual ref with an explicit-port http URL")

try:
    zarr.open_group(repo.readonly_session("main").store, mode="r")["x"][0:1]
    print("read succeeded (port preserved) — bug not reproduced")
except ic.IcechunkError as e:
    loc = str(e).split("location (")[-1].split(")")[0]
    print(f"stored location read back as: {loc}")
    print(f"expected:                    {LOCATION}")
    assert loc == "http://127.0.0.1/data.bin", "expected the port to be dropped"
    print("=> icechunk dropped the port from the virtual-chunk location URL")
