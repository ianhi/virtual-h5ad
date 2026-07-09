# /// script
# requires-python = ">=3.11"
# dependencies = ["icechunk==2.0.6"]
# ///
"""Icechunk footgun: `repo.config.set_*(); repo.save_config()` silently no-ops.

`repo.config` returns a fresh COPY each access, and `save_config()` persists the
repo's *internal* config (the one it was opened/created with) — not your edited
copy. So the natural "mutate then save" idiom silently discards the change, with
no error. A user can end up with a committed repo missing the virtual-chunk
container they thought they added (→ virtual refs later resolve to nothing).

Run:  uv run notes/icechunk_config_save_footgun.py
"""

import os
import tempfile

import icechunk as ic


def persisted_containers(storage):
    """Reopen from scratch and report what's actually on disk."""
    return ic.Repository.open(storage=storage).config.virtual_chunk_containers


def main() -> None:
    d = tempfile.mkdtemp()
    storage = ic.local_filesystem_storage(os.path.join(d, "repo"))
    ic.Repository.create(storage=storage)

    prefix = f"file://{d}/src/"
    vcc = ic.VirtualChunkContainer(prefix, ic.local_filesystem_store(f"{d}/src/"))

    # `repo.config` is a fresh copy every time — the root of the footgun.
    repo = ic.Repository.open(storage=storage)
    print("repo.config is repo.config :", repo.config is repo.config)  # -> False

    # ---- THE TRAP: mutate the copy, then save --------------------------------
    repo.config.set_virtual_chunk_container(vcc)
    repo.save_config()
    print("after mutate-copy + save_config :", persisted_containers(storage))
    # -> None   (silently discarded; no error)

    # ---- THE FIX: set on a config, pass it to open, THEN save ----------------
    cfg = ic.RepositoryConfig.default()
    cfg.set_virtual_chunk_container(vcc)
    repo2 = ic.Repository.open(storage=storage, config=cfg)  # cfg becomes the internal config
    repo2.save_config()                                      # persists it
    print("after open(config=cfg) + save_config :",
          list(persisted_containers(storage) or {}))
    # -> ['file://.../src/']   (correctly persisted)


if __name__ == "__main__":
    main()
