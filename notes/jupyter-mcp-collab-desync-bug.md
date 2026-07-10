# Bug report: Jupyter MCP edits silently lost with RTC / multiple collab servers

## Summary
Editing a notebook through the Jupyter MCP (`update_cell` / `insert_cell` /
`delete_cell` / `execute_cell`) reported success and was reflected by the MCP's own
`get_notebook_content`, but the changes **never reached the `.ipynb` on disk** and were
**never visible in the browser**. All work (a full multi-cell rewrite + executed
outputs) was effectively lost from the user's point of view.

## Environment
- jupyter-server 2.20.0, jupyter-collaboration 4.4.1, jupyter_ydoc 3.5.0,
  pycrdt 0.13.1, jupyterlab 4.5.9 (RTC/collaboration ON, SQLite YStore).
- macOS. Notebook launched via `uv run --with jupyterlab --with jupyter-collaboration ...`.
- **Multiple JupyterLab collab servers running at once**, rooted at different dirs:
  one at `~/Documents/dev/bio-virtualization/` (higher) and one at
  `.../virtual-h5ad/` (project). Each has its own `.jupyter_ystore.db`.
- Server was also restarted several times mid-session (ports 8970 -> 8922 -> 8907).

## Symptoms / evidence
- Every MCP edit returned: "No browser peers connected to this notebook room — the edit
  reached the server but the user's tab may not be showing it." (Consistent, every call.)
- MCP `get_notebook_content` returned the EDITED notebook (11 cells, new content +
  executed outputs) while a direct `json.load()` of the `.ipynb` on disk returned the
  ORIGINAL (13 cells, no edits). Persistent divergence.
- The divergence survived an MCP reconnect across a server restart: after reconnecting
  to the new port, `get_notebook_content` still returned the edited (cached room)
  content, not the on-disk original — i.e. the MCP served a stale/held YDoc, masking
  that nothing had persisted.
- The project `.jupyter_ystore.db` keys rooms by fileId UUID (`.notebook:<uuid>.y`), and
  a second ystore exists one directory up — consistent with two servers minting
  different fileIds/rooms for the same file.

## Root-cause hypothesis
With jupyter-collaboration, a notebook is edited in a Yjs **room** that the server
persists to `.ipynb` via the ContentsManager on save/autosave. The MCP joined/created a
room that was **not the fileId-bound document room being persisted** (or was on a
*different* collab server than the browser). Its writes accumulated in a room with **no
browser peer and no room->file save sink**, so they never hit disk. Contributing
factors:
1. **Multiple collab servers on overlapping roots** — MCP on server A, browser on
   server B; rooms don't cross servers; same file seeded independently.
2. **MCP holds a client-side YDoc across reconnects** and answers reads from it, so
   `get_notebook_content` can't be trusted to reflect disk after a server restart.
3. No surfaced error — "no browser peers" is a soft warning, not a failure, so edits
   look successful.

## Suggested repro
1. Start two `jupyter lab` servers with jupyter-collaboration, roots A (parent) and
   B (child), both able to serve `child/nb.ipynb`.
2. Open `nb.ipynb` in a browser on server A.
3. Point the MCP at server B (or the reverse) and `update_cell`.
4. Observe: MCP `get_notebook_content` shows the edit; browser and disk do not.

## Suggested fixes / mitigations (MCP side)
- On `connect`/every read, **verify the MCP's room is the server's fileId-bound document
  room** and that a room->file save sink is active; else refuse or warn hard.
- After an edit, **confirm persistence** (round-trip: trigger a document save and re-read
  the file), and surface a real error if the file didn't change.
- Treat "no browser peers" plus "no persistence sink" as a **hard** error, not a warning.
- Detect and refuse when multiple collab servers could serve the same path, or bind by
  absolute file path + server base URL and validate.
- On reconnect, **re-sync from the server's persisted document** rather than serving a
  cached YDoc.

## Workaround that recovered the work
Wrote the rebuilt notebook JSON directly to the `.ipynb` on disk (bypassing the MCP/RTC),
then: stop the extra server(s), keep one server rooted at the project dir, move the stale
`.jupyter_ystore.db` aside so the room re-seeds from the on-disk file, restart, reopen.
