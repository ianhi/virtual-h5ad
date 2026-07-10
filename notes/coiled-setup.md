# Running the demo on Coiled (fork icechunk + coalescing-zarr)

Scope: how to run `virtual_omics_demo.ipynb` on a Coiled notebook given the
non-released install (forked icechunk with `get_many_chunks`, coalescing-zarr from
git). Covers *why* the mac→Linux gap is a non-issue and the exact recipe.

## Why not Coiled package sync ("venv sync")

Package sync (the default for `coiled notebook start`) mirrors your **local** env.
Locally icechunk is the **macOS** wheel, and the fork isn't on any package index —
so package sync has no Linux build to resolve to and can't reproduce it on the
worker. Use an **explicit software environment** instead (below). `--sync` is still
useful, but for *code* (mutagen), not packages.

## Why mac→Linux doesn't matter

You never compile the fork for Coiled. The fork's CI publishes **prebuilt
`cp312-abi3` wheels for every platform** on the `fork-coalescing-wip` release
(macOS x86_64/arm64, manylinux x86_64/aarch64, musllinux, win). Locally, uv picks
the mac wheel (see this repo's `[tool.uv.sources]`); on Coiled you point the
software env at the **Linux manylinux wheel URL**. Different wheels, same commit,
zero local build.

## Recipe — prebuilt wheel (recommended, no Rust)

Coiled workers are Linux x86_64 / Python 3.12. List the icechunk wheel **first**.

```python
import coiled

coiled.create_software_environment(
    name="virtual-omics",
    workspace="earthmover-devs",          # your workspace
    pip=[
        # forked icechunk — Linux x86_64 wheel, first so it pins the resolve
        "https://github.com/ianhi/icechunk/releases/download/fork-coalescing-wip/"
        "icechunk-2.1.0-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "coalescing-zarr @ git+https://github.com/ianhi/coalescing-zarr",
        "virtual-h5ad @ git+https://github.com/ianhi/virtual-h5ad",  # once pushed
        "arraylake", "scanpy", "jupyterlab",   # arraylake is NOT a runtime dep of virtual-h5ad
    ],
)
```

Launch + sync the demo notebook (needs `mutagen` locally), then open it on the VM:

```sh
coiled notebook start --software virtual-omics --sync \
  --vm-type c7a.large --idle-timeout 15m --name virtual-omics
```

On the VM, authenticate Arraylake in a cell (or a terminal), then run the notebook:

```
!arraylake auth login       # interactive; or set ARRAYLAKE_TOKEN via --env
```

Tear down: `coiled notebook stop --name virtual-omics`.

## Fallback — build the fork from source on Coiled (no wheel for a platform)

This is what the icechunk benchmarks did (`icechunk-python/benchmarks/runner.py`):
put **`rust` in the conda deps** so pip can compile the git URL on Coiled's Linux
build host. Slower env build; only needed if no prebuilt wheel fits the target.

```python
coiled.create_software_environment(
    name="virtual-omics-src",
    conda={"channels": ["conda-forge"], "dependencies": ["rust", "python=3.12", "pip"]},
    pip=[
        "git+https://github.com/ianhi/icechunk@ian/more-specific-vritual#subdirectory=icechunk-python",
        "coalescing-zarr @ git+https://github.com/ianhi/coalescing-zarr",
        "virtual-h5ad @ git+https://github.com/ianhi/virtual-h5ad",
        "arraylake", "scanpy", "jupyterlab",
    ],
)
```

## Notes

- `create_software_environment` is idempotent — repeated identical calls are a
  no-op; pass `force_rebuild=True` to force.
- `virtual-h5ad` must be pushed to GitHub for the `git+` line to resolve. Until
  then, `--sync` this repo and `pip install -e .` on the VM instead.
- Historical: past Coiled use in sibling repos (`arraylake-token-caching`,
  `icechunk-ingest`) used `create_software_environment` with **released** icechunk;
  fork-on-Coiled via prebuilt wheels (this recipe) is new and didn't exist then.
