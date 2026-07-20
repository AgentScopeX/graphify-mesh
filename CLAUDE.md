# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`graphify-mesh` = scheduled sync engine + stdio MCP query server. Merges many per-repo [`graphify`](https://github.com/Graphify-Labs/graphify) knowledge graphs (`graph.json`) into one repo-attributed, human-named global graph, plus a cross-project overlay (depends_on / provides_api / consumes_api / similar_approach edges).

- pip name: `graphify-mesh`, import package: `graphify_mesh`
- console scripts: `graphify-mesh-sync`, `graphify-mesh-server`, `graphify-mesh-reap`
- upstream dependency `graphifyy` (PyPI name; import name `graphify`) — sync pipeline shells out to the `graphify` binary AND imports `graphify.build.distinct_repo_tags` directly at merge time, so it's a real import dependency, not just a CLI on PATH.

## Commands

```bash
pip install -e .                     # editable install (src-layout, pulls in graphifyy)
python -m pytest tests/              # full suite (pythonpath=src set in pyproject.toml, no install needed)
python -m pytest tests/sync/test_naming.py           # single test file
python -m pytest tests/sync/test_naming.py::test_x   # single test

graphify-mesh-sync --once --dry-run --mesh-root <dir> --scan-root <dir>   # dry run, writes nothing outside staging
graphify-mesh-sync --once --mesh-root <dir> --scan-root <dir>            # real run, publishes a generation
PYTHONPATH=src python -m graphify_mesh.server.server                     # run MCP server without install
```

No lint/format tooling configured in this repo.

## Architecture

Two independent halves under `src/graphify_mesh/`: `sync/` (the pipeline that builds generations) and `server/` (the stdio MCP server that reads the published generation).

### Sync pipeline (`sync/pipeline.py`)

One ordered pipeline per invocation, guarded by a whole-transaction lock (`sync/locking.py`) so runs never overlap:

```
discovery -> per-project sync -> merge -> recluster+remap -> name
  -> embed-changed -> overlay-resolve -> lexical-index -> validate -> atomic publish
```

- **discovery** (`sync/discovery.py`) walks `scan_root` for `graphify-out` symlinks, reconciles against `registry.json`, rejects symlinks resolving outside the approved root.
- **per-project sync** (`sync/sync_project.py`) diffs a source digest against saved state (`sync/state.py`) to pick `update` (AST-only) / `extract` (semantic) / `noop`; refuses unexpected shrinkage.
- **merge** always calls `graphify merge-graphs` with a deterministic, sorted-by-`repo_id` file list — **never** `graphify global add`, which is stateful and silently rewires/drops cross-repo edges on out-of-order re-adds. Rebuild-from-empty every run is what makes generations deterministic.
- **recluster + remap** (`sync/repo_tags.py`) re-clusters the merged graph and rewrites graphify's auto node-id tags (which collide across repos under the same product dir) to the true `repo_id`, before anything downstream runs.
- **name** (`sync/naming.py`) labels only communities whose membership fingerprint changed, via an Ollama backend; degrades to placeholder names if unreachable.
- **embed-changed** (`sync/embedding.py`, `sync/embed_similarity.py`) embeds only nodes whose durable content key changed; shards persisted/GC'd (`keep_embedding_generations`) only on successful publish.
- **overlay-resolve** (`sync/overlay.py`, `overlay_api.py`, `overlay_depends.py`, `overlay_refs.py`, `overlay_similar.py`) builds the cross-project overlay — this is the **only** place `depends_on`/`provides_api`/`consumes_api`/`similar_approach` edges get written; they never enter the structural graph.
- **validate** (`sync/validate.py`) blocks publish on structural problems, too-high stale-repo ratio (unless `--allow-shrink`), or failed backend-pin check.
- **atomic publish** (`sync/publish.py`) writes `global-graph.json`, `cross-project-overlay.json`, `lexical-index.json` into a fresh generation dir, then flips the `current` symlink — readers never see a partially-written generation.

Hard invariants (do not violate when touching this pipeline):
1. Structural graph never contains cross-repo edges — those live only in `cross-project-overlay.json`.
2. Every generation is rebuilt from empty, never incrementally merged — this is why `graphify global add` must never be called anywhere in this package.

### MCP server (`server/server.py`)

Stdio JSON-RPC 2.0, stdlib only (no `mcp` SDK), one process per client session, exits on stdin close. Advertised name: `graphify-mesh`. Fails closed (errors rather than serving stale/partial data) if no consistent generation has been published.

5 tools, each backed by its own module:
- `search` (`server/retrieval.py`, `server/ranking.py`) — hybrid lexical+vector+structural, scoped via `server/scope.py` (`current`/`all`/`repo:<id>`; fails closed if `current` can't resolve against `registry.json`).
- `cross_project` — explicit cross-repo search, optionally restricted to a repo list.
- `find_similar` (`server/similar.py`) — structural/semantic neighbors, optional `cross_repo_only`.
- `project_map` (`server/project_map.py`) — structural overview of one **registered** `repo_id` (never an arbitrary path).
- `context_pack` (`server/context_pack.py`) — evidence cards truncated to a token budget without splitting a card mid-way.

Note: graphify's own server (`graphify.serve`) can independently be repointed at the published `global-graph.json` for structural/PR tools (`god_nodes`, `shortest_path`, `get_pr_impact`, etc.) — that's a config change on graphify's side, not something this package implements. It reads `community_name` verbatim, while this package's `sync/naming.py` strips and relabels names during sync — so the same node can legitimately show a different community name in each server. This is expected, not a bug to reconcile.

### Config resolution

Everything resolves through `GRAPHIFY_MESH_*` env vars or CLI flags into `sync/config.py:Settings` (sync) / `server/config.py:ServerConfig` (server) — no machine-specific paths are hardcoded. Two vars keep upstream naming: `GRAPHIFY_BIN`, `GRAPHIFY_NO_BACKUP`. Full var/flag/field tables: `docs/configuration.md`.

`registry.json` is the source of truth for which repos are in the mesh (`repo_id`, `root`, `collection_path`, `enabled`); schema in `docs/configuration.md`. `manual-relations.json` holds human-declared overlay edges the sync engine can't infer, validated against `examples/manual-relations.schema.json` — dangling source/target refs are a hard error at load time.

### Tests

`tests/sync/`, `tests/server/`, `tests/hardening/` mirror the package layout. `pyproject.toml` sets `pythonpath = ["src"]` so `pytest tests/` works against the src-layout package without an install.

## Further reading

- `docs/architecture.md` — full pipeline detail and the two invariants above.
- `docs/configuration.md` — every env var, CLI flag, `Settings`/`ServerConfig` field, and the `registry.json`/`manual-relations.json` schemas.
- `docs/mcp-server.md` — full tool arg tables and JSON-RPC protocol.
- `docs/setup.md` — end-to-end setup walkthrough.

## Knowledge Base
project_id: agentscopex-graphify-mesh
