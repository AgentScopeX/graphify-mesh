# Setup

Step-by-step: get from zero to a merged, queryable global graph across your own
repos. Every path below is a placeholder — substitute your own.

## 1. Prerequisites

- **Python 3.11+**
- **The upstream `graphify` CLI.** graphify-mesh shells out to the `graphify`
  binary and imports `graphify.build` at merge time. It is **not** on PyPI, so
  install it separately:

  ```bash
  pipx install graphifyy      # provides the `graphify` command
  graphify --help             # confirm it is on PATH
  ```

  If `graphify` is not on `PATH` for the environment that runs the sync (e.g. a
  systemd service), set `GRAPHIFY_BIN` to its absolute path.

- **(Optional) An Ollama host** for the community-naming and embedding stages.
  Both stages degrade gracefully when Ollama is unreachable (communities keep
  placeholder names; search falls back to lexical + structural only).

## 2. Install graphify-mesh

```bash
pip install graphify-mesh
# or, from a checkout:
pip install -e .
```

This provides three console scripts: `graphify-mesh-sync`,
`graphify-mesh-server`, and `graphify-mesh-reap`.

### Installing from the project's own package index

graphify-mesh is published to a PEP 503 "simple" index hosted on GitHub Pages
(see [`publishing.md`](publishing.md) for why this is used instead of GitHub
Packages). Once Pages is enabled on the repo, install a specific version with:

```bash
pip install \
  --index-url https://agentscopex.github.io/graphify-mesh/simple/ \
  graphify-mesh==0.1.0
```

The exact URL scheme is:

- **Index root:** `https://agentscopex.github.io/graphify-mesh/simple/`
- **Project page:** `https://agentscopex.github.io/graphify-mesh/simple/graphify-mesh/`
- Each release's wheel and sdist are linked from the project page, pointing at
  the GitHub Release asset download URLs.

## 3. Point `graphify` at each repo

For every repo you want in the mesh, run `graphify` once so it produces a
`graphify-out/graph.json`, and make sure that output is reachable from a single
scan root (a `graphify-out` symlink per checkout is the convention). Example
layout:

```
/path/to/your/workspace/checkouts/
    backend-a/graphify-out    -> ../../graph-mesh/graphify/example-org/backend-a
    frontend-b/graphify-out   -> ../../graph-mesh/graphify/example-org/frontend-b
```

## 4. Write your registry.json

Copy the example and edit it to describe *your* repos:

```bash
cp examples/registry.example.json \
   /path/to/your/workspace/graph-mesh/bin/registry.json
$EDITOR /path/to/your/workspace/graph-mesh/bin/registry.json
```

Each entry needs a stable `repo_id`, the checkout `root`, and the
`collection_path` where that repo's `graph.json` lives. See
[`configuration.md`](configuration.md#registryjson) for the full schema.

## 5. First dry run

A dry run prints every action and writes nothing outside a private staging
directory — safe to run anywhere:

```bash
graphify-mesh-sync --once --dry-run \
  --mesh-root  /path/to/your/workspace/graph-mesh \
  --scan-root  /path/to/your/workspace/checkouts
```

Review the JSON report: `reconciliation`, `project_actions`, `stale_repos`,
`merge_ok`, and `validation_ok` tell you what a real run would do.

## 6. First real run

```bash
graphify-mesh-sync --once \
  --mesh-root  /path/to/your/workspace/graph-mesh \
  --scan-root  /path/to/your/workspace/checkouts
```

On success this publishes a new immutable generation and flips
`<mesh_root>/graphify/global/current` to point at it.

You can also drive everything from env vars instead of flags — see
[`configuration.md`](configuration.md). To run on a schedule, adapt the units in
`examples/systemd/` (edit every `/path/to/...` placeholder first).

## 7. Register the MCP server

`graphify-mesh-server` speaks newline-delimited JSON-RPC 2.0 over stdio, one
process per client session. Register it with your MCP-capable client:

```json
{
  "mcpServers": {
    "graphify-mesh": {
      "command": "graphify-mesh-server",
      "env": { "GRAPHIFY_MESH_ROOT": "/path/to/your/workspace/graph-mesh" }
    }
  }
}
```

See [`mcp-server.md`](mcp-server.md) for the tools and protocol details.
