# MCP server

`graphify-mesh-server` is a stdio MCP server implemented with the Python
standard library only (no `mcp` SDK). It speaks newline-delimited JSON-RPC 2.0
over stdin/stdout, one process per client session, and exits cleanly the moment
the client closes stdin.

The advertised server name is **`graphify-mesh`**.

## Protocol

- Transport: newline-delimited JSON-RPC 2.0 objects on stdin/stdout.
- Methods: `initialize`, `tools/list`, `tools/call`.
- Lifecycle: the client starts the process, exchanges messages, then closes
  stdin; the server then exits with code 0. It is never a long-lived shared
  daemon.

Minimal handshake:

```json
{"jsonrpc": "2.0", "id": 1, "method": "initialize"}
{"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
{"jsonrpc": "2.0", "id": 3, "method": "tools/call",
 "params": {"name": "search", "arguments": {"q": "auth guard", "scope": "current"}}}
```

If no consistent generation has been published yet, tool calls fail closed
(an error result), rather than serving a partial or stale graph.

## The 5 tools

### `search`
Hybrid lexical + vector + structural search within a **scope**. Scope defaults
to the current project and only widens when asked. Fails closed if
`scope="current"` cannot be resolved against `registry.json`.

| Arg | Type | Default | Notes |
|-----|------|---------|-------|
| `q` | string | — (required) | Query text. |
| `scope` | string | `current` | `current`, `all`, or `repo:<id>`. |
| `k` | integer | ranking default | Max results. |

### `cross_project`
Explicit cross-repo hybrid search, optionally restricted to a list of repos.

| Arg | Type | Default | Notes |
|-----|------|---------|-------|
| `q` | string | — (required) | Query text. |
| `repos` | string[] | all repos | Restrict to these `repo_id`s. |
| `k` | integer | ranking default | Max results. |

### `find_similar`
Structurally / semantically similar nodes to a given node or label; can be
restricted to cross-repo matches only.

| Arg | Type | Default | Notes |
|-----|------|---------|-------|
| `node` | string | — (required) | Node id or label. |
| `k` | integer | ranking default | Max results. |
| `cross_repo_only` | boolean | `false` | Exclude same-repo matches. |

### `project_map`
Structural overview of one **registered** repo in the current generation: node
count, community breakdown, top hub nodes. Takes a `repo_id` that must resolve
against `registry.json` — never an arbitrary on-disk path.

| Arg | Type | Notes |
|-----|------|-------|
| `repo` | string (required) | A registered `repo_id`. |

### `context_pack`
Evidence cards (citations, snippets, confidence) for a goal, truncated to a
token budget without ever splitting a card mid-way.

| Arg | Type | Default | Notes |
|-----|------|---------|-------|
| `goal` | string | — (required) | What you are trying to do. |
| `scope` | string | — | Same scope grammar as `search`. |
| `token_budget` | integer | server default | Hard cap on returned card volume. |

## Registering with a client

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

You can also invoke the module directly (useful before an install, with the
package on `PYTHONPATH`):

```bash
PYTHONPATH=src python -m graphify_mesh.server.server
```
