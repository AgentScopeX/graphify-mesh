"""graphify-mesh: companion stdio MCP server for the merged global graph.

Two constraints are the most load-bearing for this package: the `project_map`
input must resolve a registered `repo_id` (never an arbitrary on-disk path),
and a generation is only ever served whole once its manifest passes a
consistency check.

## Dependency decision

This server is implemented with the Python STANDARD LIBRARY ONLY — no `mcp`
SDK package, no third-party JSON-RPC/transport library, and therefore no
dedicated virtualenv is required. It runs under the same interpreter as
`graphify_mesh.sync`.

Why no `mcp` SDK: the MCP stdio transport is exactly newline-delimited
JSON-RPC 2.0 request/response objects on stdin/stdout (see `protocol.py`),
which stdlib `json` + `sys` implements completely. This keeps the server
dependency-free beyond `graphify` / `graphify_mesh`, and trivially portable
to any Python 3.11+ interpreter without an install step. If a richer MCP
feature (resources, prompts, sampling) is needed later, introduce the
official SDK at that time — nothing here needs to change to add one alongside
this package.

## Two MCP servers: this one vs graphify's own

graphify's own MCP server (`graphify.serve`) can be repointed to read the
PUBLISHED `<mesh_root>/graphify/global/current/global-graph.json` for
STRUCTURAL/PR tools only: `god_nodes`, `shortest_path`, `get_pr_impact`,
`get_community`, `get_neighbors`, `get_node`, `graph_stats`, `list_prs`,
`triage_prs`. Those tools operate purely on the structural graph shape and
gain nothing from this package's scope/fusion/overlay logic — repointing them
is a config change (point at the new published path instead of
`~/.graphify/global-graph.json`), not a rewrite.

graphify-mesh (this package) owns everything hybrid/cross-project/evidence-
oriented: `search`, `cross_project`, `find_similar`, `project_map`,
`context_pack`. These need scope resolution, the lexical index, the embedding
index, and the cross-project overlay — none of which graphify's own server
has access to.

**By design, the two servers can show different community names** for the
same node: graphify's server reads the published graph's `community_name`
attribute directly, but graphify-mesh's naming stage strips and relabels
community names during the sync pipeline (see
`graphify_mesh.sync.naming.strip_project_community_attrs`). Historical caches
or a not-yet-repointed upstream instance may still show an old per-project
name for a transitional period. This is not a bug to reconcile; it is a known
consequence of the relabeling running only in this pipeline. Route
accordingly: local per-project structural questions and PR tools -> graphify's
server; concept/cross-project/evidence-pack questions -> graphify-mesh.

## No arbitrary-path caching

This server never exposes a "give me any path on disk and I'll load/cache it"
parameter. `project_map(repo)` takes a `repo_id` that MUST resolve against the
registry (see `scope.py`) — there is no arbitrary-path acceptance and
therefore no unbounded per-path cache: the only thing ever cached is the
single current *generation* (`store.py`'s `GenerationStore`), reloaded
wholesale (never partially) on hot-reload and not keyed by caller-supplied
paths at all.
"""
