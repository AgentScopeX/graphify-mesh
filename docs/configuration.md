# Configuration

Everything is configurable via CLI flags, `GRAPHIFY_MESH_*` environment
variables, or the `Settings` / `ServerConfig` dataclasses directly. No
machine-specific paths are baked into the package.

## Environment-variable prefix

This package uses the **`GRAPHIFY_MESH_`** prefix for all of its own
environment variables. (An earlier internal iteration used a different prefix;
it was renamed to `GRAPHIFY_MESH_` when the engine became a standalone generic
package so that nothing carries organization-specific naming.)

Two variables belong to the upstream `graphify` CLI, not to this package, and
keep their upstream names: `GRAPHIFY_BIN` and `GRAPHIFY_NO_BACKUP`.

## Environment variables

| Variable | Used by | Default | Meaning |
|----------|---------|---------|---------|
| `GRAPHIFY_MESH_ROOT` | sync + server | current working directory | Root that contains the `graphify/global/` tree this engine publishes into and serves from. |
| `GRAPHIFY_MESH_SCAN_ROOT` | sync | current working directory | Root scanned for per-repo `graphify-out` symlinks; also the "approved root" for the symlink-traversal guard. |
| `GRAPHIFY_MESH_REGISTRY` | sync + server | `<root>/bin/registry.json` | Path to `registry.json`. |
| `GRAPHIFY_MESH_OLLAMA_BASE_URL` | sync (naming) | `http://localhost:11434/v1` | OpenAI-compatible `/v1` endpoint for the community-labeling LLM. |
| `GRAPHIFY_MESH_OLLAMA_API_KEY` | sync (naming) | `dummy` | API key sent to the `/v1` endpoint (Ollama ignores it, but the client requires one). |
| `GRAPHIFY_MESH_OLLAMA_MODEL` | sync (naming) | `qwen2.5-coder:14b` | Model for community labeling. |
| `GRAPHIFY_MESH_OLLAMA_HEALTH_TIMEOUT` | sync (naming) | `3.0` | Seconds to wait on the naming-stage health check before degrading. |
| `GRAPHIFY_MESH_OLLAMA_EMBED_BASE_URL` | sync (embed) | `http://localhost:11434` | **Native** `/api/embed` endpoint (no `/v1` suffix). |
| `GRAPHIFY_MESH_OLLAMA_EMBED_MODEL` | sync (embed) | `qwen3-embedding:0.6b` | Embedding model. |
| `GRAPHIFY_MESH_OLLAMA_EMBED_HEALTH_TIMEOUT` | sync (embed) | `3.0` | Seconds to wait on the embed-stage health check before degrading. |
| `GRAPHIFY_BIN` | sync | `graphify` | Name/path of the upstream `graphify` binary. |
| `GRAPHIFY_NO_BACKUP` | sync | (set to `1` on child calls) | Suppresses `graphify`'s dated backup dirs; the sync engine always sets this on the graphify subprocesses it spawns. |

## CLI flags (`graphify-mesh-sync`)

| Flag | Meaning |
|------|---------|
| `--once` | Single run (the only supported mode; there is no daemon loop). |
| `--dry-run` | Print every action; write nothing outside a private staging dir. |
| `--mesh-root PATH` | Override `GRAPHIFY_MESH_ROOT`. |
| `--scan-root PATH` | Override `GRAPHIFY_MESH_SCAN_ROOT`. |
| `--registry PATH` | Override `GRAPHIFY_MESH_REGISTRY`. |
| `--skip-labeling` / `--no-skip-labeling` | Skip / enforce the non-placeholder community-name check. |
| `--skip-embedding` | Log-skip the embedding stage. |
| `--allow-shrink` | Authorize publishing a smaller graph than the previous generation. |
| `-v`, `--verbose` | Debug logging. |

## `Settings` fields (sync)

`graphify_mesh.sync.config.Settings` — resolved runtime configuration for one
pipeline run. Notable fields and derived paths:

- `mesh_root`, `scan_root`, `approved_root`, `registry_path` — base locations.
- `graphify_bin`, `stale_threshold`, `dry_run`, `skip_labeling`,
  `skip_embedding`, `allow_shrink` — run behavior.
- `ollama_*` / `ollama_embed_*` — naming and embedding endpoints, models, and
  health timeouts (plus test-only injectable health checks).
- `keep_embedding_generations` — how many published generations' embedding
  shards to keep on disk (older ones are GC'd at publish time).

Derived path properties (all under `mesh_root`):

| Property | Location |
|----------|----------|
| `global_dir` | `<mesh_root>/graphify/global` |
| `generations_dir` | `<global_dir>/generations` |
| `current_symlink` | `<global_dir>/current` |
| `status_path` | `<global_dir>/status.json` |
| `state_path` | `<global_dir>/state/source-manifests.json` |
| `lock_path` | `<global_dir>/.graphify-mesh-sync.lock` |
| `naming_dir` | `<global_dir>/naming` |
| `embeddings_dir` | `<global_dir>/embeddings` |
| `manual_relations_path` | `<mesh_root>/bin/manual-relations.json` |
| `manual_relations_schema_path` | `<mesh_root>/bin/manual-relations.schema.json` |

## `ServerConfig` fields (server)

`graphify_mesh.server.config.ServerConfig` — `mesh_root` and `registry_path`,
plus derived `global_dir`, `current_symlink`, and `embeddings_current_symlink`.
Resolved from `GRAPHIFY_MESH_ROOT` / `GRAPHIFY_MESH_REGISTRY`.

## `registry.json`

Source of truth for which repos are in the mesh. See
`examples/registry.example.json`.

```json
{
  "repos": [
    {
      "repo_id": "example-org.backend-a",
      "root": "/path/to/your/checkouts/backend-a",
      "collection_path": "/path/to/your/graph-mesh/graphify/example-org/backend-a",
      "enabled": true
    }
  ],
  "disabled": [],
  "external_roots": []
}
```

| Field | Meaning |
|-------|---------|
| `repos[].repo_id` | Stable logical id; becomes the node-id prefix / `repo` attribute in the merged graph. Must match `^[A-Za-z0-9][A-Za-z0-9._-]*$` (it is used as a filename, e.g. embedding shards) and be unique — duplicates are a load-time error. |
| `repos[].root` | The repo's checkout directory. |
| `repos[].collection_path` | Directory holding that repo's `graph.json`. |
| `repos[].enabled` | If `false`, the repo is skipped. |
| `disabled` | List of `repo_id`s to force-disable. |
| `external_roots` | Additional approved roots for symlink resolution. |

## `manual-relations.json`

Human-declared cross-project overlay edges the sync engine cannot infer,
validated against `examples/manual-relations.schema.json`. See
`examples/manual-relations.example.json`.

Top level is `{ "relations": [ ... ] }`. Each relation has:

| Field | Meaning |
|-------|---------|
| `type` | One of `depends_on`, `similar_approach`, `provides_api`, `consumes_api`. |
| `source`, `target` | A logical ref: `{ repo, source_file, qualified_label, signature? }`. Both must resolve against the current generation's per-repo graphs at load time — a dangling reference is a hard error. |
| `confidence` | Optional number in `[0, 1]`. |
| `evidence` | Optional human-readable justification string. |
