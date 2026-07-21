"""Generation-aware, all-or-nothing hot-reload loader (C28, C11).

`GenerationStore` holds exactly ONE loaded generation's worth of artifacts
in memory at a time: the published `global-graph.json`, its
`generation-manifest.json`, `cross-project-overlay.json`, and
`lexical-index.json`, plus the WS3 embedding shards + id-map (a sibling
publish under `embeddings/current/`, flipped atomically alongside the main
generation — see `graphify_mesh.sync.embedding.persist_generation`).

Hot reload is all-or-nothing (C28): before ANY tool call, `ensure_fresh()`
cheaply stats the `current` symlink's target and the manifest's mtime; if
either changed since the last successful load, a reload is attempted. If
the new generation fails `validate_manifest_consistency` (schema, hash,
count, or tokenizer-version mismatch), the reload is REJECTED and the
previously-loaded generation keeps serving, with `degraded` populated with
the reason — this server never serves a half-loaded or inconsistent
generation, and never crashes a running session because a publish landed
mid-reload.

This is the only cache this server keeps, and it is not path-keyed (C26):
callers never supply an arbitrary filesystem path that gets cached per-path
here — `project_map(repo)` etc. all resolve against the single in-memory
generation state below.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from graphify_mesh.server.config import ServerConfig
from graphify_mesh.sync.embedding import node_key
from graphify_mesh.sync.lexical_index import (
    LEXICAL_SCHEMA_VERSION as EXPECTED_LEXICAL_SCHEMA_VERSION,
)
from graphify_mesh.sync.lexical_index import TOKENIZER_VERSION as EXPECTED_TOKENIZER_VERSION
from graphify_mesh.sync.publish import output_hash
from graphify_mesh.sync.validate import validate_generation_manifest

log = logging.getLogger("graphify_mesh.server.store")


class GenerationUnavailableError(RuntimeError):
    """No generation has ever loaded successfully (fresh install with no publish
    yet, or every publish so far failed manifest consistency)."""


@dataclass
class Generation:
    generation_id: str
    manifest: dict
    graph: dict
    overlay: dict
    lexical: dict
    embeddings: dict[str, dict[str, list[float]]]  # repo_id -> {node_key: vector}
    node_by_id: dict = field(default_factory=dict)
    nodes_by_repo: dict = field(default_factory=dict)
    adjacency: dict = field(default_factory=dict)  # node_id -> list[(neighbor_id, edge_data)]
    key_by_node_id: dict = field(
        default_factory=dict
    )  # graph node id -> durable logical-ref key (C27)
    node_id_by_key: dict = field(default_factory=dict)  # inverse of key_by_node_id

    def build_indexes(self) -> None:
        for node in self.graph.get("nodes", []):
            if not isinstance(node, dict) or "id" not in node:
                continue
            self.node_by_id[node["id"]] = node
            repo_id = node.get("repo")
            self.nodes_by_repo.setdefault(repo_id, []).append(node["id"])
            if repo_id:
                key = node_key(repo_id, node)
                if key is not None:
                    self.key_by_node_id[node["id"]] = key
                    self.node_id_by_key[key] = node["id"]
        for link in self.graph.get("links", self.graph.get("edges", [])):
            if not isinstance(link, dict):
                continue
            src, dst = link.get("source"), link.get("target")
            if src is None or dst is None:
                continue
            self.adjacency.setdefault(src, []).append((dst, link))
            self.adjacency.setdefault(dst, []).append((src, link))

    def degree(self, node_id: str) -> int:
        return len(self.adjacency.get(node_id, []))


def validate_manifest_consistency(manifest: dict, graph: dict, lexical: dict) -> list[str]:
    """C28: hard consistency gate. Returns a list of errors; empty = ok.
    Reuses `graphify_mesh.sync.validate.validate_generation_manifest` for the
    required-keys check rather than re-deriving that list here."""
    errors: list[str] = list(validate_generation_manifest(manifest).errors)

    expected_nodes = manifest.get("output_node_count")
    expected_edges = manifest.get("output_edge_count")
    actual_nodes = len(graph.get("nodes", []))
    actual_edges = len(graph.get("links", graph.get("edges", [])))
    if expected_nodes is not None and expected_nodes != actual_nodes:
        errors.append(
            f"generation-manifest: output_node_count={expected_nodes} "
            f"but graph has {actual_nodes} nodes"
        )
    if expected_edges is not None and expected_edges != actual_edges:
        errors.append(
            f"generation-manifest: output_edge_count={expected_edges} "
            f"but graph has {actual_edges} edges"
        )

    expected_hash = manifest.get("output_hash")
    if expected_hash is not None and expected_hash != output_hash(graph):
        errors.append(
            "generation-manifest: output_hash does not match recomputed hash of global-graph.json"
        )

    manifest_tok = manifest.get("lexical_index_tokenizer_version")
    lexical_tok = lexical.get("tokenizer_version") if isinstance(lexical, dict) else None
    if manifest_tok is not None and lexical_tok is not None and manifest_tok != lexical_tok:
        errors.append(
            f"lexical-index: manifest tokenizer_version={manifest_tok!r} != "
            f"lexical-index.json tokenizer_version={lexical_tok!r}"
        )
    if lexical_tok is not None and lexical_tok != EXPECTED_TOKENIZER_VERSION:
        errors.append(
            f"lexical-index: tokenizer_version={lexical_tok!r} is not a version this server "
            f"understands (expected {EXPECTED_TOKENIZER_VERSION!r}) — "
            "refusing to serve stale-shaped index"
        )

    # C28: schema_version gate — independent of tokenizer_version above.
    # This covers the on-disk CONTAINER shape (postings/alias_exact entry
    # representation), not term-splitting rules; a schema_version mismatch
    # means this server would misindex into entries assuming the wrong
    # shape (e.g. dict-style `["weight"]` access against a v2 compact
    # array), so it is rejected exactly like a tokenizer mismatch.
    manifest_schema = manifest.get("lexical_index_schema_version")
    lexical_schema = lexical.get("schema_version") if isinstance(lexical, dict) else None
    schema_mismatch = manifest_schema is not None and lexical_schema is not None
    if schema_mismatch and manifest_schema != lexical_schema:
        errors.append(
            f"lexical-index: manifest schema_version={manifest_schema!r} != "
            f"lexical-index.json schema_version={lexical_schema!r}"
        )
    if lexical_schema is not None and lexical_schema != EXPECTED_LEXICAL_SCHEMA_VERSION:
        errors.append(
            f"lexical-index: schema_version={lexical_schema!r} is not a version this server "
            f"understands (expected {EXPECTED_LEXICAL_SCHEMA_VERSION!r}) — "
            "refusing to serve stale-shaped index"
        )
    return errors


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# Per-shard byte ceiling for the read side. Shards are produced by our own
# sync pipeline, but the server must not OOM on a corrupt or hand-edited
# shard file — skipping one repo's vectors is the documented degraded mode
# (vector channel drops out, lexical/structural still serve).
MAX_SHARD_BYTES = 256 * 1024 * 1024


def _load_embeddings(embeddings_current: Path) -> dict[str, dict[str, list[float]]]:
    if not embeddings_current.is_dir():
        return {}
    out: dict[str, dict[str, list[float]]] = {}
    for shard_path in sorted(embeddings_current.glob("*.json")):
        if shard_path.name == "id-map.json":
            continue
        try:
            if shard_path.stat().st_size > MAX_SHARD_BYTES:
                continue
        except OSError:
            continue
        data = _read_json(shard_path)
        if not isinstance(data, dict):
            continue
        repo_id = data.get("repo_id", shard_path.stem)
        entries = data.get("entries", {})
        if not isinstance(entries, dict):
            continue
        out[repo_id] = {
            k: v["embedding"]
            for k, v in entries.items()
            if isinstance(v, dict) and v.get("embedding")
        }
    return out


class GenerationStore:
    def __init__(self, config: ServerConfig):
        self.config = config
        self._generation: Generation | None = None
        self._manifest_mtime: float | None = None
        self._current_target: str | None = None
        self.degraded: list[str] = []

    def _stat_signature(self) -> tuple[str | None, float | None]:
        current = self.config.current_symlink
        if not current.exists():
            return None, None
        try:
            target = os.path.realpath(current)
            manifest_path = current / "generation-manifest.json"
            mtime = manifest_path.stat().st_mtime if manifest_path.is_file() else None
        except OSError:
            return None, None
        return target, mtime

    def ensure_fresh(self) -> None:
        target, mtime = self._stat_signature()
        if target is None:
            if self._generation is None:
                self.degraded = ["no_generation_published"]
            return
        if target == self._current_target and mtime == self._manifest_mtime:
            return  # unchanged, nothing to do
        self._try_reload(target, mtime)

    def _try_reload(self, target: str, mtime: float | None) -> None:
        current = self.config.current_symlink
        manifest = _read_json(current / "generation-manifest.json")
        graph = _read_json(current / "global-graph.json")
        overlay = _read_json(current / "cross-project-overlay.json") or {"edges": []}
        lexical = _read_json(current / "lexical-index.json") or {}

        if manifest is None or graph is None:
            log.warning(
                "graphify-mesh: reload skipped — manifest or graph unreadable at %s", current
            )
            # Always surface the rejection in `degraded`, even when a
            # previously-loaded generation keeps serving (see module
            # docstring: "the reload is REJECTED ... with `degraded`
            # populated with the reason" — not conditioned on whether this
            # is the very first load).
            self.degraded = ["reload_failed_unreadable_artifacts"]
            return

        errors = validate_manifest_consistency(manifest, graph, lexical)
        if errors:
            log.warning(
                "graphify-mesh: rejecting inconsistent generation %s (%d errors): %s",
                manifest.get("generation_id", "?"),
                len(errors),
                "; ".join(errors[:3]),
            )
            reason = (
                "no_consistent_generation_available"
                if self._generation is None
                else "reload_rejected_previous_generation_still_serving"
            )
            self.degraded = [reason] + errors[:3]
            # All-or-nothing: keep serving whatever was already loaded (if
            # anything), never swap in the inconsistent one.
            return

        embeddings = _load_embeddings(self.config.embeddings_current_symlink)
        generation = Generation(
            generation_id=manifest["generation_id"],
            manifest=manifest,
            graph=graph,
            overlay=overlay,
            lexical=lexical,
            embeddings=embeddings,
        )
        generation.build_indexes()
        self._generation = generation
        self._current_target = target
        self._manifest_mtime = mtime
        self.degraded = [] if embeddings else ["embeddings_unavailable"]

    @property
    def generation(self) -> Generation:
        self.ensure_fresh()
        if self._generation is None:
            raise GenerationUnavailableError(
                "no consistent published generation is available yet "
                "(fresh install, or every publish so far "
                "failed manifest consistency) — run the graphify-mesh-sync pipeline at least once"
            )
        return self._generation
