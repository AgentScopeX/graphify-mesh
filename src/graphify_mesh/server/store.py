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

import numpy as np

from graphify_mesh.server.config import ServerConfig
from graphify_mesh.sync.embedding import (
    SHARD_MATRIX_SUFFIX,
    SHARD_META_SUFFIX,
    _validate_v2_shard,
    node_key,
)
from graphify_mesh.sync.lexical_index import SUPPORTED_LEXICAL_SCHEMA_VERSIONS
from graphify_mesh.sync.lexical_index import TOKENIZER_VERSION as EXPECTED_TOKENIZER_VERSION
from graphify_mesh.sync.publish import output_hash
from graphify_mesh.sync.validate import validate_generation_manifest
from graphify_mesh.sync.vectors import RepoVectors

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
    embeddings: dict[str, RepoVectors]  # repo_id -> RepoVectors (sorted-key float32 matrix)
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
    if lexical_schema is not None and lexical_schema not in SUPPORTED_LEXICAL_SCHEMA_VERSIONS:
        errors.append(
            f"lexical-index: schema_version={lexical_schema!r} is not a version this server "
            f"understands (supported: {sorted(SUPPORTED_LEXICAL_SCHEMA_VERSIONS)}) — "
            "refusing to serve stale-shaped index"
        )
    return errors


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


# Per-shard byte ceiling for the read side. Shards are produced by our own
# sync pipeline, but the server must not OOM on a corrupt or hand-edited
# shard file — skipping one repo's vectors is the documented degraded mode
# (vector channel drops out, lexical/structural still serve). This gate
# applies to the `.json`/`.meta.json` reads below (both are read whole into
# memory via `_read_json`). It is deliberately NOT applied to the v2
# `.npy` matrix file itself: that is opened with `np.load(..., mmap_mode="r")`,
# which avoids eager load-time allocation, so merely opening an oversized
# `.npy` cannot OOM this process the way an oversized JSON blob can. That
# said, this is not a blanket safety guarantee for the whole v2 read path —
# `RepoVectors.normalized()` (see server/retrieval.py) still materializes
# one full float32 copy of the matrix at query time, so an absurdly
# oversized `.npy` could still cost real RAM there. No size ceiling is
# enforced on that path deliberately: unlike a hand-edited/corrupt JSON
# shard, the `.npy` matrix is produced only by our own sync pipeline, not
# arbitrary input.
MAX_SHARD_BYTES = 256 * 1024 * 1024


def _load_v1_shard_vectors(shard_path: Path) -> tuple[str, RepoVectors] | None:
    """v1 shard (`<repo>.json`, entries carry inline `embedding` lists).
    Returns `None` (repo skipped entirely) only when the shard's own
    top-level shape is untrustworthy (oversized, unparseable, not an
    object, or `entries` not a dict) — an individual malformed entry is
    dropped on its own without invalidating the rest of the shard."""
    try:
        if shard_path.stat().st_size > MAX_SHARD_BYTES:
            log.warning(
                "%s: shard exceeds MAX_SHARD_BYTES — skipping this repo's vectors", shard_path
            )
            return None
    except OSError:
        return None

    data = _read_json(shard_path)
    if not isinstance(data, dict):
        return None
    repo_id = data.get("repo_id", shard_path.stem)
    entries = data.get("entries", {})
    if not isinstance(entries, dict):
        return None

    vector_sources = {
        key: entry["embedding"]
        for key, entry in entries.items()
        if isinstance(entry, dict) and entry.get("embedding")
    }
    return repo_id, RepoVectors.from_mapping(vector_sources)


def _load_v2_shard_vectors(
    embeddings_current: Path, repo_id: str, meta_path: Path
) -> RepoVectors | None:
    """v2 shard (`<repo>.meta.json` + `<repo>.npy`, mmap'd). Returns `None`
    (repo skipped entirely, same documented per-shard degraded mode as the
    v1 oversized-shard skip) whenever the meta/matrix pair fails validation
    — reuses `graphify_mesh.sync.embedding._validate_v2_shard` so the read
    side trusts the exact same shape checks (shard_format allowlist, dim
    match, row range, duplicate rows) the sync side already enforces on
    write."""
    try:
        if meta_path.stat().st_size > MAX_SHARD_BYTES:
            log.warning(
                "%s: shard meta %s exceeds MAX_SHARD_BYTES — skipping this repo's vectors",
                repo_id,
                meta_path,
            )
            return None
    except OSError:
        return None

    meta = _read_json(meta_path)
    if not isinstance(meta, dict):
        log.warning(
            "%s: shard meta at %s is not a JSON object — skipping this repo's vectors",
            repo_id,
            meta_path,
        )
        return None

    matrix_path = embeddings_current / f"{repo_id}{SHARD_MATRIX_SUFFIX}"
    if not matrix_path.is_file():
        log.warning(
            "%s: v2 shard meta present but matrix missing at %s — skipping this repo's vectors",
            repo_id,
            matrix_path,
        )
        return None
    try:
        matrix = np.load(matrix_path, mmap_mode="r")
    except (OSError, ValueError) as exc:
        log.warning(
            "%s: failed to load shard matrix %s (%s) — skipping this repo's vectors",
            repo_id,
            matrix_path,
            exc,
        )
        return None

    invalid_reason = _validate_v2_shard(meta, matrix)
    if invalid_reason is not None:
        log.warning(
            "%s: invalid v2 shard in %s (%s) — skipping this repo's vectors",
            repo_id,
            meta_path,
            invalid_reason,
        )
        return None

    entries: dict[str, dict] = meta["entries"]
    n_rows = matrix.shape[0]
    keys_by_row: list[str | None] = [None] * n_rows
    for key, entry in entries.items():
        row = entry.get("row")
        if row is None:
            continue
        keys_by_row[row] = key

    if any(key is None for key in keys_by_row):
        log.warning(
            "%s: shard matrix has rows with no owning entry in %s — skipping this repo's vectors",
            repo_id,
            meta_path,
        )
        return None

    # Every row was assigned a key above (the `any(... is None)` guard
    # returned early otherwise) — narrow `list[str | None]` to `list[str]`
    # explicitly so mypy sees what we already know at runtime.
    narrowed_keys: list[str] = [key for key in keys_by_row if key is not None]
    if len(narrowed_keys) != len(keys_by_row):
        log.warning("embeddings: internal row/key narrowing mismatch — skipping shard")
        return None

    # `RepoVectors.from_rows` canonicalizes to the sorted-key invariant:
    # already-sorted `keys_by_row` (the common case — our own writer,
    # `stage_embeddings`, always serializes rows in sorted-key order) keeps
    # the mmap array as-is (zero-copy); out-of-order rows (a hand-edited or
    # otherwise out-of-band-written shard) get argsort-permuted, at the
    # cost of materializing a full copy for this one repo. Shared with the
    # sync-side reader (`graphify_mesh.sync.embedding._read_v2_shard`) so
    # this canonicalization logic exists exactly once.
    return RepoVectors.from_rows(narrowed_keys, matrix)


def _load_embeddings(embeddings_current: Path) -> dict[str, RepoVectors]:
    """Loads this generation's per-repo embedding shards. Two on-disk shard
    formats are supported side by side: the sync pipeline now writes v2
    shards (`<repo>.meta.json` + mmap'd `<repo>.npy`); older publishes (or
    a generation produced before this server understood v2) may still have
    v1 plain-JSON shards (`<repo>.json`) on disk. Per repo: v2 wins if its
    meta file is present, else fall back to the v1 file. A shard that fails
    validation is skipped for its repo only (documented per-shard degraded
    mode) — the vector channel drops that one repo, everything else
    (lexical, structural, and every other repo's vectors) still serves.
    """
    if not embeddings_current.is_dir():
        return {}

    out: dict[str, RepoVectors] = {}
    v2_repo_ids: set[str] = set()

    for meta_path in sorted(embeddings_current.glob(f"*{SHARD_META_SUFFIX}")):
        repo_id = meta_path.name[: -len(SHARD_META_SUFFIX)]
        v2_repo_ids.add(repo_id)
        vectors = _load_v2_shard_vectors(embeddings_current, repo_id, meta_path)
        if vectors is not None:
            out[repo_id] = vectors

    for shard_path in sorted(embeddings_current.glob("*.json")):
        if shard_path.name == "id-map.json":
            continue
        if shard_path.name.endswith(SHARD_META_SUFFIX):
            continue
        if shard_path.stem in v2_repo_ids:
            continue  # v2 shard already loaded for this repo — v2 wins
        loaded = _load_v1_shard_vectors(shard_path)
        if loaded is None:
            continue
        repo_id, vectors = loaded
        out[repo_id] = vectors

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

        # `_read_json` conflates "file missing" and "file present but
        # unparseable" into the same `None` — both are NOT the same state
        # here. Missing lexical-index.json is the documented degraded mode
        # (vector-style absence: lexical drops out, structural still
        # serves) and must not produce a validation error. Present-but-
        # corrupt (unparseable JSON, or parseable but not an object) is a
        # real artifact problem and must be surfaced as a validation error
        # alongside `validate_manifest_consistency`'s own errors.
        lexical_path = current / "lexical-index.json"
        lexical_raw = _read_json(lexical_path)
        lexical_error: str | None = None
        if lexical_raw is None and lexical_path.is_file():
            lexical_error = (
                "lexical-index: artifact present but unreadable/not a JSON "
                "object — refusing generation"
            )
        if lexical_raw is not None and not isinstance(lexical_raw, dict):
            lexical_error = (
                "lexical-index: artifact present but unreadable/not a JSON "
                "object — refusing generation"
            )
        lexical = lexical_raw if isinstance(lexical_raw, dict) else {}

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
        if lexical_error is not None:
            errors.append(lexical_error)
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
