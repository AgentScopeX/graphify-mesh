"""WS3 embedding index: snippet builder, batched native `/api/embed` calls,
id-map + tombstones (C27), sharded/resumable storage, and GC.

Runs AFTER the WS2 naming/label stage and BEFORE WS4 overlay-resolve (plan
WS1 item 6 order: `... -> label -> embed changed -> overlay resolve -> ...`).

Design notes (read before changing anything here):

  * C6 (embedding input): every embedded node's input string is
    `label + bounded source snippet + path + community name` — see
    `build_embedding_input`. Nodes carry no context text by default, so the
    snippet is read directly from `source_file` on disk, bounded to a small
    line/char window (`SNIPPET_WINDOW_LINES`/`SNIPPET_MAX_CHARS`) so this
    never turns into an unbounded file read.

  * C9 (native contract): batched calls hit `{base_url}/api/embed` — the
    NATIVE Ollama endpoint, NOT the `/v1` OpenAI-compat surface the WS2
    naming stage's LLM calls use. Verified by a real curl during WS3 build
    (see config.py's EMBED_DEFAULT_* docstring for the exact request/response
    shape). Never reuse naming.py's `default_ollama_health_check` here — it
    targets `/v1/models`, a different API surface entirely; this module has
    its own `default_embed_health_check` against the native `/api/tags`.

  * C27 (id-map + tombstones): `build_id_map` rebuilds the id-map to the
    EXACT current node-key set every generation. A key that existed in the
    previous id-map as "active" but is absent from the current run's key set
    is marked "tombstoned" (with the generation_id it disappeared in) rather
    than silently dropped. Keys are `LogicalRef`-shaped
    (repo, source_file, qualified_label) — the same durable-reference shape
    WS4's overlay module already uses — never a raw graphify node id, which
    is not stable across generations.

  * Resumability / changed-only: the pipeline already knows, per repo, which
    repos were untouched this run (`decide_action` -> ACTION_SKIP, see
    sync_project.py). `run_embedding_stage` is handed that same set
    (`unchanged_repo_ids`) and reuses the ENTIRE previous shard for those
    repos without recomputing anything. For repos that did change, each
    node's embedding input is content-hashed; a node whose hash matches the
    previous published shard's entry is reused, otherwise it is
    (re)embedded. This deliberately reuses WS1's existing changed-detection
    signal instead of re-deriving one.

  * Skip heuristic: trivial one-line getters/setters (see `is_trivial_node`)
    are never embedded — the embedding call budget is spent on nodes likely
    to carry distinguishing semantic content. Skipped nodes carry no vector
    at all in the shard; `overlay_similar.py`'s fallback (exact label +
    same-community match) is the documented behavior for them (deliverable
    7) rather than leaving them silently unhandled.

  * Storage/GC: shards are staged into the per-run ephemeral tempdir during
    the pipeline run (`stage_embeddings`) and only copied into the permanent,
    untracked `settings.embeddings_dir/generations/<generation_id>/` tree
    (plus the `current` symlink flip and GC of old generations) once the
    surrounding pipeline run actually publishes (`persist_generation`) — the
    same "nothing durable happens unless publish happens" rule the rest of
    the pipeline already follows for naming/overlay artifacts.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from graphify_mesh.sync.config import Settings
from graphify_mesh.sync.overlay_refs import LogicalRef

log = logging.getLogger("graphify_mesh.sync.embedding")

EMBED_HEALTHY = "ok"
EMBED_DEGRADED = "degraded"

# C6 snippet window: bounded read around the node's line (if known), or the
# file's head if no line is available. Kept small and fixed rather than
# reading whole files — nodes have no context text by default per C6, and a
# bounded window is the whole point of the constraint.
SNIPPET_WINDOW_LINES = 20
SNIPPET_MAX_CHARS = 800
# Hard cap on how many lines of a source file are ever scanned to locate the
# window, so a single pathologically large file can't blow up one embed run.
SNIPPET_READ_LINE_CAP = 4000

# C6 final embedding input: label + snippet + path + community name,
# truncated to this many chars as a last-resort guard regardless of how the
# pieces above were individually bounded.
EMBED_INPUT_MAX_CHARS = 3000

# Skip heuristic: trivial one-line accessors are not worth an embedding call.
# A node is "trivial" iff its label matches this accessor-naming pattern AND
# its snippet (once built) has at most TRIVIAL_MAX_SNIPPET_LINES non-blank
# lines — i.e. it really is a bare one-liner, not just an accessor-named
# method with a meaningful body.
TRIVIAL_LABEL_PATTERN = re.compile(r"^(get|set|is|has)[A-Z_]\w*$")
TRIVIAL_MAX_SNIPPET_LINES = 2

EMBED_BATCH_SIZE = 16
EMBED_REQUEST_TIMEOUT = 30.0

HealthCheckFn = Callable[[str, float], bool]


@dataclass
class EmbeddingRecipe:
    """C28: the exact recipe recorded in the generation manifest so a future
    generation (or a human debugging drift) can tell what produced a vector
    without re-deriving it from source."""

    model: str
    dim: int
    snippet_window_lines: int = SNIPPET_WINDOW_LINES
    snippet_max_chars: int = SNIPPET_MAX_CHARS
    input_max_chars: int = EMBED_INPUT_MAX_CHARS
    skip_heuristic: str = (
        f"trivial accessor skip: label matches {TRIVIAL_LABEL_PATTERN.pattern!r} "
        f"AND snippet has <= {TRIVIAL_MAX_SNIPPET_LINES} non-blank lines"
    )
    tokenizer_notes: str = (
        "no client-side tokenization/truncation beyond EMBED_INPUT_MAX_CHARS char cap; "
        "the /api/embed backend applies its own model-native tokenizer server-side"
    )

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "dim": self.dim,
            "snippet_window_lines": self.snippet_window_lines,
            "snippet_max_chars": self.snippet_max_chars,
            "input_max_chars": self.input_max_chars,
            "skip_heuristic": self.skip_heuristic,
            "tokenizer_notes": self.tokenizer_notes,
        }


@dataclass
class EmbeddingStats:
    embedded: int = 0
    skipped_trivial: int = 0
    reused: int = 0
    reused_repos_unchanged: int = 0
    tombstoned: int = 0

    def to_dict(self) -> dict:
        return {
            "embedded": self.embedded,
            "skipped_trivial": self.skipped_trivial,
            "reused": self.reused,
            "reused_repos_unchanged": self.reused_repos_unchanged,
            "tombstoned": self.tombstoned,
        }


@dataclass
class EmbeddingStageResult:
    status: str  # EMBED_HEALTHY | EMBED_DEGRADED
    vectors_by_repo: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    # Full shard entries (content_hash + embedding, including trivial-skip
    # `{"embedding": None}` markers) per repo — this is what actually gets
    # persisted to disk (`stage_embeddings`/`persist_generation`), so a
    # future run can resume from content_hash comparisons even for nodes
    # that carry no vector. `vectors_by_repo` above is the vector-only view
    # `overlay_similar.py` consumes.
    shards_by_repo: dict[str, dict[str, dict]] = field(default_factory=dict)
    stats: EmbeddingStats = field(default_factory=EmbeddingStats)
    recipe: EmbeddingRecipe | None = None
    id_map: dict[str, dict] = field(default_factory=dict)
    reason: str = ""


def key_to_ref(key: str) -> LogicalRef:
    """Inverse of `LogicalRef.to_key()` — splits an embedding shard/id-map
    key back into its logical-ref parts for building an OverlayEdge."""
    repo, source_file, label = key.split("\x1f", 2)
    return LogicalRef(repo=repo, source_file=source_file, qualified_label=label)


def node_key(repo_id: str, node: dict) -> str | None:
    """Durable logical key for a node, matching the `LogicalRef` shape WS4's
    overlay module already uses (C27) — never the raw per-repo graphify node
    id, which is not stable across generations."""
    source_file = node.get("source_file")
    label = node.get("label")
    if not source_file or not label:
        return None
    return LogicalRef(repo=repo_id, source_file=source_file, qualified_label=label).to_key()


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_snippet(source_root: Path | None, source_file: str | None, line: int | None) -> str:
    """C6 bounded source snippet. Returns "" (not an error) whenever a
    snippet cannot be produced — missing root, missing file, unreadable file
    — embedding still proceeds with label/path/community alone."""
    if source_root is None or not source_file:
        return ""
    path = source_root / source_file
    if not path.is_file():
        return ""

    half = SNIPPET_WINDOW_LINES // 2
    start = max(0, (line - 1 - half)) if line else 0
    end = start + SNIPPET_WINDOW_LINES

    lines: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for idx, raw_line in enumerate(fh):
                if idx >= SNIPPET_READ_LINE_CAP:
                    break
                if idx < start:
                    continue
                if idx >= end:
                    break
                lines.append(raw_line.rstrip("\n"))
    except OSError:
        return ""

    snippet = "\n".join(lines)
    return snippet[:SNIPPET_MAX_CHARS]


def build_embedding_input(label: str, snippet: str, source_file: str, community_name: str | None) -> str:
    """C6: label + bounded source snippet + path + community name,
    concatenated into a single embedding input string."""
    parts = [
        f"label: {label}",
        f"path: {source_file}",
        f"community: {community_name or 'unassigned'}",
    ]
    if snippet:
        parts.append(f"snippet:\n{snippet}")
    text = "\n".join(parts)
    return text[:EMBED_INPUT_MAX_CHARS]


def is_trivial_node(label: str, snippet: str) -> bool:
    """Skip heuristic: trivial one-line accessors are not embedded at all
    (deliverable 2). Documented fallback for these nodes when a caller wants
    "similar" results anyway lives in `overlay_similar.py` (exact label +
    same-community match, no embedding lookup — deliverable 7)."""
    if not TRIVIAL_LABEL_PATTERN.match(label or ""):
        return False
    non_blank = [ln for ln in snippet.splitlines() if ln.strip()]
    return len(non_blank) <= TRIVIAL_MAX_SNIPPET_LINES


def default_embed_health_check(base_url: str, timeout: float) -> bool:
    """GET `{base_url}/api/tags` — the cheapest native-endpoint call that
    confirms the Ollama host is reachable, without triggering an actual
    embedding computation. Any failure => unhealthy; never raises (C9's
    "never crash the pipeline on a down local service" pattern, mirroring
    naming.default_ollama_health_check, but against the native surface)."""
    url = base_url.rstrip("/") + "/api/tags"
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed internal endpoint
            status = getattr(resp, "status", resp.getcode())
            return 200 <= status < 300
    except Exception as exc:  # noqa: BLE001 - any failure => unhealthy, never crash the pipeline
        log.warning("embed health check failed for %s: %s", url, exc)
        return False


def embed_batch(base_url: str, model: str, inputs: list[str], timeout: float = EMBED_REQUEST_TIMEOUT) -> list[list[float]]:
    """POST `{base_url}/api/embed` with `{"model": model, "input": inputs}`
    (native contract, verified via a real curl during WS3 build — NOT the
    `/v1/embeddings` OpenAI-compat shape, per C9). Returns one vector per
    input, in input order. Raises RuntimeError on any transport/shape
    failure — callers decide whether that means "degraded mode" or a hard
    failure; this function never silently returns partial/wrong-length
    results."""
    if not inputs:
        return []
    url = base_url.rstrip("/") + "/api/embed"
    payload = json.dumps({"model": model, "input": inputs}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed internal endpoint
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"embed_batch request to {url} failed: {exc}") from exc

    vectors = body.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(inputs):
        got_description = len(vectors) if isinstance(vectors, list) else repr(vectors)
        raise RuntimeError(
            f"embed_batch response shape mismatch from {url}: expected {len(inputs)} embeddings, "
            f"got {got_description}"
        )
    return vectors


def _iter_embeddable_nodes(repo_id: str, graph_data: dict, source_root: Path | None):
    """Yields (key, label, source_file, community_name, snippet, is_trivial)
    for every node in this repo's raw graph that has enough fields to build
    a logical key at all. Nodes without a resolvable key (no source_file or
    label — e.g. bare external/library nodes) are not embeddable and are
    skipped entirely, same as they're unindexable for overlay logical refs."""
    for node in graph_data.get("nodes", []):
        if not isinstance(node, dict):
            continue
        key = node_key(repo_id, node)
        if key is None:
            continue
        label = node["label"]
        source_file = node["source_file"]
        community_name = node.get("community_name")
        snippet = build_snippet(source_root, source_file, node.get("line"))
        yield key, label, source_file, community_name, snippet, is_trivial_node(label, snippet)


def compute_repo_shard(
    repo_id: str,
    graph_data: dict,
    source_root: Path | None,
    previous_shard: dict[str, dict],
    base_url: str,
    model: str,
    stats: EmbeddingStats,
) -> dict[str, dict]:
    """Builds this repo's shard for a repo that changed this run. Each shard
    entry is `{"content_hash": ..., "embedding": [...] | None}` — `None`
    means the node was skipped by the trivial heuristic, never that
    embedding failed silently (a real failure raises, per `embed_batch`)."""
    shard: dict[str, dict] = {}
    to_embed_keys: list[str] = []
    to_embed_inputs: list[str] = []

    for key, label, source_file, community_name, snippet, trivial in _iter_embeddable_nodes(
        repo_id, graph_data, source_root
    ):
        if trivial:
            shard[key] = {"content_hash": None, "embedding": None}
            stats.skipped_trivial += 1
            continue

        input_text = build_embedding_input(label, snippet, source_file, community_name)
        digest = _content_hash(input_text)
        prev_entry = previous_shard.get(key)
        if prev_entry is not None and prev_entry.get("content_hash") == digest and prev_entry.get("embedding"):
            shard[key] = {"content_hash": digest, "embedding": prev_entry["embedding"]}
            stats.reused += 1
            continue

        to_embed_keys.append(key)
        to_embed_inputs.append(input_text)
        shard[key] = {"content_hash": digest, "embedding": None}  # placeholder, filled below

    total_batches = -(-len(to_embed_keys) // EMBED_BATCH_SIZE) if to_embed_keys else 0
    if total_batches:
        log.info("%s: embedding %d nodes in %d batches ...", repo_id, len(to_embed_keys), total_batches)
    for batch_start in range(0, len(to_embed_keys), EMBED_BATCH_SIZE):
        batch_num = batch_start // EMBED_BATCH_SIZE + 1
        batch_keys = to_embed_keys[batch_start : batch_start + EMBED_BATCH_SIZE]
        batch_inputs = to_embed_inputs[batch_start : batch_start + EMBED_BATCH_SIZE]
        vectors = embed_batch(base_url, model, batch_inputs)
        for key, vector in zip(batch_keys, vectors):
            shard[key]["embedding"] = vector
            stats.embedded += 1
        log.info("%s: batch %d/%d done (%d nodes embedded so far)", repo_id, batch_num, total_batches, stats.embedded)

    return shard


def build_id_map(previous_id_map: dict[str, dict], current_keys: set[str], generation_id: str) -> dict[str, dict]:
    """C27: rebuild the id-map to the EXACT current node-key set every
    generation. Keys present now are (re)marked "active"; keys that were
    "active" in the previous id-map but are absent now are marked
    "tombstoned" (never silently dropped) with the generation_id they
    disappeared in. A key already "tombstoned" stays tombstoned (its
    original tombstoned_at generation_id is preserved) unless it reappears,
    in which case it goes back to "active"."""
    new_map: dict[str, dict] = {}
    for key in current_keys:
        # Present now, regardless of prior status (including a previously
        # tombstoned key reappearing) -> active again, tombstone cleared.
        new_map[key] = {"status": "active", "generation_id": generation_id, "tombstoned_at": None}

    for key, prior in previous_id_map.items():
        if key in current_keys:
            continue
        if prior.get("status") == "tombstoned":
            new_map[key] = prior
            continue
        new_map[key] = {
            "status": "tombstoned",
            "generation_id": prior.get("generation_id"),
            "tombstoned_at": generation_id,
        }
    return new_map


def read_previous_shard(embeddings_current_dir: Path | None, repo_id: str) -> dict[str, dict]:
    if embeddings_current_dir is None or not embeddings_current_dir.is_dir():
        return {}
    shard_path = embeddings_current_dir / f"{repo_id}.json"
    if not shard_path.is_file():
        return {}
    try:
        return json.loads(shard_path.read_text(encoding="utf-8")).get("entries", {})
    except (OSError, json.JSONDecodeError):
        return {}


def read_previous_id_map(embeddings_current_dir: Path | None) -> dict[str, dict]:
    if embeddings_current_dir is None or not embeddings_current_dir.is_dir():
        return {}
    id_map_path = embeddings_current_dir / "id-map.json"
    if not id_map_path.is_file():
        return {}
    try:
        return json.loads(id_map_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def run_embedding_stage(
    graph_paths_by_repo,
    repo_roots_by_id,
    graphs_by_repo: dict[str, dict],
    unchanged_repo_ids: set[str],
    settings: Settings,
    provisional_generation_id: str,
    health_check: HealthCheckFn | None = None,
) -> EmbeddingStageResult:
    """Top-level WS3 embed-changed stage. `graphs_by_repo` is the same
    per-repo raw graph data overlay.py already loads (post per-project
    labeling, i.e. carrying per-repo `community_name`) — passed in rather
    than reloaded, so pipeline.py only reads each per-repo graph.json once.

    `provisional_generation_id` is a run-scoped identifier used only for the
    id-map's "tombstoned_at"/"generation_id" bookkeeping while staging;
    nothing durable is written until `persist_generation` is called after a
    successful publish (see module docstring)."""
    check = health_check if health_check is not None else default_embed_health_check
    healthy = check(settings.ollama_embed_base_url, settings.ollama_embed_health_timeout)

    previous_current_dir = settings.embeddings_current_symlink if settings.embeddings_current_symlink.exists() else None
    previous_id_map = read_previous_id_map(previous_current_dir)

    if not healthy:
        log.warning(
            "embed service unhealthy at %s — skipping embed-changed entirely (degraded mode); "
            "similar_approach falls back to the exact-match scorer this generation",
            settings.ollama_embed_base_url,
        )
        # Degraded: carry forward whatever was last published, verbatim, so a
        # transient outage doesn't erase the whole index; nodes added since
        # then simply have no vector until a healthy run recomputes them.
        vectors_by_repo: dict[str, dict[str, list[float]]] = {}
        shards_by_repo: dict[str, dict[str, dict]] = {}
        for repo_id in graphs_by_repo:
            prev_shard = read_previous_shard(previous_current_dir, repo_id)
            shards_by_repo[repo_id] = prev_shard
            vectors_by_repo[repo_id] = {k: v["embedding"] for k, v in prev_shard.items() if v.get("embedding")}
        return EmbeddingStageResult(
            status=EMBED_DEGRADED,
            vectors_by_repo=vectors_by_repo,
            shards_by_repo=shards_by_repo,
            reason="embed health check failed",
            id_map=previous_id_map,
        )

    stats = EmbeddingStats()
    vectors_by_repo = {}
    all_keys: set[str] = set()
    staged_shards: dict[str, dict] = {}

    total_repos = len(graphs_by_repo)
    for i, (repo_id, graph_data) in enumerate(graphs_by_repo.items(), start=1):
        source_root = repo_roots_by_id.get(repo_id)
        previous_shard = read_previous_shard(previous_current_dir, repo_id)

        if repo_id in unchanged_repo_ids and previous_shard:
            # WS1's own changed-detection (decide_action -> ACTION_SKIP)
            # already told us this repo is untouched this run — reuse its
            # whole previous shard rather than re-reading/re-hashing every
            # node in it.
            log.info("[%d/%d] %s: reusing previous shard (unchanged) ...", i, total_repos, repo_id)
            shard = previous_shard
            stats.reused_repos_unchanged += 1
        else:
            log.info("[%d/%d] %s: computing shard ...", i, total_repos, repo_id)
            shard = compute_repo_shard(
                repo_id,
                graph_data,
                source_root,
                previous_shard,
                settings.ollama_embed_base_url,
                settings.ollama_embed_model,
                stats,
            )

        staged_shards[repo_id] = shard
        all_keys.update(shard.keys())
        vectors_by_repo[repo_id] = {k: v["embedding"] for k, v in shard.items() if v.get("embedding")}

    id_map = build_id_map(previous_id_map, all_keys, provisional_generation_id)
    stats.tombstoned = sum(1 for v in id_map.values() if v.get("status") == "tombstoned")

    recipe = EmbeddingRecipe(model=settings.ollama_embed_model, dim=_infer_dim(vectors_by_repo))

    return EmbeddingStageResult(
        status=EMBED_HEALTHY,
        vectors_by_repo=vectors_by_repo,
        shards_by_repo=staged_shards,
        stats=stats,
        recipe=recipe,
        id_map=id_map,
    )


def _infer_dim(vectors_by_repo: dict[str, dict[str, list[float]]]) -> int:
    from graphify_mesh.sync.config import EMBED_DEFAULT_DIM

    for shard in vectors_by_repo.values():
        for vector in shard.values():
            if vector:
                return len(vector)
    return EMBED_DEFAULT_DIM


def stage_embeddings(staging_root: Path, staged_shards_source: dict[str, dict], id_map: dict[str, dict]) -> Path:
    """Write this run's shards + id-map into the ephemeral per-run staging
    tempdir. Nothing here touches the real, persistent embeddings_dir — see
    `persist_generation`."""
    out_dir = staging_root / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    for repo_id, entries in staged_shards_source.items():
        shard_path = out_dir / f"{repo_id}.json"
        shard_path.write_text(json.dumps({"repo_id": repo_id, "entries": entries}), encoding="utf-8")
    (out_dir / "id-map.json").write_text(json.dumps(id_map, indent=2, sort_keys=True), encoding="utf-8")
    return out_dir


def persist_generation(embeddings_dir: Path, generation_id: str, staged_dir: Path, keep: int) -> None:
    """Called only after a successful publish (mirrors publish.flip_current):
    copies the staged shard files into
    `embeddings_dir/generations/<generation_id>/`, flips
    `embeddings_dir/current` to point at it, then GCs old generations,
    keeping only the last `keep`."""
    generations_dir = embeddings_dir / "generations"
    generations_dir.mkdir(parents=True, exist_ok=True)
    gen_dir = generations_dir / generation_id
    if gen_dir.exists():
        shutil.rmtree(gen_dir)
    shutil.copytree(staged_dir, gen_dir)

    current = embeddings_dir / "current"
    tmp_link = embeddings_dir / ".current.tmp"
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    tmp_link.symlink_to(gen_dir.resolve(), target_is_directory=True)
    import os

    os.rename(str(tmp_link), str(current))

    gc_old_generations(generations_dir, keep)


def gc_old_generations(generations_dir: Path, keep: int) -> list[str]:
    """Keep only the `keep` most-recently-created generation dirs (sorted by
    directory name, which is the pipeline's lexicographically-sortable
    timestamp-prefixed generation_id — same ordering convention as
    `publish.write_generation`). Returns the list of removed generation_ids."""
    if not generations_dir.is_dir():
        return []
    gen_dirs = sorted((p for p in generations_dir.iterdir() if p.is_dir()), key=lambda p: p.name)
    to_remove = gen_dirs[:-keep] if keep > 0 else gen_dirs
    removed = []
    for gen_dir in to_remove:
        shutil.rmtree(gen_dir, ignore_errors=True)
        removed.append(gen_dir.name)
    return removed
