"""WS5 candidate generation + fusion orchestration.

Scope filtering happens INSIDE each `*_candidates` function, before any
ranking/scoring is computed — never as a post-filter of an already-ranked
list (see `scope.py` module docstring for why that ordering matters).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from graphify_mesh.server import ranking
from graphify_mesh.server.store import Generation
from graphify_mesh.sync.embed_similarity import cosine_similarity
from graphify_mesh.sync.lexical_index import normalize_alias_query, tokenize_text

EmbedQueryFn = Callable[[str], list[float] | None]


@dataclass
class Hit:
    key: str
    repo: str
    source_file: str
    label: str
    node_id: str
    community_name: str | None
    degree: int
    score: float
    match_type: str  # "exact" | "fused"
    deprecated: bool


@dataclass
class RankedResult:
    hits: list[Hit] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)


def exact_alias_hits(query: str, lexical: dict, repo_filter: frozenset[str] | None) -> list[str]:
    """Exact FQCN/label bypass candidates (WS5 fusion contract): looked up
    via the lexical index's O(1) `alias_exact` table, normalized with the
    SAME normalization the index was built with (`normalize_alias_query`).

    Entries are the schema_version 2 compact `[repo, key]` array shape
    (not the old `{"repo": r, "key": k}` dict) — see `lexical_index.py`."""
    if not query or not query.strip():
        return []
    norm = normalize_alias_query(query.strip())
    entries = lexical.get("alias_exact", {}).get(norm, [])
    keys = {
        e[1]
        for e in entries
        if isinstance(e, list) and len(e) == 2 and (repo_filter is None or e[0] in repo_filter)
    }
    return sorted(keys)


def lexical_candidates(
    query: str,
    lexical: dict,
    repo_filter: frozenset[str] | None,
    depth: int = ranking.CANDIDATE_DEPTH_LEXICAL,
) -> list[str]:
    """Postings entries are the schema_version 2 compact `[repo, key, field]`
    array shape. `weight` is no longer stored per-entry — it is derived from
    the bundle's own `field_boosts` table so this reader stays decoupled
    from the writer's internal `FIELD_BOOSTS` constant name."""
    tokens = tokenize_text(query)
    if not tokens:
        return []
    postings = lexical.get("postings", {})
    doc_freq_global = lexical.get("doc_freq", {}).get("global", {})
    total_docs = max(lexical.get("document_count", 1), 1)
    field_boosts = lexical.get("field_boosts", {})

    scores: dict[str, float] = {}
    for term in tokens:
        entries = postings.get(term, [])
        df = doc_freq_global.get(term, len(entries)) or 1
        idf = math.log(1 + (total_docs / df))
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 3:
                # Malformed index entry: tolerate and skip, never crash.
                continue
            repo, key, field_name = entry
            if repo_filter is not None and repo not in repo_filter:
                continue
            if key is None:
                continue
            weight = field_boosts.get(field_name, 1.0)
            scores[key] = scores.get(key, 0.0) + weight * idf

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ranked[:depth]]


def vector_candidates(
    query: str,
    embeddings: dict[str, dict[str, list[float]]],
    repo_filter: frozenset[str] | None,
    embed_query_fn: EmbedQueryFn,
    depth: int = ranking.CANDIDATE_DEPTH_VECTOR,
) -> tuple[list[str], bool]:
    """Returns (ranked keys, degraded). `degraded=True` means the vector
    retriever contributed nothing this call (no embeddings published this
    generation, or the query-embed call failed) — the caller renormalizes
    fusion over whichever OTHER retrievers succeeded and must surface
    `"embeddings_unavailable"` in the response's `degraded` field."""
    if not embeddings:
        return [], True
    vector = embed_query_fn(query)
    if not vector:
        return [], True

    scored: list[tuple[str, float]] = []
    for repo_id, shard in embeddings.items():
        if repo_filter is not None and repo_id not in repo_filter:
            continue
        for key, vec in shard.items():
            scored.append((key, cosine_similarity(vector, vec)))
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in scored[:depth]], False


def structural_candidates(
    seed_keys: list[str],
    generation: Generation,
    repo_filter: frozenset[str] | None,
    include_inferred: bool = False,
    depth: int = ranking.CANDIDATE_DEPTH_STRUCTURAL,
) -> list[str]:
    """1-hop neighbors of `seed_keys` in the structural graph. INFERRED
    edges are excluded by default (only EXTRACTED-confidence edges count as
    traversal candidates unless the caller opts in) — baseline systemic
    failure #4 (INFERRED edge pollution)."""
    proximity: dict[str, int] = {}
    for seed_key in seed_keys:
        seed_id = generation.node_id_by_key.get(seed_key)
        if seed_id is None:
            continue
        for neighbor_id, edge in generation.adjacency.get(seed_id, []):
            confidence = edge.get("confidence", ranking.CONFIDENCE_EXTRACTED)
            if confidence == ranking.CONFIDENCE_INFERRED and not include_inferred:
                continue
            neighbor_key = generation.key_by_node_id.get(neighbor_id)
            if neighbor_key is None:
                continue
            neighbor_node = generation.node_by_id.get(neighbor_id, {})
            if repo_filter is not None and neighbor_node.get("repo") not in repo_filter:
                continue
            proximity[neighbor_key] = proximity.get(neighbor_key, 0) + 1

    ranked = sorted(proximity.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ranked[:depth]]


def _hit_from_key(key: str, generation: Generation, score: float, match_type: str) -> Hit | None:
    node_id = generation.node_id_by_key.get(key)
    if node_id is None:
        return None
    node = generation.node_by_id.get(node_id, {})
    source_file = node.get("source_file") or ""
    return Hit(
        key=key,
        repo=node.get("repo", ""),
        source_file=source_file,
        label=node.get("label", ""),
        node_id=node_id,
        community_name=node.get("community_name"),
        degree=generation.degree(node_id),
        score=score,
        match_type=match_type,
        deprecated=ranking.DEPRECATED_PATH_MARKER in source_file,
    )


def rank(
    query: str,
    generation: Generation,
    repo_filter: frozenset[str] | None,
    k: int,
    embed_query_fn: EmbedQueryFn,
    include_inferred: bool = False,
) -> RankedResult:
    """Top-level WS5 search orchestration: exact-alias bypass, then
    lexical+vector+structural candidate generation, RRF fusion, hub/
    DEPRECATED penalties, MMR diversification, deterministic tie-break."""
    k = max(1, min(k, ranking.MAX_K))
    degraded: list[str] = list(generation.manifest.get("_runtime_degraded", []))

    exact_keys = exact_alias_hits(query, generation.lexical, repo_filter)
    exact_hits = [
        h for h in (_hit_from_key(k_, generation, float("inf"), "exact") for k_ in exact_keys) if h
    ]
    exact_hits.sort(key=lambda h: h.key)
    selected_keys = {h.key for h in exact_hits}
    remaining_slots = max(0, k - len(exact_hits))

    fused_hits: list[Hit] = []
    if remaining_slots > 0:
        lexical_ranked = lexical_candidates(query, generation.lexical, repo_filter)
        vector_ranked, vec_degraded = vector_candidates(
            query, generation.embeddings, repo_filter, embed_query_fn
        )
        if vec_degraded and "embeddings_unavailable" not in degraded:
            degraded.append("embeddings_unavailable")

        seed_for_structural = (exact_keys + lexical_ranked)[:10]
        structural_ranked = structural_candidates(
            seed_for_structural, generation, repo_filter, include_inferred=include_inferred
        )

        rankings = {
            "lexical": lexical_ranked,
            "vector": vector_ranked,
            "structural": structural_ranked,
        }
        fused_scores = ranking.fuse_rankings(rankings)
        for excluded in selected_keys:
            fused_scores.pop(excluded, None)

        degree_by_key = {
            key: generation.degree(generation.node_id_by_key[key])
            for key in fused_scores
            if key in generation.node_id_by_key
        }
        path_by_key = {
            key: generation.node_by_id.get(generation.node_id_by_key.get(key, ""), {}).get(
                "source_file", ""
            )
            for key in fused_scores
            if key in generation.node_id_by_key
        }
        penalized = [
            (key, ranking.apply_penalties(key, score, degree_by_key, path_by_key))
            for key, score in fused_scores.items()
        ]
        diversified_keys = ranking.mmr_select(penalized, path_by_key, remaining_slots)
        final_scores = dict(penalized)
        for key in diversified_keys:
            hit = _hit_from_key(key, generation, final_scores.get(key, 0.0), "fused")
            if hit:
                fused_hits.append(hit)

    return RankedResult(hits=exact_hits + fused_hits, degraded=sorted(set(degraded)))
