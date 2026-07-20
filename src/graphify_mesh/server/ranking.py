"""WS5 fusion + ranking contract: pinned RRF constants, hub/DEPRECATED
penalties, and MMR diversification. Every constant here is named and
documented, per project style rule (no magic numbers).

Fusion contract (plan WS5 bullet 3):
  * Reciprocal Rank Fusion (RRF) combines however many retrievers actually
    returned candidates this call (lexical always; vector only if the
    generation has embeddings; structural/community always).
  * Exact FQCN/label hits (resolved via the lexical index's `alias_exact`
    table, see `retrieval.exact_alias_hits`) bypass fusion, the hub-degree
    penalty, and MMR entirely — they are ranked first, deterministically.
  * If a retriever has nothing to contribute this generation (most notably:
    no embeddings at all, cold start / Ollama down during the sync run),
    RRF simply sums over whichever retrievers DID return something — there
    is no per-retriever normalization constant that assumes a fixed retriever
    count, so "renormalizing over available retrievers" falls out for free;
    the caller is responsible for surfacing the `degraded` flag (e.g.
    `"embeddings_unavailable"`) so this is visible to the client, never
    silently absorbed.
  * Deterministic tie-breaking: candidates with an identical fused score are
    ordered by their logical-ref string (`repo\x1fsource_file\x1flabel`),
    so repeated identical queries return identical order regardless of dict
    iteration order.
"""

from __future__ import annotations

# --- RRF (Reciprocal Rank Fusion) ------------------------------------------

# Standard RRF smoothing constant (Cormack et al. 2009's k=60 is the
# widely-cited default; kept as a named, overridable constant rather than a
# bare literal at each call site).
RRF_K = 60

# Per-retriever candidate depth: how many top-ranked candidates each of the
# lexical/vector/structural retrievers contributes into the fusion pool
# before RRF combines them. Kept well above the largest supported `k` so
# fusion has enough material to diversify from.
CANDIDATE_DEPTH_LEXICAL = 50
CANDIDATE_DEPTH_VECTOR = 50
CANDIDATE_DEPTH_STRUCTURAL = 50

# --- Ranking penalties -------------------------------------------------

# Hub/degree penalty: nodes above this total-degree threshold are generic
# hubs (CacheKey/User/Where-style god-nodes that flooded every baseline
# probe — see graphify/baseline-2026-07-20.md probes 2/11) and get
# multiplicatively down-weighted rather than excluded outright (a real hit
# that happens to be a hub should still be findable, just not favored).
HUB_DEGREE_THRESHOLD = 50
HUB_PENALTY_FACTOR = 0.5

# DEPRECATED-path down-weight (baseline systemic failure #8: DEPRECATED
# code dominated gamestream API results).
DEPRECATED_PATH_MARKER = "/DEPRECATED/"
DEPRECATED_PENALTY_FACTOR = 0.3

# Confidence handling: INFERRED edges are excluded by default from
# traversal-based (structural) candidate generation; callers opt in via
# `include_inferred=True`. EXTRACTED is the graphify default confidence
# class for edges that don't carry an explicit `confidence` attribute (see
# graphify/export.py's `_CONFIDENCE_SCORE_DEFAULTS`).
CONFIDENCE_EXTRACTED = "EXTRACTED"
CONFIDENCE_INFERRED = "INFERRED"

# --- MMR (maximal marginal relevance) diversification -----------------

# Trade-off between relevance (fused score) and novelty vs already-selected
# results. 1.0 = pure relevance (no diversification), 0.0 = pure novelty.
MMR_LAMBDA = 0.7

# --- Tool contracts -----------------------------------------------------

MAX_K = 100
DEFAULT_K = 10
PAGE_SIZE = 20


def rrf_contribution(rank: int) -> float:
    """`rank` is 0-indexed position within one retriever's ranked list."""
    return 1.0 / (RRF_K + rank + 1)


def fuse_rankings(rankings: dict[str, list[str]]) -> dict[str, float]:
    """`rankings`: retriever name -> its ranked list of doc keys (best
    first). Returns doc key -> summed RRF score. Retrievers that returned
    nothing simply don't contribute — no fixed-N normalization assumption."""
    scores: dict[str, float] = {}
    for ranked in rankings.values():
        for idx, key in enumerate(ranked):
            scores[key] = scores.get(key, 0.0) + rrf_contribution(idx)
    return scores


def apply_penalties(
    doc_key: str,
    base_score: float,
    degree_by_key: dict[str, int],
    path_by_key: dict[str, str],
) -> float:
    """Multiplicative hub-degree + DEPRECATED-path penalties. Never applied
    to exact-alias bypass hits (those skip this function entirely — see
    `retrieval.rank_candidates`)."""
    score = base_score
    if degree_by_key.get(doc_key, 0) > HUB_DEGREE_THRESHOLD:
        score *= HUB_PENALTY_FACTOR
    path = path_by_key.get(doc_key, "") or ""
    if DEPRECATED_PATH_MARKER in path:
        score *= DEPRECATED_PENALTY_FACTOR
    return score


def _structural_similarity_proxy(key_a: str, key_b: str, path_by_key: dict[str, str]) -> float:
    """Cheap, dependency-free doc-doc similarity proxy for MMR, used
    instead of vector cosine so diversification still functions when
    embeddings are unavailable (degraded mode, C9 cold start). Same source
    file => high similarity (near-duplicate results from the same file);
    otherwise 0. Deterministic and requires nothing beyond data already
    loaded for ranking."""
    path_a = path_by_key.get(key_a)
    path_b = path_by_key.get(key_b)
    if path_a and path_a == path_b:
        return 0.9
    return 0.0


def mmr_select(
    scored_candidates: list[tuple[str, float]],
    path_by_key: dict[str, str],
    k: int,
    lam: float = MMR_LAMBDA,
) -> list[str]:
    """Greedy MMR selection over an already-scored candidate pool.
    Deterministic: ties broken by the candidate's own key (sorted lexical-
    ref string) at every selection step."""
    pool = sorted(scored_candidates, key=lambda c: (-c[1], c[0]))
    selected: list[str] = []
    while pool and len(selected) < k:
        if not selected:
            chosen = pool[0]
        else:

            def mmr_value(candidate: tuple[str, float]) -> float:
                key, score = candidate
                max_sim = max(
                    (_structural_similarity_proxy(key, s, path_by_key) for s in selected),
                    default=0.0,
                )
                return lam * score - (1 - lam) * max_sim

            # Explicit deterministic tie-break: highest mmr_value wins;
            # ties broken by ascending key string, never by pool iteration
            # order (`sorted` + first-element selection is stable and
            # order-independent, unlike relying on `max()`'s left-to-right
            # tie behavior over an unsorted pool).
            ranked_pool = sorted(pool, key=lambda c: (-mmr_value(c), c[0]))
            chosen = ranked_pool[0]
        selected.append(chosen[0])
        pool.remove(chosen)
    return selected


def deterministic_sort_key(logical_ref_key: str) -> str:
    """Canonical tie-break key: the logical ref string itself
    (`repo\x1fsource_file\x1fqualified_label`, same shape as
    `overlay_refs.LogicalRef.to_key()`)."""
    return logical_ref_key
