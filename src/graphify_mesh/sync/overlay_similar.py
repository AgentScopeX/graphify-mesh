"""WS4 ship-order item 2: `similar_approach` overlay edges — now backed by
the WS3 embedding index for real ANN mutual top-k cosine similarity.

Two scoring paths, both cross-repo only (same-repo pairs are never a
cross-project relation):

  1. Real path (`embedding_vectors_by_repo` has vectors for a node): mutual
     top-k ANN cosine similarity over the embedding index
     (`embed_similarity.mutual_top_k_pairs`), threshold + top_k as named,
     documented-untuned config (C7).

  2. Fallback path (a node has no vector at all — either the WS3 embed stage
     is degraded/not run this generation, or the node was skipped by WS3's
     trivial-accessor heuristic, `embedding.is_trivial_node`): the original
     WS4 placeholder scorer — exact normalized-label + same-`community_name`
     match. This is the documented `find_similar` fallback for
     trivial/unembedded nodes (deliverable 7): no vector lookup is attempted
     for them, ever; a structural exact-match is used instead so they are
     never silently left with zero similarity candidates.

Per-pair cap: in addition to the per-node `top_k` cap, no more than
`MAX_EDGES_PER_REPO_PAIR` similar_approach edges are ever emitted between
the same two repos in one generation, so two large/near-duplicate repos
can't flood the overlay with edges (WS4 ship-order item 2: "per-pair cap").
"""

from __future__ import annotations

from graphify_mesh.sync.embed_similarity import (
    SIMILARITY_THRESHOLD_DEFAULT,
    mutual_top_k_pairs,
)
from graphify_mesh.sync.embedding import key_to_ref, node_key
from graphify_mesh.sync.overlay_refs import LogicalRef, OverlayEdge
from graphify_mesh.sync.vectors import RepoVectors

PLACEHOLDER_PROVENANCE = "PLACEHOLDER_STRUCTURAL_MATCH"
PLACEHOLDER_CONFIDENCE = 0.4
EMBEDDING_PROVENANCE = "EXTRACTED_EMBEDDING_ANN_COSINE"

# WS4 ship-order item 2: cap on similar_approach edges between any single
# pair of repos, independent of the per-node top_k cap.
MAX_EDGES_PER_REPO_PAIR = 50


def _normalize_label(label: str) -> str:
    return label.strip().lower()


def normalize_label(label: str) -> str:
    """Public wrapper around `_normalize_label` so callers outside this
    module (WS5's `graphify-mesh` MCP server, `find_similar`'s trivial/unembedded
    fallback path) reuse the IDENTICAL normalization this module's own
    exact-match scorer uses, rather than re-deriving it."""
    return _normalize_label(label)


def _fallback_exact_match_pairs(
    graphs_by_repo: dict[str, dict],
    excluded_keys: set[str],
    top_k: int,
) -> list[OverlayEdge]:
    """Original WS4 placeholder scorer, restricted to nodes NOT already
    covered by the real embedding path (`excluded_keys`). See module
    docstring path 2. Exact label+community match, mutual top-k trivially
    satisfied since candidacy here is symmetric and unranked."""
    buckets: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    for repo_id, graph_data in graphs_by_repo.items():
        for node in graph_data.get("nodes", []):
            if not isinstance(node, dict):
                continue
            label = node.get("label")
            community_name = node.get("community_name")
            source_file = node.get("source_file")
            if not label or not community_name or not source_file:
                continue
            key = node_key(repo_id, node)
            if key is not None and key in excluded_keys:
                continue
            bucket_key = (_normalize_label(label), community_name)
            buckets.setdefault(bucket_key, []).append((repo_id, source_file, label))

    edges: list[OverlayEdge] = []
    per_node_emitted: dict[tuple[str, str, str], int] = {}
    per_repo_pair_emitted: dict[frozenset, int] = {}
    for (norm_label, community_name), members in buckets.items():
        distinct_repos = {m[0] for m in members}
        if len(distinct_repos) < 2:
            continue
        seen_pairs: set[frozenset] = set()
        for i, (repo_a, file_a, label_a) in enumerate(members):
            for repo_b, file_b, label_b in members[i + 1 :]:
                if repo_a == repo_b:
                    continue
                node_a_key = (repo_a, file_a, label_a)
                node_b_key = (repo_b, file_b, label_b)
                pair_key = frozenset((node_a_key, node_b_key))
                if pair_key in seen_pairs:
                    continue
                if (
                    per_node_emitted.get(node_a_key, 0) >= top_k
                    or per_node_emitted.get(node_b_key, 0) >= top_k
                ):
                    continue
                repo_pair_key = frozenset((repo_a, repo_b))
                if per_repo_pair_emitted.get(repo_pair_key, 0) >= MAX_EDGES_PER_REPO_PAIR:
                    continue
                seen_pairs.add(pair_key)
                per_node_emitted[node_a_key] = per_node_emitted.get(node_a_key, 0) + 1
                per_node_emitted[node_b_key] = per_node_emitted.get(node_b_key, 0) + 1
                per_repo_pair_emitted[repo_pair_key] = (
                    per_repo_pair_emitted.get(repo_pair_key, 0) + 1
                )
                edges.append(
                    OverlayEdge(
                        type="similar_approach",
                        source=LogicalRef(repo=repo_a, source_file=file_a, qualified_label=label_a),
                        target=LogicalRef(repo=repo_b, source_file=file_b, qualified_label=label_b),
                        provenance=PLACEHOLDER_PROVENANCE,
                        confidence=PLACEHOLDER_CONFIDENCE,
                        evidence=(
                            "placeholder fallback scorer (no embedding available): "
                            "exact label+community match "
                            f"({norm_label!r} in {community_name!r})"
                        ),
                    )
                )
    return edges


def _embedding_pairs(
    embedding_vectors_by_repo: dict[str, RepoVectors],
    top_k: int,
    threshold: float,
    embedding_model: str,
) -> tuple[list[OverlayEdge], set[str]]:
    total = sum(len(rv) for rv in embedding_vectors_by_repo.values())
    all_keys = {key for rv in embedding_vectors_by_repo.values() for key in rv.keys}
    if total < 2:
        return [], all_keys

    pairs = mutual_top_k_pairs(embedding_vectors_by_repo, top_k=top_k, threshold=threshold)

    edges: list[OverlayEdge] = []
    per_repo_pair_emitted: dict[frozenset, int] = {}
    for key_a, key_b, score in pairs:
        ref_a = key_to_ref(key_a)
        ref_b = key_to_ref(key_b)
        repo_pair_key = frozenset((ref_a.repo, ref_b.repo))
        if per_repo_pair_emitted.get(repo_pair_key, 0) >= MAX_EDGES_PER_REPO_PAIR:
            continue
        per_repo_pair_emitted[repo_pair_key] = per_repo_pair_emitted.get(repo_pair_key, 0) + 1
        edges.append(
            OverlayEdge(
                type="similar_approach",
                source=ref_a,
                target=ref_b,
                provenance=EMBEDDING_PROVENANCE,
                confidence=round(score, 4),
                evidence=(
                    f"embedding ANN cosine similarity={score:.4f} model={embedding_model!r} "
                    f"threshold={threshold} (mutual top-{top_k}, LSH-bucketed candidates)"
                ),
            )
        )
    return edges, all_keys


def compute_similar_approach_edges(
    graphs_by_repo: dict[str, dict],
    embedding_vectors_by_repo: dict[str, RepoVectors] | None = None,
    top_k: int = 5,
    threshold: float = SIMILARITY_THRESHOLD_DEFAULT,
    embedding_model: str = "unknown",
) -> list[OverlayEdge]:
    """Mutual top-k cross-project similarity (WS4 item 2), backed by the WS3
    embedding index when available, falling back to the exact-match
    placeholder scorer for nodes without a vector (see module docstring).

    Args:
        graphs_by_repo: repo_id -> that repo's raw per-repo graph.json data.
        embedding_vectors_by_repo: repo_id -> RepoVectors, produced by
            `embedding.run_embedding_stage` this generation. `None` or empty
            means no embedding index is available at all this run (e.g. the
            embed stage was degraded) — every node falls back to the
            exact-match scorer, reproducing the pre-WS3 placeholder
            behavior exactly.
        top_k: cap on edges emitted per node (both scoring paths).
        threshold: cosine similarity cutoff for the embedding path (C7:
            named, documented-untuned default; see embed_similarity.py).
    """
    embedding_edges: list[OverlayEdge] = []
    embedded_keys: set[str] = set()
    if embedding_vectors_by_repo:
        embedding_edges, embedded_keys = _embedding_pairs(
            embedding_vectors_by_repo, top_k, threshold, embedding_model
        )

    fallback_edges = _fallback_exact_match_pairs(graphs_by_repo, embedded_keys, top_k)
    return embedding_edges + fallback_edges
