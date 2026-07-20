"""WS3 ANN mutual-top-k similarity over the embedding index (C7).

C7 forbids all-pairs (O(n^2)) similarity computation. This module implements
a small, dependency-free approximate nearest-neighbor mechanism instead:
random-hyperplane locality-sensitive hashing (LSH) for cosine similarity.
Each vector is hashed to a bucket signature (one bit per hyperplane, sign of
the dot product); only vectors sharing a bucket are ever compared directly,
so the candidate-generation step is sub-quadratic instead of exhaustive.
Once two vectors ARE compared, the similarity score is exact cosine
similarity — LSH here only reduces *which pairs get compared*, it never
approximates the score itself. Hyperplanes are seeded deterministically
(`LSH_SEED`) so results are reproducible across runs and in tests.

Known limitation (documented, not hidden): a genuinely similar pair whose
vectors happen to sit near a hyperplane boundary can be hashed into
different buckets and therefore missed — this is the accepted approximate
nature of LSH. If node counts grow enough that this starts costing real
recall, swap in a real ANN library (faiss/hnswlib) behind the same
`mutual_top_k_pairs` signature; nothing downstream needs to change.

C7 also states the similarity threshold must be tuned on labeled pairs —
none exist yet (no labeled ground truth has been collected). Until they do,
`SIMILARITY_THRESHOLD_DEFAULT` below is a documented, untuned, best-guess
default; it is a named constant specifically so it's easy to find and
override once real tuning data exists (see `overlay_similar.py` callers).
"""

from __future__ import annotations

import math
import random

# Untuned default (C7) — real tuning needs labeled similar/dissimilar node
# pairs, which do not exist yet. Revisit once WS7's eval harness collects
# some.
SIMILARITY_THRESHOLD_DEFAULT = 0.82

LSH_NUM_HYPERPLANES = 12
LSH_SEED = 1337


def cosine_similarity(a: list[float], b: list[float]) -> float:
    # Mixed-dimension vectors (e.g. a shard embedded under two different
    # models) carry no comparable signal — score them 0 instead of raising.
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _random_hyperplanes(dim: int, num_planes: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)  # noqa: S311 - deterministic LSH hyperplanes, not cryptographic
    return [[rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(num_planes)]


def _bucket_signature(vector: list[float], hyperplanes: list[list[float]]) -> str:
    bits = []
    for plane in hyperplanes:
        # strict=False: a vector shorter than the plane (mixed-dim shard)
        # still buckets; final scoring goes through cosine_similarity which
        # rejects mixed dimensions.
        dot = sum(v * p for v, p in zip(vector, plane, strict=False))
        bits.append("1" if dot >= 0 else "0")
    return "".join(bits)


def mutual_top_k_pairs(
    items: dict[str, tuple[str, list[float]]],
    top_k: int,
    threshold: float = SIMILARITY_THRESHOLD_DEFAULT,
    num_planes: int = LSH_NUM_HYPERPLANES,
    seed: int = LSH_SEED,
) -> list[tuple[str, str, float]]:
    """Cross-repo-only mutual top-k similarity pairs.

    Args:
        items: key -> (repo_id, vector). `repo_id` is used to enforce the
            cross-repo-only constraint (WS4 ship order item 2) — same-repo
            pairs are never candidates regardless of similarity.
        top_k: cap on candidates kept per node before the mutual filter.
        threshold: minimum cosine similarity to be a candidate at all.

    Returns:
        List of (key_a, key_b, score) with key_a < key_b (deterministic
        ordering, no duplicate/reversed pairs), where each of key_a/key_b is
        in the other's top-k candidate list ("mutual" top-k) and the repos
        differ.
    """
    keys = list(items.keys())
    if len(keys) < 2:
        return []

    dim = len(next(iter(items.values()))[1])
    hyperplanes = _random_hyperplanes(dim, num_planes, seed)

    buckets: dict[str, list[str]] = {}
    for key, (_, vector) in items.items():
        sig = _bucket_signature(vector, hyperplanes)
        buckets.setdefault(sig, []).append(key)

    # Candidate generation: only within-bucket comparisons (ANN, not
    # all-pairs). Exact cosine + threshold filtering still applies to every
    # candidate pair actually compared, so scoring is never approximate.
    candidates: dict[str, list[tuple[str, float]]] = {key: [] for key in keys}
    for bucket_keys in buckets.values():
        for i, key_a in enumerate(bucket_keys):
            repo_a, vec_a = items[key_a]
            for key_b in bucket_keys[i + 1 :]:
                repo_b, vec_b = items[key_b]
                if repo_a == repo_b:
                    continue
                score = cosine_similarity(vec_a, vec_b)
                if score < threshold:
                    continue
                candidates[key_a].append((key_b, score))
                candidates[key_b].append((key_a, score))

    top_candidates: dict[str, set[str]] = {}
    for key, scored in candidates.items():
        scored.sort(key=lambda pair: pair[1], reverse=True)
        top_candidates[key] = {other for other, _ in scored[:top_k]}

    seen: set[frozenset] = set()
    pairs: list[tuple[str, str, float]] = []
    for key_a, scored in candidates.items():
        for key_b, score in scored:
            if key_b not in top_candidates[key_a] or key_a not in top_candidates[key_b]:
                continue
            pair_id = frozenset((key_a, key_b))
            if pair_id in seen:
                continue
            seen.add(pair_id)
            ordered = tuple(sorted((key_a, key_b)))
            pairs.append((ordered[0], ordered[1], score))

    return pairs
