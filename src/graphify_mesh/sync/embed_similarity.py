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

import random

import numpy as np

# Untuned default (C7) — real tuning needs labeled similar/dissimilar node
# pairs, which do not exist yet. Revisit once WS7's eval harness collects
# some.
SIMILARITY_THRESHOLD_DEFAULT = 0.82

LSH_NUM_HYPERPLANES = 12
LSH_SEED = 1337


def cosine_similarity(
    a: "list[float] | np.ndarray", b: "list[float] | np.ndarray"
) -> float:
    # Mixed-dimension vectors (e.g. a shard embedded under two different
    # models) carry no comparable signal — score them 0 instead of raising.
    if len(a) != len(b):
        return 0.0
    vec_a = np.asarray(a, dtype=np.float32)
    vec_b = np.asarray(b, dtype=np.float32)
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def _planes_matrix(
    dim: int, num_planes: int = LSH_NUM_HYPERPLANES, seed: int = LSH_SEED
) -> np.ndarray:
    """Deterministic random hyperplanes for LSH bucketing.

    CRITICAL determinism constraint: hyperplane VALUES are generated via
    `random.Random(seed).gauss(0.0, 1.0)` in the exact loop order of the
    pre-numpy `_random_hyperplanes` (planes outer loop, dim inner loop),
    then converted with `np.asarray(..., dtype=np.float32)` — that part is
    still binding. The resulting bucket signatures are deterministic for
    identical inputs but are NOT guaranteed bit-identical to the retired
    pre-numpy scalar path (see the comment on `_bucket_signature_batch`).
    """
    rng = random.Random(seed)  # noqa: S311 - deterministic LSH hyperplanes, not cryptographic
    planes = [[rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(num_planes)]
    return np.asarray(planes, dtype=np.float32)  # shape (num_planes, dim)


def _bucket_signature_batch(matrix: np.ndarray, planes: np.ndarray) -> list[str]:
    # Signatures are deterministic for identical inputs on the same platform
    # (same process or a fresh one), but are NOT guaranteed bit-identical to
    # the pre-numpy scalar implementation: dot products near a hyperplane
    # boundary can flip sign under batched/BLAS summation vs. Python's
    # sequential sum, regardless of dtype. LSH tolerates this by design — a
    # boundary flip only changes which candidate pairs get compared; see
    # the module docstring's known-limitation paragraph on boundary misses.
    # matrix (n, dim) @ planes.T (dim, p) -> (n, p); sign bit per plane.
    dots = matrix @ planes.T
    bits = dots >= 0
    return ["".join("1" if b else "0" for b in row) for row in bits]


def mutual_top_k_pairs(
    items: dict[str, tuple[str, "list[float] | np.ndarray"]],
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

    # Stack all vectors in key order into one float32 matrix. Mixed-dim
    # vectors are dropped here — same "scores zero everywhere" outcome as
    # the scalar implementation, since cosine_similarity always rejected
    # mixed dimensions; dropping them up front just avoids a ragged stack.
    dim = len(items[keys[0]][1])
    kept_keys = [key for key in keys if len(items[key][1]) == dim]
    if len(kept_keys) < 2:
        return []

    repo_by_key = {key: items[key][0] for key in kept_keys}
    matrix = np.asarray([items[key][1] for key in kept_keys], dtype=np.float32)

    # L2-normalize rows once; zero-norm rows normalize to zero vectors so
    # they never contribute a nonzero similarity to anything.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    normalized = np.where(norms == 0.0, 0.0, matrix / safe_norms)

    planes = _planes_matrix(dim, num_planes, seed)
    signatures = _bucket_signature_batch(matrix, planes)

    buckets: dict[str, list[int]] = {}
    for row_index, sig in enumerate(signatures):
        buckets.setdefault(sig, []).append(row_index)

    # Candidate generation: only within-bucket comparisons (ANN, not
    # all-pairs). Exact cosine + threshold filtering still applies to every
    # candidate pair actually compared, so scoring is never approximate.
    candidates: dict[str, list[tuple[str, float]]] = {key: [] for key in kept_keys}
    for bucket_rows in buckets.values():
        if len(bucket_rows) < 2:
            continue
        sub = normalized[bucket_rows]
        sims = sub @ sub.T
        for i, row_a in enumerate(bucket_rows):
            key_a = kept_keys[row_a]
            repo_a = repo_by_key[key_a]
            for j in range(i + 1, len(bucket_rows)):
                row_b = bucket_rows[j]
                key_b = kept_keys[row_b]
                repo_b = repo_by_key[key_b]
                if repo_a == repo_b:
                    continue
                score = float(sims[i, j])
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
