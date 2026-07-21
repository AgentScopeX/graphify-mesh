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

import logging
import random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from graphify_mesh.sync.vectors import RepoVectors

log = logging.getLogger(__name__)

# Untuned default (C7) — real tuning needs labeled similar/dissimilar node
# pairs, which do not exist yet. Revisit once WS7's eval harness collects
# some.
SIMILARITY_THRESHOLD_DEFAULT = 0.82

LSH_NUM_HYPERPLANES = 12
LSH_SEED = 1337

# Cap on rows per sims-matrix chunk inside one LSH bucket: bounds the
# transient b x chunk float32 product (a hot 4096-row bucket would
# otherwise allocate a 64MB b x b matrix in one shot).
BUCKET_CHUNK_ROWS = 1024


def cosine_similarity(
    a: list[float] | np.ndarray, b: list[float] | np.ndarray
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
    vectors_by_repo: dict[str, RepoVectors],
    top_k: int,
    threshold: float = SIMILARITY_THRESHOLD_DEFAULT,
    num_planes: int = LSH_NUM_HYPERPLANES,
    seed: int = LSH_SEED,
) -> list[tuple[str, str, float]]:
    """Cross-repo-only mutual top-k similarity pairs.

    Args:
        vectors_by_repo: repo_id -> RepoVectors, the per-repo containers
            already resident from the embedding stage. `repo_id` is used to
            enforce the cross-repo-only constraint (WS4 ship order item 2)
            — same-repo pairs are never candidates regardless of
            similarity.
        top_k: cap on candidates kept per node before the mutual filter.
        threshold: minimum cosine similarity to be a candidate at all.

    Returns:
        List of (key_a, key_b, score) with key_a < key_b (deterministic
        ordering, no duplicate/reversed pairs), where each of key_a/key_b is
        in the other's top-k candidate list ("mutual" top-k) and the repos
        differ.
    """
    repo_ids = sorted(repo_id for repo_id, rv in vectors_by_repo.items() if len(rv))
    if not repo_ids:
        return []
    dim = vectors_by_repo[repo_ids[0]].dim

    # Repos whose dim differs from the first (sorted) repo's dim are dropped
    # entirely — same "scores zero everywhere" outcome as the old per-vector
    # drop, since cosine_similarity always rejected mixed dimensions.
    kept_repo_ids: list[str] = []
    for repo_id in repo_ids:
        rv = vectors_by_repo[repo_id]
        if rv.dim != dim:
            log.warning(
                "similar_approach: repo %s embedding dim %d != %d — dropped from ANN",
                repo_id,
                rv.dim,
                dim,
            )
            continue
        kept_repo_ids.append(repo_id)

    # Flattened row addressing: (repo_id, local_row) per global row, in
    # sorted-repo then sorted-key order — matches the old items-dict
    # insertion order exactly, so bucket contents and pair ordering are
    # stable across this refactor.
    row_refs: list[tuple[str, int]] = []
    keys: list[str] = []
    for repo_id in kept_repo_ids:
        rv = vectors_by_repo[repo_id]
        for local_row, key in enumerate(rv.keys):
            row_refs.append((repo_id, local_row))
            keys.append(key)
    if len(keys) < 2:
        return []

    planes = _planes_matrix(dim, num_planes, seed)
    # Signatures per repo straight off the RAW resident matrix — no combined
    # copy, and raw (not normalized) keeps bucket assignments identical to
    # the previous implementation.
    signatures: list[str] = []
    for repo_id in kept_repo_ids:
        signatures.extend(_bucket_signature_batch(vectors_by_repo[repo_id].matrix, planes))

    buckets: dict[str, list[int]] = {}
    for row_index, sig in enumerate(signatures):
        buckets.setdefault(sig, []).append(row_index)

    # Candidate generation: only within-bucket comparisons (ANN, not
    # all-pairs). Exact cosine + threshold filtering still applies to every
    # candidate pair actually compared, so scoring is never approximate.
    candidates: dict[str, list[tuple[str, float]]] = {key: [] for key in keys}
    for bucket_rows in buckets.values():
        if len(bucket_rows) < 2:
            continue
        # Gather normalized rows for this bucket only, from each repo's
        # cached normalized() matrix — the only allocation is bucket-sized.
        sub = np.stack(
            [
                vectors_by_repo[row_refs[r][0]].normalized()[row_refs[r][1]]
                for r in bucket_rows
            ]
        )
        for chunk_start in range(0, len(bucket_rows), BUCKET_CHUNK_ROWS):
            chunk_end = min(chunk_start + BUCKET_CHUNK_ROWS, len(bucket_rows))
            sims = sub[chunk_start:chunk_end] @ sub.T
            for ci in range(chunk_end - chunk_start):
                i = chunk_start + ci
                row_a = bucket_rows[i]
                key_a = keys[row_a]
                repo_a = row_refs[row_a][0]
                for j in range(i + 1, len(bucket_rows)):
                    row_b = bucket_rows[j]
                    repo_b = row_refs[row_b][0]
                    if repo_a == repo_b:
                        continue
                    score = float(sims[ci, j])
                    if score < threshold:
                        continue
                    key_b = keys[row_b]
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
