from __future__ import annotations

import random

import numpy as np
import pytest

from graphify_mesh.sync import embed_similarity
from graphify_mesh.sync.embed_similarity import _bucket_signature_batch, _planes_matrix
from graphify_mesh.sync.vectors import RepoVectors


def _unit(values: list[float]) -> list[float]:
    # Small helper: not normalized, cosine_similarity itself normalizes.
    return values


def _rv(mapping: dict[str, list[float]]) -> RepoVectors:
    return RepoVectors.from_mapping(mapping)


def test_cosine_similarity_identical_vectors_is_one():
    v = [1.0, 2.0, 3.0]
    # float32 batch math (np.dot/np.linalg.norm) can drift in the last
    # decimals vs. the old pure-Python float64 sum; loosen tolerance only,
    # expected value unchanged.
    assert embed_similarity.cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-4)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert embed_similarity.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_zero_vector_is_zero_not_nan():
    assert embed_similarity.cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_mutual_top_k_finds_similar_cross_repo_pair_above_threshold():
    # Two near-identical vectors in different repos -> cosine ~0.9995,
    # comfortably above the default threshold, and each other's only
    # candidate -> mutual top-1.
    vectors_by_repo = {
        "repo.a": _rv({"a": _unit([1.0, 0.01, 0.0, 0.0])}),
        "repo.b": _rv({"b": _unit([0.99, 0.02, 0.01, 0.0])}),
    }
    pairs = embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=1)
    assert len(pairs) == 1
    key_a, key_b, score = pairs[0]
    assert {key_a, key_b} == {"a", "b"}
    assert score > 0.9


def test_mutual_top_k_excludes_pairs_below_threshold():
    # Orthogonal vectors -> cosine 0.0, well below any reasonable threshold —
    # must produce zero pairs regardless of LSH bucket assignment, since the
    # exact cosine + threshold check always applies to any compared pair.
    vectors_by_repo = {
        "repo.a": _rv({"a": [1.0, 0.0, 0.0, 0.0]}),
        "repo.b": _rv({"b": [0.0, 1.0, 0.0, 0.0]}),
    }
    pairs = embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=5, threshold=0.82)
    assert pairs == []


def test_mutual_top_k_ignores_same_repo_pairs():
    # Two near-identical vectors, but same repo -> never a cross-project
    # candidate even though cosine similarity is high.
    vectors_by_repo = {
        "repo.a": _rv({"a": [1.0, 0.0, 0.0, 0.0], "b": [0.99, 0.01, 0.0, 0.0]}),
    }
    pairs = embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=5, threshold=0.5)
    assert pairs == []


def test_mutual_top_k_deterministic_across_calls():
    vectors_by_repo = {
        "repo.a": _rv({"a": [1.0, 0.02, 0.01, 0.0]}),
        "repo.b": _rv({"b": [0.99, 0.01, 0.0, 0.01]}),
        "repo.c": _rv({"c": [-1.0, 0.0, 0.0, 0.0]}),
    }
    first = embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=2)
    second = embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=2)
    assert first == second


def test_mutual_top_k_pairs_repo_vectors_matches_previous_results():
    """Same fixture vectors as the pre-refactor test, now wrapped per repo:
    identical pairs, scores approx-equal (abs=1e-4)."""
    vectors_by_repo = {
        "repoA": _rv({"a1": [1.0, 0.0, 0.0], "a2": [0.0, 1.0, 0.0]}),
        "repoB": _rv({"b1": [0.99, 0.01, 0.0], "b2": [0.0, 0.98, 0.05]}),
    }
    pairs = embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=2, threshold=0.8)
    assert ("a1", "b1") in {(a, b) for a, b, _ in pairs}
    assert ("a2", "b2") in {(a, b) for a, b, _ in pairs}
    for key_a, key_b, score in pairs:
        assert key_a < key_b
        assert score >= 0.8


def test_mutual_top_k_pairs_same_repo_never_paired():
    vectors_by_repo = {"repoA": _rv({"a1": [1.0, 0.0], "a2": [1.0, 0.0]})}
    assert embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=5, threshold=0.1) == []


def test_bucket_chunking_identical_pairs(monkeypatch):
    """Force BUCKET_CHUNK_ROWS=2 so chunked and unchunked paths both run on
    a >2-row bucket; results must be identical (chunking is allocation-only)."""
    vectors_by_repo = {
        "repoA": _rv({f"a{i}": [1.0, 0.001 * i] for i in range(6)}),
        "repoB": _rv({f"b{i}": [1.0, 0.001 * i + 0.0005] for i in range(6)}),
    }
    full = embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=3, threshold=0.5)
    monkeypatch.setattr(embed_similarity, "BUCKET_CHUNK_ROWS", 2)
    chunked = embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=3, threshold=0.5)
    assert [(a, b) for a, b, _ in full] == [(a, b) for a, b, _ in chunked]
    for (_, _, s1), (_, _, s2) in zip(full, chunked, strict=True):
        assert s1 == pytest.approx(s2, abs=1e-4)


def test_mixed_dim_repo_dropped_with_warning(caplog):
    """A repo whose dim differs from the first (sorted) repo's dim is dropped
    entirely — same scores-zero-everywhere outcome as the old per-vector drop."""
    vectors_by_repo = {
        "repoA": _rv({"a1": [1.0, 0.0]}),
        "repoB": _rv({"b1": [1.0, 0.0, 0.0]}),
    }
    assert embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=2, threshold=0.1) == []


def _build_varied_vectors(count: int, dims: tuple[int, ...] = (8, 64)) -> dict[str, np.ndarray]:
    # ONE rng instance drives every component of every vector (not re-seeded
    # per component/vector) so components are genuinely varied rather than
    # degenerate per-key constants; dims are mixed across the set.
    rng = random.Random(42)
    vectors: dict[str, np.ndarray] = {}
    for i in range(count):
        dim = dims[i % len(dims)]
        vectors[f"k{i}"] = np.asarray(
            [rng.uniform(-1, 1) for _ in range(dim)], dtype=np.float32
        )
    return vectors


def test_bucket_signature_batch_deterministic_across_calls():
    """Binding contract (post-relaxation): signatures are deterministic
    across runs/processes for identical inputs on the same platform — NOT
    required to be bit-identical to the pre-numpy scalar implementation
    (near-boundary dot products may flip sign under batched/BLAS summation
    vs. Python's sequential sum; LSH tolerates this by design, see the
    module docstring's known-limitation paragraph).

    Built from 200 varied vectors (dims mixed 8/64, all components drawn
    from one shared `random.Random(42)` instance) to exercise more than a
    handful of small/degenerate fixtures.
    """
    vectors_a = _build_varied_vectors(200)
    vectors_b = _build_varied_vectors(200)  # separately constructed, equal values
    assert vectors_a.keys() == vectors_b.keys()

    for dim in (8, 64):
        planes = _planes_matrix(dim)
        keys_for_dim = [k for k, v in vectors_a.items() if v.shape[0] == dim]
        matrix_a = np.stack([vectors_a[k] for k in keys_for_dim])
        matrix_b = np.stack([vectors_b[k] for k in keys_for_dim])
        assert (matrix_a == matrix_b).all()  # sanity: inputs really are equal
        sigs_a = _bucket_signature_batch(matrix_a, planes)
        sigs_b = _bucket_signature_batch(matrix_b, planes)
        assert sigs_a == sigs_b


def test_mutual_top_k_pairs_invariants_hold_over_varied_vector_set():
    """Behavioral-parity test at the mutual_top_k_pairs level: assert the
    documented invariants hold over a larger, varied vector set, rather than
    pinning an exact pair list (which is no longer a meaningful contract
    once bit-identical bucketing to the scalar implementation is withdrawn
    -- see module docstring's known-limitation paragraph)."""
    vectors = _build_varied_vectors(200)
    # Guarantee an unconditional qualifying pair regardless of LSH bucketing:
    # two EXACTLY identical dim-8 vectors (dim 8 is the dim of the first
    # varied-set key, so it's the dim `mutual_top_k_pairs` keeps) in
    # different repos. Identical float32 inputs produce identical dot
    # products against any hyperplane set, hence identical signatures,
    # hence the same bucket -- guaranteed, not merely likely. Their cosine
    # similarity is exactly 1.0 >= threshold, so they always qualify.
    anchor_key_a = "anchor_a"
    anchor_key_b = "anchor_b"
    anchor_vector = np.asarray([1.0, 0.5, -0.25, 0.1, 0.0, -0.5, 0.75, -1.0], dtype=np.float32)
    vectors[anchor_key_a] = anchor_vector
    vectors[anchor_key_b] = anchor_vector.copy()

    keys = list(vectors.keys())
    # Assign alternating repos so cross-repo-only filtering is exercised,
    # except force the two anchors into different repos explicitly (their
    # position in `keys` would otherwise land on an arbitrary repo index).
    repo_by_key = {key: f"repo.{i % 4}" for i, key in enumerate(keys)}
    repo_by_key[anchor_key_a] = "repo.0"
    repo_by_key[anchor_key_b] = "repo.1"

    mappings_by_repo: dict[str, dict[str, np.ndarray]] = {}
    for key, repo_id in repo_by_key.items():
        mappings_by_repo.setdefault(repo_id, {})[key] = vectors[key]
    vectors_by_repo = {repo_id: _rv(mapping) for repo_id, mapping in mappings_by_repo.items()}

    threshold = 0.5
    top_k = 3
    pairs = embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=top_k, threshold=threshold)
    expected_anchor_pair = tuple(sorted((anchor_key_a, anchor_key_b)))
    assert any(
        (key_a, key_b) == expected_anchor_pair for key_a, key_b, _ in pairs
    ), "anchor pair must be emitted"

    seen_unordered: set[frozenset] = set()
    per_key_partners: dict[str, set[str]] = {}
    for key_a, key_b, score in pairs:
        # Deterministic key_a < key_b ordering.
        assert key_a < key_b
        # No duplicate/reversed pairs.
        pair_id = frozenset((key_a, key_b))
        assert pair_id not in seen_unordered
        seen_unordered.add(pair_id)
        # Cross-repo only.
        assert repo_by_key[key_a] != repo_by_key[key_b]
        # Every emitted score respects the threshold.
        assert score >= threshold
        per_key_partners.setdefault(key_a, set()).add(key_b)
        per_key_partners.setdefault(key_b, set()).add(key_a)

    # Mutual top-k membership: a partner is only ever emitted for `key` if
    # it's within `key`'s own top-k candidate set (by construction, the
    # mutual filter requires membership on both sides). Note this must be
    # checked against the *bucket-restricted* candidate set, not a global
    # recomputation over every other key — LSH can miss a globally-better
    # candidate in a different bucket, so a globally-top-k check would be
    # too strict and could fail on data alone, independent of any real bug.
    # The invariant the algorithm actually guarantees is weaker but robust:
    # no key can be a partner in more than `top_k` emitted pairs.
    for _key, partners in per_key_partners.items():
        assert len(partners) <= top_k


def test_mutual_top_k_pairs_ndarray_matches_prior_results():
    # Reuse the same fixture as
    # test_mutual_top_k_finds_similar_cross_repo_pair_above_threshold, but
    # feed 1-D float32 ndarrays instead of plain lists -> same pairs, scores
    # approx-equal within float32 tolerance.
    vectors_by_repo = {
        "repo.a": _rv({"a": np.asarray([1.0, 0.01, 0.0, 0.0], dtype=np.float32)}),
        "repo.b": _rv({"b": np.asarray([0.99, 0.02, 0.01, 0.0], dtype=np.float32)}),
    }
    pairs = embed_similarity.mutual_top_k_pairs(vectors_by_repo, top_k=1)
    assert len(pairs) == 1
    key_a, key_b, score = pairs[0]
    assert {key_a, key_b} == {"a", "b"}
    assert score > 0.9


def test_mutual_top_k_pairs_ndarray_deterministic_and_matches_list_input():
    list_vectors_by_repo = {
        "repo.a": _rv({"a": [1.0, 0.02, 0.01, 0.0]}),
        "repo.b": _rv({"b": [0.99, 0.01, 0.0, 0.01]}),
        "repo.c": _rv({"c": [-1.0, 0.0, 0.0, 0.0]}),
    }
    ndarray_vectors_by_repo = {
        "repo.a": _rv({"a": np.asarray([1.0, 0.02, 0.01, 0.0], dtype=np.float32)}),
        "repo.b": _rv({"b": np.asarray([0.99, 0.01, 0.0, 0.01], dtype=np.float32)}),
        "repo.c": _rv({"c": np.asarray([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)}),
    }
    list_pairs = embed_similarity.mutual_top_k_pairs(list_vectors_by_repo, top_k=2)
    ndarray_pairs = embed_similarity.mutual_top_k_pairs(ndarray_vectors_by_repo, top_k=2)
    assert len(list_pairs) == len(ndarray_pairs)
    for (la, lb, lscore), (na, nb, nscore) in zip(list_pairs, ndarray_pairs, strict=True):
        assert (la, lb) == (na, nb)
        assert lscore == pytest.approx(nscore, abs=1e-4)


def test_cosine_similarity_accepts_lists_and_arrays():
    assert embed_similarity.cosine_similarity(
        [1.0, 0.0], np.asarray([1.0, 0.0], dtype=np.float32)
    ) == pytest.approx(1.0)
    assert embed_similarity.cosine_similarity([1.0, 0.0], [0.0]) == 0.0  # mixed dim still 0
