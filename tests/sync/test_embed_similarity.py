from __future__ import annotations

from graphify_mesh.sync import embed_similarity


def _unit(values: list[float]) -> list[float]:
    # Small helper: not normalized, cosine_similarity itself normalizes.
    return values


def test_cosine_similarity_identical_vectors_is_one():
    v = [1.0, 2.0, 3.0]
    assert embed_similarity.cosine_similarity(v, v) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert embed_similarity.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_zero_vector_is_zero_not_nan():
    assert embed_similarity.cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_mutual_top_k_finds_similar_cross_repo_pair_above_threshold():
    # Two near-identical vectors in different repos -> cosine ~0.9995,
    # comfortably above the default threshold, and each other's only
    # candidate -> mutual top-1.
    items = {
        "a": ("repo.a", _unit([1.0, 0.01, 0.0, 0.0])),
        "b": ("repo.b", _unit([0.99, 0.02, 0.01, 0.0])),
    }
    pairs = embed_similarity.mutual_top_k_pairs(items, top_k=1)
    assert len(pairs) == 1
    key_a, key_b, score = pairs[0]
    assert {key_a, key_b} == {"a", "b"}
    assert score > 0.9


def test_mutual_top_k_excludes_pairs_below_threshold():
    # Orthogonal vectors -> cosine 0.0, well below any reasonable threshold —
    # must produce zero pairs regardless of LSH bucket assignment, since the
    # exact cosine + threshold check always applies to any compared pair.
    items = {
        "a": ("repo.a", [1.0, 0.0, 0.0, 0.0]),
        "b": ("repo.b", [0.0, 1.0, 0.0, 0.0]),
    }
    pairs = embed_similarity.mutual_top_k_pairs(items, top_k=5, threshold=0.82)
    assert pairs == []


def test_mutual_top_k_ignores_same_repo_pairs():
    # Two near-identical vectors, but same repo -> never a cross-project
    # candidate even though cosine similarity is high.
    items = {
        "a": ("repo.a", [1.0, 0.0, 0.0, 0.0]),
        "b": ("repo.a", [0.99, 0.01, 0.0, 0.0]),
    }
    pairs = embed_similarity.mutual_top_k_pairs(items, top_k=5, threshold=0.5)
    assert pairs == []


def test_mutual_top_k_deterministic_across_calls():
    items = {
        "a": ("repo.a", [1.0, 0.02, 0.01, 0.0]),
        "b": ("repo.b", [0.99, 0.01, 0.0, 0.01]),
        "c": ("repo.c", [-1.0, 0.0, 0.0, 0.0]),
    }
    first = embed_similarity.mutual_top_k_pairs(items, top_k=2)
    second = embed_similarity.mutual_top_k_pairs(items, top_k=2)
    assert first == second
