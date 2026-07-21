"""vector_candidates matrix-scoring parity: v1-loaded vs v2-loaded shards
must rank identically for the same query (Task 5 — server reads both shard
formats, scoring moved to `RepoVectors.normalized() @ query`)."""

from __future__ import annotations

import json

from graphify_mesh.server.retrieval import vector_candidates
from graphify_mesh.server.store import _load_embeddings
from graphify_mesh.sync.embedding import RepoShard, stage_embeddings
from graphify_mesh.sync.vectors import RepoVectors


def _repo_shard(raw_vectors: dict[str, list[float]]) -> RepoShard:
    vectors = RepoVectors.from_mapping(raw_vectors)
    entries = {key: {"content_hash": None, "row": None} for key in vectors.keys}
    for row, key in enumerate(vectors.keys):
        entries[key]["row"] = row
    return RepoShard(entries=entries, vectors=vectors)


def _fake_embed_query_fn(vector):
    return lambda query: vector


def test_vector_candidates_identical_ranking_v1_vs_v2_loaded_embeddings(tmp_path):
    raw = {
        "n1": [1.0, 0.0, 0.0],
        "n2": [0.0, 1.0, 0.0],
        "n3": [0.9, 0.1, 0.0],
        "n4": [0.0, 0.0, 1.0],
    }

    v1_dir = tmp_path / "v1"
    v1_dir.mkdir()
    (v1_dir / "repo.a.json").write_text(
        json.dumps(
            {
                "repo_id": "repo.a",
                "entries": {
                    key: {"content_hash": None, "embedding": vec} for key, vec in raw.items()
                },
            }
        ),
        encoding="utf-8",
    )

    v2_dir = stage_embeddings(tmp_path / "v2-staging", {"repo.a": _repo_shard(raw)}, id_map={})

    v1_embeddings = _load_embeddings(v1_dir)
    v2_embeddings = _load_embeddings(v2_dir)

    query = [1.0, 0.05, 0.0]
    v1_ranked, v1_degraded = vector_candidates(
        "q", v1_embeddings, None, _fake_embed_query_fn(query), depth=10
    )
    v2_ranked, v2_degraded = vector_candidates(
        "q", v2_embeddings, None, _fake_embed_query_fn(query), depth=10
    )

    assert v1_degraded is False
    assert v2_degraded is False
    assert v1_ranked == v2_ranked
    assert v1_ranked[0] == "n1"  # sanity: most-similar vector ranks first


def test_vector_candidates_no_embeddings_is_degraded():
    ranked, degraded = vector_candidates("q", {}, None, _fake_embed_query_fn([1.0]), depth=10)
    assert ranked == []
    assert degraded is True


def test_vector_candidates_embed_fn_failure_is_degraded():
    embeddings = {"repo.a": RepoVectors.from_mapping({"n1": [1.0, 0.0]})}
    ranked, degraded = vector_candidates("q", embeddings, None, lambda query: None, depth=10)
    assert ranked == []
    assert degraded is True


def test_vector_candidates_repo_filter_excludes_other_repos():
    embeddings = {
        "repo.a": RepoVectors.from_mapping({"a1": [1.0, 0.0]}),
        "repo.b": RepoVectors.from_mapping({"b1": [1.0, 0.0]}),
    }
    ranked, degraded = vector_candidates(
        "q", embeddings, frozenset({"repo.a"}), _fake_embed_query_fn([1.0, 0.0]), depth=10
    )
    assert degraded is False
    assert ranked == ["a1"]
