from __future__ import annotations

from graphify_mesh.sync import validate


def test_dangling_id_detected():
    data = {
        "nodes": [{"id": "a"}, {"id": "b"}],
        "links": [{"source": "a", "target": "ghost"}],
    }
    result = validate.validate_dangling_ids(data)
    assert not result.ok
    assert any("ghost" in e for e in result.errors)


def test_dangling_id_clean():
    data = {"nodes": [{"id": "a"}, {"id": "b"}], "links": [{"source": "a", "target": "b"}]}
    result = validate.validate_dangling_ids(data)
    assert result.ok


def test_forbidden_edge_cross_repo_flag():
    data = {"nodes": [{"id": "a"}, {"id": "b"}], "links": [{"source": "a", "target": "b", "cross_repo": True}]}
    result = validate.validate_forbidden_edges(data)
    assert not result.ok


def test_forbidden_edge_relation_type():
    data = {
        "nodes": [{"id": "a"}, {"id": "b"}],
        "links": [{"source": "a", "target": "b", "relation": "semantically_similar_to"}],
    }
    result = validate.validate_forbidden_edges(data)
    assert not result.ok


def test_forbidden_edge_clean():
    data = {"nodes": [{"id": "a"}, {"id": "b"}], "links": [{"source": "a", "target": "b", "relation": "calls"}]}
    result = validate.validate_forbidden_edges(data)
    assert result.ok


def test_shrink_guard_refuses_smaller_graph():
    result = validate.validate_shrink_guard((5, 5), (10, 10), allow_shrink=False)
    assert not result.ok


def test_shrink_guard_allows_with_override():
    result = validate.validate_shrink_guard((5, 5), (10, 10), allow_shrink=True)
    assert result.ok


def test_community_name_placeholder_fails():
    data = {"nodes": [{"id": "a", "community": 0, "community_name": "Community 0"}]}
    result = validate.validate_community_names(data, skip_labeling=False)
    assert not result.ok


def test_community_name_skip_labeling_bypasses():
    data = {"nodes": [{"id": "a", "community": 0, "community_name": "Community 0"}]}
    result = validate.validate_community_names(data, skip_labeling=True)
    assert result.ok
    assert result.errors  # logged reason still present


def test_generation_manifest_missing_keys():
    result = validate.validate_generation_manifest({"generation_id": "x"})
    assert not result.ok
    assert len(result.errors) > 1
