from __future__ import annotations

from conftest import build_generation, key_for, make_link, make_node

from graphify_mesh.server import ranking
from graphify_mesh.server.project_map import project_map


def test_project_map_unresolved_when_repo_not_in_current_generation():
    gen = build_generation([make_node("repo.a", "Alpha", "src/alpha.py", node_id="a1")])
    result = project_map("repo.never-synced", gen)
    assert result.resolved is False
    assert "repo_not_in_current_generation" in result.degraded


def test_project_map_node_count_and_community_breakdown():
    nodes = [
        make_node("repo.a", "Alpha", "src/alpha.py", node_id="a1", community_name="commerce"),
        make_node("repo.a", "Beta", "src/beta.py", node_id="a2", community_name="commerce"),
        make_node("repo.a", "Gamma", "src/gamma.py", node_id="a3", community_name="billing"),
    ]
    gen = build_generation(nodes)
    result = project_map("repo.a", gen)
    assert result.resolved is True
    assert result.node_count == 3
    assert result.community_breakdown == {"commerce": 2, "billing": 1}


def test_project_map_top_hubs_sorted_by_degree_desc_deterministic():
    hub = make_node("repo.a", "HubNode", "src/hub.py", node_id="hub")
    leaves = [make_node("repo.a", f"Leaf{i}", f"src/leaf{i}.py", node_id=f"l{i}") for i in range(5)]
    links = [make_link("hub", leaf["id"]) for leaf in leaves]
    gen = build_generation([hub] + leaves, links=links)

    result = project_map("repo.a", gen)
    assert result.top_hubs[0]["key"] == key_for("repo.a", hub)
    assert result.top_hubs[0]["degree"] == 5
    assert result.top_hubs[0]["is_hub"] is (5 > ranking.HUB_DEGREE_THRESHOLD)


def test_project_map_marks_is_hub_flag_above_threshold():
    hub = make_node("repo.a", "BigHub", "src/hub.py", node_id="hub")
    fillers = [make_node("repo.a", f"F{i}", f"src/f{i}.py", node_id=f"f{i}") for i in range(60)]
    links = [make_link("hub", f["id"]) for f in fillers]
    gen = build_generation([hub] + fillers, links=links)

    result = project_map("repo.a", gen)
    hub_entry = next(h for h in result.top_hubs if h["key"] == key_for("repo.a", hub))
    assert hub_entry["is_hub"] is True
    assert hub_entry["degree"] > ranking.HUB_DEGREE_THRESHOLD
