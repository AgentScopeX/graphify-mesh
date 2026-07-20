"""Proves scope filtering happens BEFORE candidate-depth truncation/ranking,
not as a post-filter of an already-ranked global top-K (see `scope.py`'s
module docstring for the named failure mode this guards against: a large
repo's global top-K can crowd out every current-project hit before a
post-hoc filter ever runs).
"""

from __future__ import annotations

from conftest import build_generation, fake_embed_query_fn, key_for, make_node

from graphify_mesh.server import ranking
from graphify_mesh.server.retrieval import lexical_candidates, rank

CURRENT_REPO = "zzz.current"  # deliberately sorts AFTER the global repo id below
GLOBAL_REPO = "aaa.global"  # so a tied lexical score's deterministic (key-ascending)
# tie-break puts every global-repo hit ahead of the
# current-repo one — the exact ordering needed to
# demonstrate the crowding-out failure mode.


def _build_crowding_fixture():
    current_node = make_node(CURRENT_REPO, "WidgetFactory", "src/widget_factory.py", node_id="cur")
    # Exactly CANDIDATE_DEPTH_LEXICAL global-repo nodes, all matching the
    # same query term with an identical score (same field boost, same
    # global idf) — enough to fully occupy the lexical candidate depth on
    # their own, crowding the current-project node out of an UNSCOPED top-K
    # entirely once the tie-break is factored in.
    global_nodes = [
        make_node(GLOBAL_REPO, f"WidgetGlobal{i}", f"src/widget_global_{i}.py", node_id=f"g{i}")
        for i in range(ranking.CANDIDATE_DEPTH_LEXICAL)
    ]
    gen = build_generation([current_node] + global_nodes)
    return gen, current_node, global_nodes


def test_unscoped_candidate_depth_would_crowd_out_current_project_hit():
    """Negative control: WITHOUT a repo filter, the global repo's nodes
    alone saturate CANDIDATE_DEPTH_LEXICAL and the current-project node's
    weaker/later-sorted match never makes it into the candidate pool."""
    gen, current_node, global_nodes = _build_crowding_fixture()
    ranked_keys = lexical_candidates("widget", gen.lexical, repo_filter=None)
    assert len(ranked_keys) == ranking.CANDIDATE_DEPTH_LEXICAL
    assert key_for(CURRENT_REPO, current_node) not in ranked_keys


def test_scope_filter_applied_before_ranking_preserves_current_project_hit():
    """Positive: filtering to `current.repo` BEFORE depth-truncation means
    the current-project node is never competing against the 50 global
    candidates for a candidate-pool slot — it is simply the only doc in the
    filtered pool for this repo."""
    gen, current_node, global_nodes = _build_crowding_fixture()
    repo_filter = frozenset({CURRENT_REPO})

    ranked_keys = lexical_candidates("widget", gen.lexical, repo_filter=repo_filter)
    assert ranked_keys == [key_for(CURRENT_REPO, current_node)]

    result = rank("widget", gen, repo_filter, k=5, embed_query_fn=fake_embed_query_fn())
    hit_keys = {h.key for h in result.hits}
    assert key_for(CURRENT_REPO, current_node) in hit_keys
    # And scoping must ALSO exclude every global hit, not just prioritize
    # the current-project one.
    assert all(h.repo == CURRENT_REPO for h in result.hits)
