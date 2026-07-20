from __future__ import annotations

from graphify_mesh.server import ranking
from graphify_mesh.server.context_pack import build_context_pack
from conftest import build_generation, fake_embed_query_fn, make_node


def test_context_pack_basic_cards_have_citations_and_extracted_confidence():
    nodes = [
        make_node("repo.a", "OrderService", "src/order_service.py", node_id="n1", line=42),
        make_node("repo.a", "OrderRepository", "src/order_repo.py", node_id="n2", line=10),
    ]
    gen = build_generation(nodes)
    result = build_context_pack("order", gen, None, [], token_budget=10_000, embed_query_fn=fake_embed_query_fn())

    assert result.cards
    for card in result.cards:
        assert card.citation.startswith("[") and card.citation.endswith("]")
        assert ":" in card.citation
        assert card.confidence == ranking.CONFIDENCE_EXTRACTED


def test_context_pack_truncation_never_splits_a_card():
    nodes = [
        make_node("repo.a", f"OrderThing{i}", f"src/order_thing_{i}.py", node_id=f"n{i}", line=i + 1)
        for i in range(10)
    ]
    gen = build_generation(nodes)

    unbounded = build_context_pack(
        "order", gen, None, [], token_budget=1_000_000, embed_query_fn=fake_embed_query_fn()
    )
    assert len(unbounded.cards) > 1, "fixture must produce more than one candidate card to test truncation"

    # Budget for exactly the first card's cost (plus a hair): only ONE full
    # card should be included, never a partial one.
    first_cost = unbounded.cards[0].estimated_tokens()
    tight = build_context_pack(
        "order", gen, None, [], token_budget=first_cost, embed_query_fn=fake_embed_query_fn()
    )
    assert len(tight.cards) == 1
    assert tight.truncated is True
    # The one included card must be a COMPLETE card object, not a sliced one.
    assert tight.cards[0].citation == unbounded.cards[0].citation
    assert tight.cards[0].snippet == unbounded.cards[0].snippet


def test_context_pack_zero_budget_yields_no_cards_and_truncated_flag():
    nodes = [make_node("repo.a", "Solo", "src/solo.py", node_id="n1", line=1)]
    gen = build_generation(nodes)
    result = build_context_pack("solo", gen, None, [], token_budget=0, embed_query_fn=fake_embed_query_fn())
    assert result.cards == []
    assert result.truncated is True


def test_context_pack_propagates_degraded_flag_from_rank():
    nodes = [make_node("repo.a", "Alpha", "src/alpha.py", node_id="n1", line=1)]
    gen = build_generation(nodes, embeddings={})  # no embeddings => vector retriever degrades
    result = build_context_pack("alpha", gen, None, [], token_budget=10_000, embed_query_fn=fake_embed_query_fn())
    assert "embeddings_unavailable" in result.degraded


def test_context_pack_respects_repo_filter_scope():
    current = make_node("repo.current", "SharedThing", "src/current.py", node_id="cur", line=1)
    other = make_node("repo.other", "SharedThing", "src/other.py", node_id="oth", line=1)
    gen = build_generation([current, other])
    result = build_context_pack(
        "sharedthing", gen, frozenset({"repo.current"}), [], token_budget=10_000, embed_query_fn=fake_embed_query_fn()
    )
    assert all(c.repo == "repo.current" for c in result.cards)
