from __future__ import annotations

import json
from pathlib import Path

from conftest import (
    build_generation,
    fake_embed_query_fn,
    make_link,
    make_node,
    registry_repo,
    write_registry,
)

from graphify_mesh.server.config import ServerConfig
from graphify_mesh.server.server import GraphifyMeshServer


def _server(tmp_path: Path, cwd: Path, embed_query_fn=None) -> GraphifyMeshServer:
    config = ServerConfig.from_env(
        mesh_root=tmp_path, registry_path=tmp_path / "bin" / "registry.json"
    )
    server = GraphifyMeshServer(
        config, cwd=cwd, embed_query_fn=embed_query_fn or fake_embed_query_fn()
    )
    return server


def _rpc(method: str, params: dict | None = None, request_id: int = 1) -> dict:
    message = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def test_initialize_and_tools_list():
    server = _server(Path("/tmp"), Path("/tmp"))
    init_response = server.handle_message(_rpc("initialize"))
    assert init_response["result"]["serverInfo"]["name"] == "graphify-mesh"

    list_response = server.handle_message(_rpc("tools/list"))
    tool_names = {t["name"] for t in list_response["result"]["tools"]}
    assert tool_names == {"search", "cross_project", "find_similar", "project_map", "context_pack"}


def test_notification_gets_no_response():
    server = _server(Path("/tmp"), Path("/tmp"))
    message = {"jsonrpc": "2.0", "method": "notifications/initialized"}  # no "id" => notification
    assert server.handle_message(message) is None


def test_unknown_method_returns_json_rpc_error():
    server = _server(Path("/tmp"), Path("/tmp"))
    response = server.handle_message(_rpc("bogus/method"))
    assert response["error"]["code"] == -32601


def test_search_degrades_gracefully_when_no_generation_published(tmp_path):
    write_registry(tmp_path / "bin" / "registry.json", [registry_repo("acme.repo", tmp_path)])
    server = _server(tmp_path, tmp_path)
    response = server.handle_message(
        _rpc("tools/call", {"name": "search", "arguments": {"q": "widget"}})
    )
    result = response["result"]
    assert result["isError"] is True
    assert "no consistent published generation" in result["content"][0]["text"]


def test_search_scope_fail_closed_at_tool_layer(tmp_path, monkeypatch):
    registry_path = tmp_path / "bin" / "registry.json"
    registered_root = tmp_path / "registered"
    registered_root.mkdir(parents=True)
    write_registry(registry_path, [registry_repo("acme.registered", registered_root)])

    unregistered_cwd = tmp_path / "unregistered"
    unregistered_cwd.mkdir(parents=True)
    server = _server(tmp_path, unregistered_cwd)
    # Bypass the store's real generation load requirement by monkeypatching
    # `_generation` to a trivial synthetic one, so this test isolates the
    # SCOPE fail-closed behavior from generation-load concerns.
    monkeypatch.setattr(
        server, "_generation", lambda: build_generation([make_node("acme.registered", "X", "x.py")])
    )

    response = server.handle_message(
        _rpc("tools/call", {"name": "search", "arguments": {"q": "x"}})
    )
    result = response["result"]
    assert result["isError"] is True
    assert "cannot resolve implicit scope" in result["content"][0]["text"]


def test_search_tool_end_to_end_with_synthetic_generation(tmp_path, monkeypatch):
    registry_path = tmp_path / "bin" / "registry.json"
    root = tmp_path / "acme"
    root.mkdir(parents=True)
    write_registry(registry_path, [registry_repo("acme.repo", root)])

    node = make_node("acme.repo", "OrderService", "src/order.py", node_id="n1", line=7)
    generation = build_generation([node])
    server = _server(tmp_path, root)
    monkeypatch.setattr(server, "_generation", lambda: generation)

    response = server.handle_message(
        _rpc("tools/call", {"name": "search", "arguments": {"q": "OrderService"}})
    )
    result = response["result"]
    assert result["isError"] is False
    payload = json.loads(result["content"][0]["text"])
    assert payload["scope_mode"] == "repo"
    assert any(h["citation"] == "[acme.repo:src/order.py:7]" for h in payload["hits"])


def test_cross_project_tool_end_to_end(tmp_path, monkeypatch):
    write_registry(tmp_path / "bin" / "registry.json", [registry_repo("acme.repo", tmp_path)])
    node = make_node("acme.repo", "SharedThing", "src/shared.py", node_id="n1", line=3)
    generation = build_generation([node])
    server = _server(tmp_path, tmp_path)
    monkeypatch.setattr(server, "_generation", lambda: generation)

    response = server.handle_message(
        _rpc("tools/call", {"name": "cross_project", "arguments": {"q": "SharedThing"}})
    )
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["hits"]


def test_find_similar_tool_end_to_end(tmp_path, monkeypatch):
    seed = make_node("acme.repo", "Gateway", "src/gw.py", node_id="seed")
    neighbor = make_node("acme.repo", "GatewayHelper", "src/gwh.py", node_id="nb")
    generation = build_generation([seed, neighbor], links=[make_link("seed", "nb")])
    server = _server(tmp_path, tmp_path)
    monkeypatch.setattr(server, "_generation", lambda: generation)
    response = server.handle_message(
        _rpc("tools/call", {"name": "find_similar", "arguments": {"node": "Gateway"}})
    )
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["resolved"] is True


def test_project_map_tool_end_to_end(tmp_path, monkeypatch):
    write_registry(tmp_path / "bin" / "registry.json", [registry_repo("acme.repo", tmp_path)])
    node = make_node("acme.repo", "Widget", "src/widget.py", node_id="n1")
    generation = build_generation([node])
    server = _server(tmp_path, tmp_path)
    monkeypatch.setattr(server, "_generation", lambda: generation)
    response = server.handle_message(
        _rpc("tools/call", {"name": "project_map", "arguments": {"repo": "acme.repo"}})
    )
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["resolved"] is True
    assert payload["node_count"] == 1


def test_context_pack_tool_end_to_end(tmp_path, monkeypatch):
    write_registry(tmp_path / "bin" / "registry.json", [registry_repo("acme.repo", tmp_path)])
    node = make_node("acme.repo", "OrderGoal", "src/order_goal.py", node_id="n1", line=5)
    generation = build_generation([node])
    server = _server(tmp_path, tmp_path)
    monkeypatch.setattr(server, "_generation", lambda: generation)

    response = server.handle_message(
        _rpc(
            "tools/call",
            {"name": "context_pack", "arguments": {"goal": "order goal", "token_budget": 5000}},
        )
    )
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["cards"]


def test_unknown_tool_name_returns_is_error():
    server = _server(Path("/tmp"), Path("/tmp"))
    response = server.handle_message(
        _rpc("tools/call", {"name": "not-a-real-tool", "arguments": {}})
    )
    assert response["result"]["isError"] is True
