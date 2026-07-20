"""`graphify-mesh` stdio MCP server (WS5 deliverable 2): wires the 5 hybrid/
cross-project/evidence tools (`search`, `cross_project`, `find_similar`,
`project_map`, `context_pack`) onto the newline-delimited JSON-RPC 2.0
transport in `protocol.py`.

One process per client session (plan: "stdio per-session"). Scope
resolution (`scope.py`) is anchored to THIS PROCESS's cwd, resolved fresh on
every `search`/`context_pack` call against `registry.json` — matches a
dedicated per-project session, not a shared daemon serving many cwds at
once (see C26 in `graphify_mesh.server/__init__.py` for why this is not an
arbitrary-path cache).

Every tool call is wrapped so `ScopeResolutionError` and
`GenerationUnavailableError` degrade to an MCP tool-error result
(`isError: true` in the tool result, not a transport-level crash or a
JSON-RPC protocol error) — a client always gets a structured response it
can read, never a dead pipe.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

from graphify_mesh.server import context_pack as context_pack_mod
from graphify_mesh.server import project_map as project_map_mod
from graphify_mesh.server import protocol, ranking, similar as similar_mod
from graphify_mesh.server.config import ServerConfig
from graphify_mesh.server.embed_query import make_embed_query_fn
from graphify_mesh.server.retrieval import Hit, rank
from graphify_mesh.server.scope import (
    ScopeResolutionError,
    load_registry_entries,
    resolve_repo_list,
    resolve_scope,
)
from graphify_mesh.server.store import Generation, GenerationStore, GenerationUnavailableError

log = logging.getLogger("graphify_mesh.server.server")

SERVER_NAME = "graphify-mesh"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2024-11-05"

DEFAULT_TOKEN_BUDGET = 2000

# Upper bound for client-supplied `token_budget`: 50x the default. Large
# enough for any legitimate evidence pack, small enough that a hostile value
# can't drive unbounded work.
MAX_TOKEN_BUDGET = 50 * DEFAULT_TOKEN_BUDGET

# Single source of truth for the `k` ceiling: the same constant
# `retrieval.rank` clamps against (retrieval.py) — validation here and the
# clamp there can never drift apart.
MAX_K = ranking.MAX_K


class ToolError(RuntimeError):
    """Any tool-execution failure that should degrade to an MCP
    `isError: true` result rather than a JSON-RPC protocol error or a
    crashed process."""


def _validate_k(arguments: dict) -> int:
    """Client-supplied `k` must be a real int (bools excluded) in
    [1, MAX_K]. Anything else is a ToolError, never a crash or a silent
    coercion."""
    k = arguments.get("k", ranking.DEFAULT_K)
    if isinstance(k, bool) or not isinstance(k, int):
        raise ToolError(f"'k' must be an integer between 1 and {MAX_K}")
    if k < 1 or k > MAX_K:
        raise ToolError(f"'k' must be between 1 and {MAX_K}, got {k}")
    return k


def _validate_token_budget(arguments: dict) -> int:
    """Client-supplied `token_budget` must be a real int (bools excluded)
    in [1, MAX_TOKEN_BUDGET]."""
    budget = arguments.get("token_budget", DEFAULT_TOKEN_BUDGET)
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise ToolError(f"'token_budget' must be an integer between 1 and {MAX_TOKEN_BUDGET}")
    if budget < 1 or budget > MAX_TOKEN_BUDGET:
        raise ToolError(f"'token_budget' must be between 1 and {MAX_TOKEN_BUDGET}, got {budget}")
    return budget


def _citation(repo: str, source_file: str, node: dict) -> str:
    line = node.get("line")
    return f"[{repo}:{source_file}:{line if line is not None else '?'}]"


def _hit_to_dict(hit: Hit, generation: Generation) -> dict:
    node = generation.node_by_id.get(hit.node_id, {})
    return {
        "key": hit.key,
        "repo": hit.repo,
        "label": hit.label,
        "source_file": hit.source_file,
        "citation": _citation(hit.repo, hit.source_file, node),
        "community_name": hit.community_name,
        "degree": hit.degree,
        "score": hit.score,
        "match_type": hit.match_type,
        "deprecated": hit.deprecated,
    }


class GraphifyMeshServer:
    def __init__(self, config: ServerConfig, cwd: Path | None = None, embed_query_fn: Callable | None = None):
        self.config = config
        self.store = GenerationStore(config)
        self._cwd_override = cwd
        self.embed_query_fn = embed_query_fn or make_embed_query_fn()

    @property
    def cwd(self) -> Path:
        return self._cwd_override if self._cwd_override is not None else Path.cwd()

    def _registry_entries(self):
        return load_registry_entries(self.config.registry_path)

    # --- tool implementations ------------------------------------------

    def tool_search(self, arguments: dict) -> dict:
        query = arguments.get("q", "")
        k = _validate_k(arguments)
        entries = self._registry_entries()
        try:
            decision = resolve_scope(arguments.get("scope"), self.cwd, entries)
        except ScopeResolutionError as exc:
            raise ToolError(str(exc)) from exc
        generation = self._generation()
        ranked = rank(query, generation, decision.repo_ids, k, self.embed_query_fn)
        return {
            "hits": [_hit_to_dict(h, generation) for h in ranked.hits],
            "degraded": ranked.degraded,
            "scope_mode": decision.mode,
        }

    def tool_cross_project(self, arguments: dict) -> dict:
        query = arguments.get("q", "")
        k = _validate_k(arguments)
        entries = self._registry_entries()
        try:
            repo_filter = resolve_repo_list(arguments.get("repos"), entries)
        except ScopeResolutionError as exc:
            raise ToolError(str(exc)) from exc
        generation = self._generation()
        ranked = rank(query, generation, repo_filter, k, self.embed_query_fn)
        return {"hits": [_hit_to_dict(h, generation) for h in ranked.hits], "degraded": ranked.degraded}

    def tool_find_similar(self, arguments: dict) -> dict:
        node = arguments.get("node", "")
        k = _validate_k(arguments)
        cross_repo_only = bool(arguments.get("cross_repo_only", False))
        generation = self._generation()
        result = similar_mod.find_similar(node, generation, k, cross_repo_only)
        return {
            "resolved": result.resolved,
            "hits": [_hit_to_dict(h, generation) for h in result.hits],
            "degraded": result.degraded,
        }

    def tool_project_map(self, arguments: dict) -> dict:
        repo = arguments.get("repo")
        if not isinstance(repo, str) or not repo:
            raise ToolError("'repo' must be a non-empty string (a registered repo_id)")
        # Contract: project_map serves REGISTERED repo_ids only. A repo that
        # is present in the current generation but has since been removed
        # from the registry fails closed like an unknown one.
        registered = {entry.repo_id for entry in self._registry_entries()}
        if repo not in registered:
            return {
                "resolved": False,
                "repo": repo,
                "node_count": 0,
                "community_breakdown": {},
                "top_hubs": [],
                "degraded": ["repo_not_registered"],
            }
        generation = self._generation()
        result = project_map_mod.project_map(repo, generation)
        return {
            "resolved": result.resolved,
            "repo": result.repo,
            "node_count": result.node_count,
            "community_breakdown": result.community_breakdown,
            "top_hubs": result.top_hubs,
            "degraded": result.degraded,
        }

    def tool_context_pack(self, arguments: dict) -> dict:
        goal = arguments.get("goal", "")
        token_budget = _validate_token_budget(arguments)
        entries = self._registry_entries()
        try:
            decision = resolve_scope(arguments.get("scope"), self.cwd, entries)
        except ScopeResolutionError as exc:
            raise ToolError(str(exc)) from exc
        generation = self._generation()
        result = context_pack_mod.build_context_pack(
            goal, generation, decision.repo_ids, entries, token_budget, self.embed_query_fn
        )
        return {
            "goal": result.goal,
            "cards": [
                {
                    "citation": c.citation,
                    "repo": c.repo,
                    "label": c.label,
                    "community_name": c.community_name,
                    "confidence": c.confidence,
                    "snippet": c.snippet,
                    "score": c.score,
                }
                for c in result.cards
            ],
            "truncated": result.truncated,
            "degraded": result.degraded,
        }

    def _generation(self) -> Generation:
        try:
            return self.store.generation
        except GenerationUnavailableError as exc:
            raise ToolError(str(exc)) from exc

    # --- MCP wiring -------------------------------------------------------

    TOOLS: dict[str, str] = {
        "search": "tool_search",
        "cross_project": "tool_cross_project",
        "find_similar": "tool_find_similar",
        "project_map": "tool_project_map",
        "context_pack": "tool_context_pack",
    }

    def tool_schemas(self) -> list[dict]:
        return [
            {
                "name": "search",
                "description": "Hybrid lexical+vector+structural search within a scope (current project by default). Fails closed if scope='current' can't be resolved against registry.json.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string"},
                        "scope": {"type": "string", "description": "'current' (default), 'all', or 'repo:<id>'"},
                        "k": {"type": "integer", "default": ranking.DEFAULT_K},
                    },
                    "required": ["q"],
                },
            },
            {
                "name": "cross_project",
                "description": "Explicit cross-repo hybrid search, optionally restricted to a repo_id list.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string"},
                        "repos": {"type": "array", "items": {"type": "string"}},
                        "k": {"type": "integer", "default": ranking.DEFAULT_K},
                    },
                    "required": ["q"],
                },
            },
            {
                "name": "find_similar",
                "description": "Cross-project (and optionally same-project) structurally/semantically similar nodes to a given node/label.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "node": {"type": "string"},
                        "k": {"type": "integer", "default": ranking.DEFAULT_K},
                        "cross_repo_only": {"type": "boolean", "default": False},
                    },
                    "required": ["node"],
                },
            },
            {
                "name": "project_map",
                "description": "Structural overview of one registered repo in the current generation: node count, community breakdown, top hub nodes.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"repo": {"type": "string"}},
                    "required": ["repo"],
                },
            },
            {
                "name": "context_pack",
                "description": "Evidence cards (citations, snippets, confidence) for a goal, truncated to a token budget without ever splitting a card.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string"},
                        "scope": {"type": "string"},
                        "token_budget": {"type": "integer", "default": DEFAULT_TOKEN_BUDGET},
                    },
                    "required": ["goal"],
                },
            },
        ]

    def call_tool(self, name: str, arguments: dict) -> dict:
        method_name = self.TOOLS.get(name)
        if method_name is None:
            return {"content": [{"type": "text", "text": f"unknown tool: {name!r}"}], "isError": True}
        method = getattr(self, method_name)
        try:
            result = method(arguments or {})
            return {"content": [{"type": "text", "text": json.dumps(result)}], "isError": False}
        except ToolError as exc:
            return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
        except Exception:
            # Includes non-JSON-serializable results from json.dumps above.
            # Traceback to stderr only; the client sees a generic message —
            # never exception text, paths, or stack frames.
            log.exception("tool %r raised an unexpected exception", name)
            return {"content": [{"type": "text", "text": "internal error"}], "isError": True}

    # --- JSON-RPC method dispatch ------------------------------------------

    def handle_message(self, message: dict) -> dict | None:
        method = message.get("method")
        request_id = message.get("id")
        is_notification = "id" not in message

        if method == "initialize":
            return protocol.result_response(
                request_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                    "capabilities": {"tools": {}},
                },
            )
        if method == "notifications/initialized":
            return None  # notification: no response due
        if method == "tools/list":
            return protocol.result_response(request_id, {"tools": self.tool_schemas()})
        if method == "tools/call":
            params = message.get("params", {}) or {}
            result = self.call_tool(params.get("name", ""), params.get("arguments", {}))
            return protocol.result_response(request_id, result)
        if method == "ping":
            return protocol.result_response(request_id, {})
        if is_notification:
            return None  # unknown notification: silently ignored, never errors
        return protocol.error_response(request_id, -32601, f"method not found: {method!r}")


def build_server() -> GraphifyMeshServer:
    config = ServerConfig.from_env()
    return GraphifyMeshServer(config)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="graphify-mesh-server",
        description=(
            "Stdio MCP server for the merged global graph. Speaks newline-delimited "
            "JSON-RPC 2.0 on stdin/stdout; one process per client session. Takes no "
            "options — configure via GRAPHIFY_MESH_ROOT / GRAPHIFY_MESH_REGISTRY."
        ),
    )
    parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    server = build_server()
    protocol.serve(server.handle_message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
