from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BIN_DIR = Path(__file__).resolve().parents[2] / "src"
if str(_BIN_DIR) not in sys.path:
    sys.path.insert(0, str(_BIN_DIR))

from graphify_mesh.sync.embedding import node_key  # noqa: E402
from graphify_mesh.sync.lexical_index import build_lexical_index  # noqa: E402

from graphify_mesh.server.store import Generation  # noqa: E402


def make_node(repo, label, source_file, node_id=None, line=1, community_name=None, **extra) -> dict:
    node = {
        "id": node_id or f"{repo}:{label}",
        "repo": repo,
        "label": label,
        "source_file": source_file,
        "line": line,
        "community_name": community_name,
    }
    node.update(extra)
    return node


def make_link(src_id: str, dst_id: str, confidence: str = "EXTRACTED") -> dict:
    return {"source": src_id, "target": dst_id, "confidence": confidence}


def key_for(repo: str, node: dict) -> str:
    """Same durable logical key `graphify_mesh.sync.embedding.node_key` produces
    — used by tests to build embeddings dicts / assert on `Hit.key`."""
    return node_key(repo, node)


def build_generation(
    nodes: list[dict],
    links: list[dict] | None = None,
    overlay_edges: list[dict] | None = None,
    embeddings: dict[str, dict[str, list[float]]] | None = None,
    generation_id: str = "gen-test-1",
    manifest_extra: dict | None = None,
) -> Generation:
    """Builds a fully-indexed, in-memory `Generation` from synthetic nodes —
    no disk I/O, no real project data. The lexical index is built with the
    REAL `graphify_mesh.sync.lexical_index.build_lexical_index` (not a hand-rolled
    stub) so postings/alias_exact/doc_freq shapes are exactly what
    production code produces."""
    graph = {"nodes": nodes, "links": links or []}
    graphs_by_repo: dict[str, dict] = {}
    for node in nodes:
        graphs_by_repo.setdefault(node["repo"], {"nodes": []})["nodes"].append(node)
    lexical_result = build_lexical_index(graphs_by_repo, {})
    manifest = {"generation_id": generation_id, **(manifest_extra or {})}
    generation = Generation(
        generation_id=generation_id,
        manifest=manifest,
        graph=graph,
        overlay={"edges": overlay_edges or []},
        lexical=lexical_result.data,
        embeddings=embeddings or {},
    )
    generation.build_indexes()
    return generation


def write_registry(path: Path, repos: list[dict], disabled: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"repos": repos, "disabled": disabled or [], "external_roots": []}), encoding="utf-8"
    )


def registry_repo(repo_id: str, root: Path, enabled: bool = True) -> dict:
    return {
        "repo_id": repo_id,
        "root": str(root),
        "collection_path": str(root / "graphify-out"),
        "enabled": enabled,
    }


def fake_embed_query_fn(vectors_by_query: dict[str, list[float]] | None = None):
    """Deterministic stand-in for `embed_query.make_embed_query_fn`'s
    returned callable: tests never touch the network. `None` in
    `vectors_by_query` (or query absent) simulates the degraded/unavailable
    path exactly like a real transport failure would."""
    table = vectors_by_query or {}

    def embed_query(query: str):
        return table.get(query)

    return embed_query


@pytest.fixture()
def gen_factory():
    return build_generation
