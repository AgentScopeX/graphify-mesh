from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphify_mesh.server.config import ServerConfig
from graphify_mesh.server.store import GenerationStore, GenerationUnavailableError


def _write_generation(
    global_dir: Path, generation_id: str, nodes: list[dict], links: list[dict] | None = None
) -> None:
    """Writes a minimal, hash-consistent generation under
    `global_dir/generations/<id>/` and flips `current` to point at it —
    mirrors `graphify_mesh.sync.publish` closely enough for store.py's
    consistency gate without depending on the sync package's I/O helpers."""
    from graphify_mesh.sync.lexical_index import TOKENIZER_VERSION, build_lexical_index
    from graphify_mesh.sync.publish import output_hash

    graph = {"nodes": nodes, "links": links or []}
    graphs_by_repo: dict[str, dict] = {}
    for node in nodes:
        graphs_by_repo.setdefault(node["repo"], {"nodes": []})["nodes"].append(node)
    lexical = build_lexical_index(graphs_by_repo, {}).data

    manifest = {
        "generation_id": generation_id,
        "created_at": "2026-07-20T00:00:00Z",
        "repo_input_hashes": {},
        "registry_hash": "test",
        "config_hash": "test",
        "output_node_count": len(nodes),
        "output_edge_count": len(links or []),
        "output_hash": output_hash(graph),
        "clustering_backend": "louvain",
        "embedding_model": "test-model",
        "labeling": "skipped",
        "stale_repos": [],
        "lexical_index_tokenizer_version": TOKENIZER_VERSION,
    }

    gen_dir = global_dir / "generations" / generation_id
    gen_dir.mkdir(parents=True, exist_ok=True)
    (gen_dir / "global-graph.json").write_text(json.dumps(graph), encoding="utf-8")
    (gen_dir / "generation-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (gen_dir / "cross-project-overlay.json").write_text(json.dumps({"edges": []}), encoding="utf-8")
    (gen_dir / "lexical-index.json").write_text(json.dumps(lexical), encoding="utf-8")

    current = global_dir / "current"
    if current.exists() or current.is_symlink():
        current.unlink()
    current.symlink_to(gen_dir, target_is_directory=True)


def _config(tmp_path: Path) -> ServerConfig:
    return ServerConfig.from_env(
        mesh_root=tmp_path, registry_path=tmp_path / "bin" / "registry.json"
    )


def test_no_generation_published_raises_generation_unavailable(tmp_path):
    store = GenerationStore(_config(tmp_path))
    with pytest.raises(GenerationUnavailableError):
        _ = store.generation
    assert "no_generation_published" in store.degraded


def test_loads_valid_generation_and_builds_indexes(tmp_path):
    config = _config(tmp_path)
    _write_generation(
        config.global_dir,
        "gen-1",
        [{"id": "n1", "repo": "repo.a", "label": "Alpha", "source_file": "a.py"}],
    )
    store = GenerationStore(config)
    generation = store.generation
    assert generation.generation_id == "gen-1"
    assert "n1" in generation.node_by_id
    assert store.degraded == [
        "embeddings_unavailable"
    ]  # no embeddings dir published in this fixture


def test_inconsistent_manifest_rejected_all_or_nothing_keeps_previous(tmp_path):
    config = _config(tmp_path)
    _write_generation(
        config.global_dir,
        "gen-1",
        [{"id": "n1", "repo": "repo.a", "label": "Alpha", "source_file": "a.py"}],
    )
    store = GenerationStore(config)
    first = store.generation
    assert first.generation_id == "gen-1"

    # Publish a second generation whose manifest LIES about node count —
    # store.py must reject it and keep serving gen-1, not crash or half-load.
    gen2_dir = config.global_dir / "generations" / "gen-2"
    gen2_dir.mkdir(parents=True)
    graph = {
        "nodes": [{"id": "n1", "repo": "repo.a", "label": "Alpha", "source_file": "a.py"}],
        "links": [],
    }
    (gen2_dir / "global-graph.json").write_text(json.dumps(graph), encoding="utf-8")
    (gen2_dir / "generation-manifest.json").write_text(
        json.dumps(
            {
                "generation_id": "gen-2",
                "created_at": "2026-07-20T00:00:00Z",
                "repo_input_hashes": {},
                "registry_hash": "test",
                "config_hash": "test",
                "output_node_count": 999,  # deliberately wrong — graph.json has 1 node
                "output_edge_count": 0,
                "clustering_backend": "louvain",
                "embedding_model": "test-model",
                "labeling": "skipped",
                "stale_repos": [],
            }
        ),
        encoding="utf-8",
    )
    (gen2_dir / "cross-project-overlay.json").write_text(
        json.dumps({"edges": []}), encoding="utf-8"
    )
    (gen2_dir / "lexical-index.json").write_text(json.dumps({}), encoding="utf-8")
    current = config.global_dir / "current"
    current.unlink()
    current.symlink_to(gen2_dir, target_is_directory=True)

    second = store.generation  # triggers ensure_fresh() -> reload attempt -> rejected
    assert second.generation_id == "gen-1"  # still serving the last-good generation
    assert "reload_rejected_previous_generation_still_serving" in store.degraded
    assert any("output_node_count" in reason for reason in store.degraded)


def test_hot_reload_picks_up_new_valid_generation(tmp_path):
    config = _config(tmp_path)
    _write_generation(
        config.global_dir,
        "gen-1",
        [{"id": "n1", "repo": "repo.a", "label": "Alpha", "source_file": "a.py"}],
    )
    store = GenerationStore(config)
    assert store.generation.generation_id == "gen-1"

    _write_generation(
        config.global_dir,
        "gen-2",
        [{"id": "n2", "repo": "repo.a", "label": "Beta", "source_file": "b.py"}],
    )
    assert store.generation.generation_id == "gen-2"
    assert "n2" in store.generation.node_by_id


# --- _load_embeddings: corrupt/oversized shard tolerance ---------------------


def test_load_embeddings_skips_oversized_shard(tmp_path, monkeypatch):
    from graphify_mesh.server import store

    (tmp_path / "big.json").write_text(
        json.dumps({"repo_id": "big.repo", "entries": {"k": {"embedding": [1.0]}}}),
        encoding="utf-8",
    )
    (tmp_path / "ok.json").write_text(
        json.dumps({"repo_id": "ok.repo", "entries": {"k": {"embedding": [2.0]}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(store, "MAX_SHARD_BYTES", (tmp_path / "ok.json").stat().st_size)

    out = store._load_embeddings(tmp_path)
    assert "ok.repo" in out
    assert "big.repo" not in out


def test_load_embeddings_tolerates_malformed_shards(tmp_path):
    from graphify_mesh.server.store import _load_embeddings

    (tmp_path / "not-dict.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    (tmp_path / "entries-not-dict.json").write_text(
        json.dumps({"repo_id": "bad.repo", "entries": "oops"}), encoding="utf-8"
    )
    (tmp_path / "entry-not-dict.json").write_text(
        json.dumps({"repo_id": "half.repo", "entries": {"a": "oops", "b": {"embedding": [1.0]}}}),
        encoding="utf-8",
    )

    out = _load_embeddings(tmp_path)
    assert "bad.repo" not in out
    assert out["half.repo"] == {"b": [1.0]}
