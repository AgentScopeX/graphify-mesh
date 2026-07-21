from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphify_mesh.sync import embedding
from graphify_mesh.sync.config import Settings

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FAKE_GRAPHIFY = FIXTURES_DIR / "fake_graphify" / "graphify"


def _settings(tmp_path: Path, **overrides) -> Settings:
    mesh_root = tmp_path / "mesh"
    return Settings.from_env(
        mesh_root=mesh_root,
        scan_root=tmp_path / "www",
        registry_path=mesh_root / "bin" / "registry.json",
        graphify_bin=str(FAKE_GRAPHIFY),
        **overrides,
    )


# ---------------------------------------------------------------------------
# snippet builder
# ---------------------------------------------------------------------------


def test_build_snippet_bounded_around_line(tmp_path):
    src = tmp_path / "big.py"
    src.write_text("\n".join(f"line {i}" for i in range(200)), encoding="utf-8")

    snippet = embedding.build_snippet(tmp_path, "big.py", line=100)

    lines = snippet.splitlines()
    assert len(lines) <= embedding.SNIPPET_WINDOW_LINES
    assert "line 100" in snippet or "line 99" in snippet  # window centers near the requested line
    assert "line 0" not in snippet
    assert "line 199" not in snippet


def test_build_snippet_max_chars_enforced(tmp_path):
    src = tmp_path / "wide.py"
    src.write_text("\n".join("x" * 200 for _ in range(20)), encoding="utf-8")

    snippet = embedding.build_snippet(tmp_path, "wide.py", line=1)
    assert len(snippet) <= embedding.SNIPPET_MAX_CHARS


def test_build_snippet_missing_file_returns_empty(tmp_path):
    assert embedding.build_snippet(tmp_path, "does-not-exist.py", line=1) == ""


def test_build_snippet_no_root_returns_empty():
    assert embedding.build_snippet(None, "any.py", line=1) == ""


def test_build_embedding_input_truncated_to_max_chars():
    huge_snippet = "y" * 10000
    text = embedding.build_embedding_input("MyLabel", huge_snippet, "src/x.py", "Some Community")
    assert len(text) <= embedding.EMBED_INPUT_MAX_CHARS
    assert text.startswith("label: MyLabel")


def test_build_embedding_input_no_snippet_still_includes_required_parts():
    text = embedding.build_embedding_input("MyLabel", "", "src/x.py", "Some Community")
    assert "label: MyLabel" in text
    assert "path: src/x.py" in text
    assert "community: Some Community" in text
    assert "snippet:" not in text


# ---------------------------------------------------------------------------
# skip heuristic
# ---------------------------------------------------------------------------


def test_trivial_getter_with_one_line_body_is_skipped():
    assert embedding.is_trivial_node("getName", "return self.name") is True


def test_getter_with_meaningful_body_is_not_skipped():
    snippet = "\n".join(f"    step_{i}()" for i in range(5))
    assert embedding.is_trivial_node("getName", snippet) is False


def test_non_accessor_label_is_never_trivial():
    assert embedding.is_trivial_node("computeInvoiceTotal", "return 1") is False


# ---------------------------------------------------------------------------
# batched /api/embed call — native contract, mocked HTTP layer only
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, body: dict):
        self._body = json.dumps(body).encode("utf-8")
        self.status = 200

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_embed_batch_request_shape_matches_verified_native_contract(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(
            {
                "model": "qwen3-embedding:0.6b",
                "embeddings": [[0.1, 0.2], [0.3, 0.4]],
                "total_duration": 1,
                "load_duration": 1,
                "prompt_eval_count": 2,
            }
        )

    monkeypatch.setattr(embedding.urllib.request, "urlopen", fake_urlopen)

    vectors = embedding.embed_batch(
        "https://ollama.example.com:11434", "qwen3-embedding:0.6b", ["a", "b"]
    )

    # C9: native endpoint, NOT /v1/embeddings.
    assert captured["url"] == "https://ollama.example.com:11434/api/embed"
    assert captured["method"] == "POST"
    assert captured["body"] == {"model": "qwen3-embedding:0.6b", "input": ["a", "b"]}
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_batch_raises_on_shape_mismatch(monkeypatch):
    def fake_urlopen(req, timeout=None):
        return _FakeResponse(
            {"model": "x", "embeddings": [[0.1, 0.2]]}
        )  # only 1, but 2 inputs requested

    monkeypatch.setattr(embedding.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError):
        embedding.embed_batch("https://host", "model", ["a", "b"])


def test_embed_batch_empty_input_short_circuits_without_a_call(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise AssertionError("must not be called for empty input")

    monkeypatch.setattr(embedding.urllib.request, "urlopen", fake_urlopen)
    assert embedding.embed_batch("https://host", "model", []) == []


# ---------------------------------------------------------------------------
# id-map + tombstones (C27) across two synthetic generations
# ---------------------------------------------------------------------------


def test_id_map_tombstones_removed_node_instead_of_dropping():
    gen1_keys = {"repo.a\x1fsrc/a.py\x1fAlpha", "repo.a\x1fsrc/b.py\x1fBeta"}
    id_map_gen1 = embedding.build_id_map({}, gen1_keys, "gen-1")
    assert all(v["status"] == "active" for v in id_map_gen1.values())

    # Gen 2: "Beta" node removed.
    gen2_keys = {"repo.a\x1fsrc/a.py\x1fAlpha"}
    id_map_gen2 = embedding.build_id_map(id_map_gen1, gen2_keys, "gen-2")

    beta_key = "repo.a\x1fsrc/b.py\x1fBeta"
    assert beta_key in id_map_gen2  # never silently dropped
    assert id_map_gen2[beta_key]["status"] == "tombstoned"
    assert id_map_gen2[beta_key]["tombstoned_at"] == "gen-2"

    alpha_key = "repo.a\x1fsrc/a.py\x1fAlpha"
    assert id_map_gen2[alpha_key]["status"] == "active"


def test_id_map_reappearing_key_goes_back_to_active():
    keys_gen1 = {"repo.a\x1fsrc/a.py\x1fAlpha"}
    id_map_gen1 = embedding.build_id_map({}, keys_gen1, "gen-1")
    id_map_gen2 = embedding.build_id_map(id_map_gen1, set(), "gen-2")
    assert id_map_gen2["repo.a\x1fsrc/a.py\x1fAlpha"]["status"] == "tombstoned"

    id_map_gen3 = embedding.build_id_map(id_map_gen2, keys_gen1, "gen-3")
    assert id_map_gen3["repo.a\x1fsrc/a.py\x1fAlpha"]["status"] == "active"
    assert id_map_gen3["repo.a\x1fsrc/a.py\x1fAlpha"]["tombstoned_at"] is None


# ---------------------------------------------------------------------------
# resumable / changed-only recompute
# ---------------------------------------------------------------------------


def _graph_with_one_node(label="Widget", source_file="src/w.py", extra: str = "") -> dict:
    return {
        "nodes": [
            {
                "id": "n1",
                "label": label,
                "source_file": source_file,
                "community_name": "Comm" + extra,
            },
        ]
    }


def test_compute_repo_shard_reuses_unchanged_node_without_calling_embed(tmp_path, monkeypatch):
    graph = _graph_with_one_node()
    stats = embedding.EmbeddingStats()

    call_count = {"n": 0}

    def fake_embed_batch(base_url, model, inputs, timeout=30.0):
        call_count["n"] += 1
        return [[1.0, 2.0] for _ in inputs]

    monkeypatch.setattr(embedding, "embed_batch", fake_embed_batch)

    first_shard = embedding.compute_repo_shard(
        "repo.a",
        graph,
        tmp_path,
        previous_shard={},
        base_url="https://host",
        model="m",
        stats=stats,
    )
    assert call_count["n"] == 1
    key = next(iter(first_shard))
    assert first_shard[key]["embedding"] == [1.0, 2.0]

    # Second run, node completely unchanged -> content_hash matches, must be
    # reused without another embed_batch call.
    second_shard = embedding.compute_repo_shard(
        "repo.a",
        graph,
        tmp_path,
        previous_shard=first_shard,
        base_url="https://host",
        model="m",
        stats=stats,
    )
    assert call_count["n"] == 1  # unchanged: no new call
    assert second_shard[key]["embedding"] == [1.0, 2.0]
    assert stats.reused == 1


def test_compute_repo_shard_recomputes_when_content_changes(tmp_path, monkeypatch):
    stats = embedding.EmbeddingStats()
    call_count = {"n": 0}

    def fake_embed_batch(base_url, model, inputs, timeout=30.0):
        call_count["n"] += 1
        return [[float(call_count["n"])] for _ in inputs]

    monkeypatch.setattr(embedding, "embed_batch", fake_embed_batch)

    first_graph = _graph_with_one_node(extra="A")
    first_shard = embedding.compute_repo_shard(
        "repo.a",
        first_graph,
        tmp_path,
        previous_shard={},
        base_url="https://host",
        model="m",
        stats=stats,
    )
    assert call_count["n"] == 1

    # community_name changed -> embedding input hash changes -> must recompute.
    second_graph = _graph_with_one_node(extra="B")
    second_shard = embedding.compute_repo_shard(
        "repo.a",
        second_graph,
        tmp_path,
        previous_shard=first_shard,
        base_url="https://host",
        model="m",
        stats=stats,
    )
    assert call_count["n"] == 2
    key = next(iter(second_shard))
    assert second_shard[key]["embedding"] == [2.0]


def test_compute_repo_shard_skips_trivial_node_without_calling_embed(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    (src / "w.py").write_text("def getName(self):\n    return self.name\n", encoding="utf-8")
    graph = {
        "nodes": [
            {
                "id": "n1",
                "label": "getName",
                "source_file": "src/w.py",
                "line": 1,
                "community_name": "Comm",
            },
        ]
    }
    stats = embedding.EmbeddingStats()

    def fake_embed_batch(base_url, model, inputs, timeout=30.0):
        raise AssertionError("must not be called for a trivial node")

    monkeypatch.setattr(embedding, "embed_batch", fake_embed_batch)

    shard = embedding.compute_repo_shard(
        "repo.a",
        graph,
        tmp_path,
        previous_shard={},
        base_url="https://host",
        model="m",
        stats=stats,
    )
    key = next(iter(shard))
    assert shard[key]["embedding"] is None
    assert stats.skipped_trivial == 1


# ---------------------------------------------------------------------------
# GC keeps exactly N generations
# ---------------------------------------------------------------------------


def test_gc_old_generations_keeps_only_last_two(tmp_path):
    generations_dir = tmp_path / "generations"
    for name in [
        "20260101T000000Z-a",
        "20260102T000000Z-b",
        "20260103T000000Z-c",
        "20260104T000000Z-d",
    ]:
        (generations_dir / name).mkdir(parents=True)

    removed = embedding.gc_old_generations(generations_dir, keep=2)

    remaining = sorted(p.name for p in generations_dir.iterdir())
    assert remaining == ["20260103T000000Z-c", "20260104T000000Z-d"]
    assert sorted(removed) == ["20260101T000000Z-a", "20260102T000000Z-b"]


def test_gc_old_generations_noop_when_within_limit(tmp_path):
    generations_dir = tmp_path / "generations"
    (generations_dir / "gen-1").mkdir(parents=True)
    removed = embedding.gc_old_generations(generations_dir, keep=2)
    assert removed == []
    assert (generations_dir / "gen-1").exists()


def test_gc_old_generations_pins_current_target_despite_sort_order(tmp_path):
    # Clock skew: the LIVE generation sorts oldest by name. GC must pin the
    # dir `current` points at, mirroring publish.prune_old_generations.
    generations_dir = tmp_path / "generations"
    for name in [
        "20260101T000000Z-live",
        "20260102T000000Z-b",
        "20260103T000000Z-c",
        "20260104T000000Z-d",
    ]:
        (generations_dir / name).mkdir(parents=True)
    current = tmp_path / "current"
    current.symlink_to(generations_dir / "20260101T000000Z-live", target_is_directory=True)

    removed = embedding.gc_old_generations(generations_dir, keep=2, current=current)

    assert (generations_dir / "20260101T000000Z-live").exists()
    assert "20260101T000000Z-live" not in removed
    assert "20260102T000000Z-b" in removed


# ---------------------------------------------------------------------------
# run_embedding_stage: unchanged-repo short-circuit + degraded fallback
# ---------------------------------------------------------------------------


def test_run_embedding_stage_reuses_whole_shard_for_unchanged_repo(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    # Seed a "previous published" shard by writing directly under embeddings_current_symlink target.
    prev_gen_dir = settings.embeddings_dir / "generations" / "gen-0"
    prev_gen_dir.mkdir(parents=True)
    (prev_gen_dir / "repo.a.json").write_text(
        json.dumps(
            {
                "repo_id": "repo.a",
                "entries": {
                    "repo.a\x1fsrc/w.py\x1fWidget": {"content_hash": "abc", "embedding": [9.0]}
                },
            }
        ),
        encoding="utf-8",
    )
    (prev_gen_dir / "id-map.json").write_text(json.dumps({}), encoding="utf-8")
    settings.embeddings_dir.mkdir(parents=True, exist_ok=True)
    (settings.embeddings_dir / "current").symlink_to(prev_gen_dir, target_is_directory=True)

    def fail_embed_batch(base_url, model, inputs, timeout=30.0):
        raise AssertionError("unchanged repo must not trigger any embed call")

    monkeypatch.setattr(embedding, "embed_batch", fail_embed_batch)

    graphs_by_repo = {"repo.a": _graph_with_one_node()}
    result = embedding.run_embedding_stage(
        graph_paths_by_repo={},
        repo_roots_by_id={"repo.a": tmp_path},
        graphs_by_repo=graphs_by_repo,
        unchanged_repo_ids={"repo.a"},
        settings=settings,
        provisional_generation_id="gen-1",
        health_check=lambda base_url, timeout: True,
    )

    assert result.status == embedding.EMBED_HEALTHY
    assert result.stats.reused_repos_unchanged == 1
    assert result.vectors_by_repo["repo.a"]["repo.a\x1fsrc/w.py\x1fWidget"] == [9.0]


def test_run_embedding_stage_degraded_carries_forward_previous_vectors(tmp_path):
    settings = _settings(tmp_path)
    prev_gen_dir = settings.embeddings_dir / "generations" / "gen-0"
    prev_gen_dir.mkdir(parents=True)
    (prev_gen_dir / "repo.a.json").write_text(
        json.dumps(
            {
                "repo_id": "repo.a",
                "entries": {
                    "repo.a\x1fsrc/w.py\x1fWidget": {"content_hash": "abc", "embedding": [7.0]}
                },
            }
        ),
        encoding="utf-8",
    )
    (prev_gen_dir / "id-map.json").write_text(json.dumps({}), encoding="utf-8")
    (settings.embeddings_dir / "current").symlink_to(prev_gen_dir, target_is_directory=True)

    graphs_by_repo = {"repo.a": _graph_with_one_node()}
    result = embedding.run_embedding_stage(
        graph_paths_by_repo={},
        repo_roots_by_id={"repo.a": tmp_path},
        graphs_by_repo=graphs_by_repo,
        unchanged_repo_ids=set(),
        settings=settings,
        provisional_generation_id="gen-1",
        health_check=lambda base_url, timeout: False,
    )

    assert result.status == embedding.EMBED_DEGRADED
    assert result.vectors_by_repo["repo.a"]["repo.a\x1fsrc/w.py\x1fWidget"] == [7.0]


def test_run_embedding_stage_mid_run_failure_is_partial_not_crash(tmp_path, monkeypatch):
    """Regression test: a real network failure DURING embedding (health
    check passed, then embed_batch itself raises — e.g. a DNS blip) must
    degrade the run, not propagate and crash the whole pipeline. repo.a is
    processed successfully before the failure; repo.b's embed_batch call
    fails; repo.c (not yet reached) must fall back to its previous shard
    rather than being silently skipped or crashing the process."""
    settings = _settings(tmp_path)
    prev_gen_dir = settings.embeddings_dir / "generations" / "gen-0"
    prev_gen_dir.mkdir(parents=True)
    (prev_gen_dir / "repo.c.json").write_text(
        json.dumps(
            {
                "repo_id": "repo.c",
                "entries": {
                    "repo.c\x1fsrc/z.py\x1fZeta": {"content_hash": "prevc", "embedding": [3.0]}
                },
            }
        ),
        encoding="utf-8",
    )
    (prev_gen_dir / "id-map.json").write_text(json.dumps({}), encoding="utf-8")
    (settings.embeddings_dir / "current").symlink_to(prev_gen_dir, target_is_directory=True)

    def flaky_embed_batch(base_url, model, inputs, timeout=30.0):
        if "network-blip-marker" in inputs[0]:
            raise RuntimeError(
                "embed_batch request failed: <urlopen error [Errno -2] Name or service not known>"
            )
        return [[1.0] for _ in inputs]

    monkeypatch.setattr(embedding, "embed_batch", flaky_embed_batch)

    def _graph_with_marker(marker: str) -> dict:
        return {
            "nodes": [
                {
                    "id": "n1",
                    "label": marker,
                    "source_file": "src/w.py",
                    "loc": "L1",
                    "community_name": "Widgets",
                }
            ]
        }

    graphs_by_repo = {
        "repo.a": _graph_with_marker("Alpha"),
        "repo.b": _graph_with_marker("network-blip-marker-Bravo"),
        "repo.c": _graph_with_marker("Gamma"),
    }
    result = embedding.run_embedding_stage(
        graph_paths_by_repo={},
        repo_roots_by_id={"repo.a": tmp_path, "repo.b": tmp_path, "repo.c": tmp_path},
        graphs_by_repo=graphs_by_repo,
        unchanged_repo_ids=set(),
        settings=settings,
        provisional_generation_id="gen-1",
        health_check=lambda base_url, timeout: True,
    )

    assert result.status == embedding.EMBED_PARTIAL
    assert "repo.b" in result.reason
    # repo.a completed for real before the failure.
    assert result.vectors_by_repo["repo.a"]
    # repo.c (not yet reached when the failure hit) fell back to its
    # previous published vector rather than being lost or crashing.
    assert result.vectors_by_repo["repo.c"]["repo.c\x1fsrc/z.py\x1fZeta"] == [3.0]


# ---------------------------------------------------------------------------
# persist_generation: staging -> permanent storage -> GC, only on publish
# ---------------------------------------------------------------------------


def test_persist_generation_writes_current_and_gcs_old(tmp_path):
    embeddings_dir = tmp_path / "embeddings"
    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    (staged_dir / "repo.a.json").write_text(
        json.dumps({"repo_id": "repo.a", "entries": {}}), encoding="utf-8"
    )
    (staged_dir / "id-map.json").write_text("{}", encoding="utf-8")

    embedding.persist_generation(embeddings_dir, "20260101T000000Z-aaa", staged_dir, keep=2)
    embedding.persist_generation(embeddings_dir, "20260102T000000Z-bbb", staged_dir, keep=2)
    embedding.persist_generation(embeddings_dir, "20260103T000000Z-ccc", staged_dir, keep=2)

    generations = sorted(p.name for p in (embeddings_dir / "generations").iterdir())
    assert generations == ["20260102T000000Z-bbb", "20260103T000000Z-ccc"]

    current = embeddings_dir / "current"
    assert current.is_symlink()
    assert current.resolve().name == "20260103T000000Z-ccc"
    assert (current / "repo.a.json").is_file()


# ---------------------------------------------------------------------------
# hostile-input guards: shard filenames and snippet paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_id", ["../evil", "a/b", "a\\b", ".hidden", "..", ".", ""])
def test_shard_filename_rejects_unsafe_repo_id(bad_id):
    with pytest.raises(ValueError, match="unsafe repo_id"):
        embedding._shard_filename(bad_id)


def test_read_previous_shard_refuses_traversal_repo_id(tmp_path):
    current_dir = tmp_path / "current"
    current_dir.mkdir()
    (tmp_path / "outside.json").write_text(
        json.dumps({"entries": {"leaked": {}}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unsafe repo_id"):
        embedding.read_previous_shard(current_dir, "../outside")


def test_stage_embeddings_refuses_traversal_repo_id(tmp_path):
    with pytest.raises(ValueError, match="unsafe repo_id"):
        embedding.stage_embeddings(tmp_path, {"../escape": {}}, id_map={})


def test_build_snippet_rejects_absolute_source_file(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("token=very-secret", encoding="utf-8")
    root = tmp_path / "repo"
    root.mkdir()
    assert embedding.build_snippet(root, str(secret), line=1) == ""


def test_build_snippet_rejects_dotdot_traversal(tmp_path):
    (tmp_path / "secret.txt").write_text("token=very-secret", encoding="utf-8")
    root = tmp_path / "repo"
    root.mkdir()
    assert embedding.build_snippet(root, "../secret.txt", line=1) == ""
