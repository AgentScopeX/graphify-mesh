from __future__ import annotations

from pathlib import Path

from graphify_mesh.sync import publish


def _make_generation(generations_dir: Path, name: str, incomplete: bool = False) -> Path:
    gen_dir = generations_dir / name
    gen_dir.mkdir(parents=True)
    (gen_dir / "global-graph.json").write_text("{}", encoding="utf-8")
    if incomplete:
        (gen_dir / "lexical-index.json.tmp").write_text("{", encoding="utf-8")
    else:
        (gen_dir / "lexical-index.json").write_text("{}", encoding="utf-8")
    return gen_dir


def test_prune_keeps_current_and_n_most_recent(tmp_path):
    generations_dir = tmp_path / "generations"
    names = ["20260101T000000Z-a", "20260102T000000Z-b", "20260103T000000Z-c", "20260104T000000Z-d"]
    for name in names:
        _make_generation(generations_dir, name)

    current = tmp_path / "current"
    current.symlink_to(generations_dir / names[-1], target_is_directory=True)

    removed = publish.prune_old_generations(generations_dir, current, keep=2)

    remaining = {p.name for p in generations_dir.iterdir()}
    assert remaining == {names[-2], names[-1]}
    assert set(removed) == {names[0], names[1]}


def test_prune_never_removes_current_even_if_older_than_keep_window(tmp_path):
    generations_dir = tmp_path / "generations"
    names = ["20260101T000000Z-a", "20260102T000000Z-b", "20260103T000000Z-c"]
    for name in names:
        _make_generation(generations_dir, name)

    # current points at the OLDEST generation (e.g. a later run failed
    # validation and never flipped forward) — it must survive pruning
    # regardless of the keep-count sort order.
    current = tmp_path / "current"
    current.symlink_to(generations_dir / names[0], target_is_directory=True)

    publish.prune_old_generations(generations_dir, current, keep=1)

    remaining = {p.name for p in generations_dir.iterdir()}
    assert names[0] in remaining


def test_prune_removes_incomplete_generation_regardless_of_keep_count(tmp_path):
    """A generation dir with a dangling `.tmp` file (process killed between
    write_lexical_index's tmp-write and its rename) never finished
    publishing and was never `current` — safe to remove outright, even if
    it would otherwise fall inside the keep-N window by recency."""
    generations_dir = tmp_path / "generations"
    _make_generation(generations_dir, "20260101T000000Z-a")
    _make_generation(generations_dir, "20260102T000000Z-b", incomplete=True)

    current = tmp_path / "current"
    current.symlink_to(generations_dir / "20260101T000000Z-a", target_is_directory=True)

    removed = publish.prune_old_generations(generations_dir, current, keep=5)

    remaining = {p.name for p in generations_dir.iterdir()}
    assert "20260102T000000Z-b" not in remaining
    assert "20260101T000000Z-a" in remaining
    assert "20260102T000000Z-b" in removed


def test_prune_no_generations_dir_is_a_no_op(tmp_path):
    assert publish.prune_old_generations(tmp_path / "does-not-exist", tmp_path / "current", keep=2) == []
