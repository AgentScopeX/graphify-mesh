"""Crash-recovery behavior of the per-project state file: a torn/corrupt
state.json (power loss mid-write, manual tampering) must never brick
subsequent sync runs — load_state treats it exactly like a missing file."""

from __future__ import annotations

import json

from graphify_mesh.sync import state


def test_load_state_missing_file_returns_empty(tmp_path):
    assert state.load_state(tmp_path / "does-not-exist.json") == {}


def test_load_state_corrupt_json_returns_empty_instead_of_raising(tmp_path):
    state_path = tmp_path / "source-manifests.json"
    state_path.write_text('{"repo.a": {"code_hash": "abc', encoding="utf-8")
    assert state.load_state(state_path) == {}


def test_load_state_empty_file_returns_empty(tmp_path):
    state_path = tmp_path / "source-manifests.json"
    state_path.write_text("", encoding="utf-8")
    assert state.load_state(state_path) == {}


def test_load_state_non_object_json_returns_empty(tmp_path):
    state_path = tmp_path / "source-manifests.json"
    state_path.write_text('["not", "a", "dict"]', encoding="utf-8")
    assert state.load_state(state_path) == {}


def test_save_state_round_trips_and_replaces_corrupt_file(tmp_path):
    state_path = tmp_path / "source-manifests.json"
    state_path.write_text("garbage", encoding="utf-8")
    payload = {"repo.a": {"code_hash": "abc", "semantic_hash": "def", "file_count": 3}}
    state.save_state(state_path, payload)
    assert json.loads(state_path.read_text(encoding="utf-8")) == payload
    assert state.load_state(state_path) == payload
    # no stray tmp file left behind
    assert not state_path.with_suffix(state_path.suffix + ".tmp").exists()
