"""Registry validation contract: repo_id is used as a filename downstream
(embedding shards), so hostile/malformed registry content must fail loudly
at load time, never surface as path traversal or silent last-wins merges."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphify_mesh.sync.registry import load_registry


def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _entry(repo_id: str) -> dict:
    return {"repo_id": repo_id, "root": "/tmp/x", "collection_path": "/tmp/x/graphify-out"}


@pytest.mark.parametrize(
    "bad_id",
    ["../evil", "a/b", "a\\b", ".hidden", "..", ".", "/abs", "a b", "a\nb", ""],
)
def test_load_registry_rejects_unsafe_repo_id(tmp_path, bad_id):
    path = _write(tmp_path / "registry.json", {"repos": [_entry(bad_id)]})
    with pytest.raises(ValueError, match="repo_id|repos\\[0\\]"):
        load_registry(path)


@pytest.mark.parametrize("good_id", ["acme.repo", "repo-1", "Repo_2", "a"])
def test_load_registry_accepts_normal_repo_ids(tmp_path, good_id):
    path = _write(tmp_path / "registry.json", {"repos": [_entry(good_id)]})
    registry = load_registry(path)
    assert registry.repos[0].repo_id == good_id


def test_load_registry_rejects_duplicate_repo_ids(tmp_path):
    path = _write(tmp_path / "registry.json", {"repos": [_entry("acme.repo"), _entry("acme.repo")]})
    with pytest.raises(ValueError, match="duplicate repo_id"):
        load_registry(path)


def test_load_registry_rejects_empty_string_in_external_roots(tmp_path):
    # "" would Path("").resolve() to the CWD downstream, silently widening
    # the approved path set.
    path = _write(tmp_path / "registry.json", {"repos": [], "external_roots": [""]})
    with pytest.raises(ValueError, match="external_roots\\[0\\]"):
        load_registry(path)


def test_load_registry_rejects_empty_string_in_disabled(tmp_path):
    path = _write(tmp_path / "registry.json", {"repos": [], "disabled": [""]})
    with pytest.raises(ValueError, match="disabled\\[0\\]"):
        load_registry(path)
