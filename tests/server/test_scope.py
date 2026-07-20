from __future__ import annotations

from pathlib import Path

import pytest

from graphify_mesh.server.scope import (
    ScopeResolutionError,
    load_registry_entries,
    resolve_repo_list,
    resolve_scope,
)
from conftest import registry_repo, write_registry


def test_fail_closed_no_cwd_match_raises_not_silent_global(tmp_path):
    registry_path = tmp_path / "registry.json"
    registered_root = tmp_path / "www" / "known-repo"
    registered_root.mkdir(parents=True)
    write_registry(registry_path, [registry_repo("known.repo", registered_root)])
    entries = load_registry_entries(registry_path)

    unregistered_cwd = tmp_path / "www" / "unregistered-repo"
    unregistered_cwd.mkdir(parents=True)

    with pytest.raises(ScopeResolutionError):
        resolve_scope(None, unregistered_cwd, entries)
    with pytest.raises(ScopeResolutionError):
        resolve_scope("current", unregistered_cwd, entries)


def test_scope_all_never_raises_and_returns_no_filter(tmp_path):
    entries = load_registry_entries(tmp_path / "does-not-exist.json")
    decision = resolve_scope("all", tmp_path, entries)
    assert decision.mode == "all"
    assert decision.repo_ids is None


def test_scope_current_resolves_to_longest_matching_registered_root(tmp_path):
    registry_path = tmp_path / "registry.json"
    root = tmp_path / "www" / "acme-project"
    nested_cwd = root / "src" / "deep"
    nested_cwd.mkdir(parents=True)
    write_registry(registry_path, [registry_repo("acme.project", root)])
    entries = load_registry_entries(registry_path)

    decision = resolve_scope(None, nested_cwd, entries)
    assert decision.mode == "repo"
    assert decision.repo_ids == frozenset({"acme.project"})


def test_scope_repo_explicit_rejects_unknown_repo_id(tmp_path):
    registry_path = tmp_path / "registry.json"
    root = tmp_path / "www" / "acme-project"
    root.mkdir(parents=True)
    write_registry(registry_path, [registry_repo("acme.project", root)])
    entries = load_registry_entries(registry_path)

    with pytest.raises(ScopeResolutionError):
        resolve_scope("repo:does.not.exist", tmp_path, entries)

    decision = resolve_scope("repo:acme.project", tmp_path, entries)
    assert decision.repo_ids == frozenset({"acme.project"})


def test_scope_ignores_disabled_repo_for_cwd_match(tmp_path):
    registry_path = tmp_path / "registry.json"
    root = tmp_path / "www" / "disabled-project"
    root.mkdir(parents=True)
    write_registry(registry_path, [registry_repo("disabled.project", root, enabled=False)])
    entries = load_registry_entries(registry_path)

    with pytest.raises(ScopeResolutionError):
        resolve_scope(None, root, entries)


def test_invalid_scope_string_raises():
    with pytest.raises(ScopeResolutionError):
        resolve_scope("bogus-scope-value", Path("/tmp"), [])


def test_resolve_repo_list_none_means_all_registered_enabled():
    entries = load_registry_entries(Path("/nonexistent"))  # empty, harmless
    assert resolve_repo_list(None, entries) is None
    assert resolve_repo_list([], entries) is None


def test_resolve_repo_list_rejects_unknown_repo_id(tmp_path):
    registry_path = tmp_path / "registry.json"
    root = tmp_path / "www" / "acme-project"
    root.mkdir(parents=True)
    write_registry(registry_path, [registry_repo("acme.project", root)])
    entries = load_registry_entries(registry_path)

    with pytest.raises(ScopeResolutionError):
        resolve_repo_list(["acme.project", "unknown.repo"], entries)

    assert resolve_repo_list(["acme.project"], entries) == frozenset({"acme.project"})
