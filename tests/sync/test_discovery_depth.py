"""Multi-root + configurable-depth discovery (WS discovery generalization).

Covers: depth clamping behavior of `discover_filesystem` itself (the walker),
multi-root ordering, overlapping/nested-root dedup semantics, the
directory-symlink-escape regression, the path-traversal guards (candidate
dir escaping approved roots, target escaping approved roots), hidden-dir /
IGNORED_DIR_NAMES pruning, OSError-tolerant behavior (nonexistent root,
unreadable dir), and the reconcile canonical-pick tie-break.

Mirrors existing tests/sync/test_discovery.py conventions (the `env`
fixture, `Env.add_repo`) but lives in its own file since it exercises a
different axis (depth/roots) than the reconciliation-report tests already
in test_discovery.py.
"""

from __future__ import annotations

import errno
import logging
import os
from pathlib import Path

import pytest

from graphify_mesh.sync.discovery import (
    assert_registry_containment,
    discover_filesystem,
    reconcile,
)
from graphify_mesh.sync.registry import load_registry


def _project_dirs(scan_root, links):
    """Source-root dirs relative to scan_root, in yield order."""
    return [str(link.source_root.relative_to(scan_root)) for link in links]


# ---------------------------------------------------------------------------
# depth clamping
# ---------------------------------------------------------------------------


def test_depth3_and_depth4_projects_found_at_default_depth(env):
    env.add_repo("d.three", "d", "three", "a/b/three")
    env.add_repo("d.four", "d", "four", "a/b/c/four")
    env.write_registry()

    discovered = discover_filesystem([env.scan_root], [env.scan_root])
    names = {str(link.source_root) for link in discovered if link.target is not None}
    assert str(env.scan_root / "a" / "b" / "three") in names
    assert str(env.scan_root / "a" / "b" / "c" / "four") in names


def test_depth5_project_not_found_at_default_depth(env):
    env.add_repo("d.five", "d", "five", "a/b/c/d/five")
    env.write_registry()

    discovered = discover_filesystem([env.scan_root], [env.scan_root])
    names = {str(link.source_root) for link in discovered if link.target is not None}
    assert str(env.scan_root / "a" / "b" / "c" / "d" / "five") not in names


def test_depth3_project_not_found_at_depth2(env):
    env.add_repo("d.three", "d", "three", "a/b/three")
    env.write_registry()

    discovered = discover_filesystem([env.scan_root], [env.scan_root], depth=2)
    names = {str(link.source_root) for link in discovered if link.target is not None}
    assert str(env.scan_root / "a" / "b" / "three") not in names


# ---------------------------------------------------------------------------
# multi-root ordering
# ---------------------------------------------------------------------------


def test_multi_root_both_discovered_in_configured_order(env, tmp_path):
    other_root = tmp_path / "elsewhere"
    other_root.mkdir()

    root_a = env.add_repo("m.first", "m", "first", "first-project")
    collection_b = env.mesh_root / "graphify" / "m" / "second"
    collection_b.mkdir(parents=True)
    (collection_b / "graph.json").write_text("{}", encoding="utf-8")
    root_b = other_root / "second-project"
    root_b.mkdir()
    (root_b / "graphify-out").symlink_to(collection_b, target_is_directory=True)

    approved = [env.scan_root, other_root]
    discovered = discover_filesystem([env.scan_root, other_root], approved)
    linked = [link for link in discovered if link.target is not None]
    source_roots = [link.source_root for link in linked]

    assert root_a in source_roots
    assert root_b in source_roots
    # roots-order, preorder-within-root: env.scan_root's project must be
    # yielded before other_root's, since scan_root was passed first.
    assert source_roots.index(root_a) < source_roots.index(root_b)


def test_within_root_preorder_dfs_sorted_children(env):
    """One root, several projects at mixed depths, alphabetically-ordered
    children: DFS preorder means a candidate is yielded before its own
    descendants are walked, and a root's children are visited in sorted
    order before moving to the next sibling — so "a-project" (depth 1) is
    discovered before "b-parent/nested-project" (depth 2 under "b-parent",
    alphabetically between a- and c-), which in turn is discovered before
    "c-project" (depth 1, the next sorted sibling of "b-parent")."""
    env.add_repo("p.a", "p", "a", "a-project")
    env.add_repo("p.nested", "p", "nested", "b-parent/nested-project")
    env.add_repo("p.c", "p", "c", "c-project")
    env.write_registry()

    discovered = discover_filesystem([env.scan_root], [env.scan_root])
    linked = [link for link in discovered if link.target is not None]
    order = _project_dirs(env.scan_root, linked)

    assert order == [
        "a-project",
        str(Path("b-parent") / "nested-project"),
        "c-project",
    ]


def test_nested_roots_deep_project_found_both_orders(env):
    # outer = env.scan_root, nested = env.scan_root/workspace
    nested_root = env.scan_root / "workspace"
    nested_root.mkdir(parents=True, exist_ok=True)
    env.add_repo("n.deep", "n", "deep", "workspace/a/b/deep")
    env.write_registry()

    forward = discover_filesystem([env.scan_root, nested_root], [env.scan_root, nested_root])
    reverse = discover_filesystem([nested_root, env.scan_root], [env.scan_root, nested_root])

    target_dir = env.scan_root / "workspace" / "a" / "b" / "deep"
    for discovered in (forward, reverse):
        found = {link.source_root for link in discovered if link.target is not None}
        assert target_dir in found


def test_overlapping_roots_outer_first_revisits_nested_with_larger_budget(env):
    """[outer, nested] order, depth=2: the project must sit strictly BEYOND
    what outer's own walk can reach through "a" (the nested root), so that
    only the nested root's own fresh budget — not outer's, and not mere
    duplication of something outer already found — newly discovers it.

    nested_root = outer/"a". project = outer/"a"/"b"/"c" (3 levels below
    outer, 2 levels below nested_root).

    Outer alone at depth=2: outer -> yields "a" (remaining=2-1=1, >0 so
    recurse) -> within "a" yields "b" (remaining=1-1=0, so "b" is yielded
    but NOT recursed into) -> "c" (the actual project dir) is never reached.
    `visited["a"] = 1` is the only entry recorded; "b" and "c" never enter
    `visited` since remaining hit 0 before either assignment fires.

    [outer, nested_root] at depth=2: after outer's pass above, nested_root
    ("a") is processed as its own scan root. `prior_budget = visited["a"] =
    1`; the root request is the FULL depth (2) — 1 >= 2 is false, so this
    root is NOT skipped and gets a fresh walk from "a" with its own full
    budget: "a" -> yields "b" (remaining=2-1=1, >0, recurse) -> within "b"
    yields "c" (remaining=1-1=0, still yielded even though not recursed
    further) -> "c" IS the project dir, so it is discovered here — a
    project genuinely unreachable from outer's own budget, found only
    because the nested root got a fresh, larger budget (regression coverage
    for the fixed under-scan, not just duplicate-yield coverage)."""
    nested_root = env.scan_root / "a"
    root = env.add_repo("o.proj", "o", "proj", "a/b/c")

    outer_only = discover_filesystem([env.scan_root], [env.scan_root], depth=2)
    outer_only_found = {link.source_root for link in outer_only if link.target is not None}
    assert root not in outer_only_found

    combined = discover_filesystem(
        [env.scan_root, nested_root], [env.scan_root, nested_root], depth=2
    )
    combined_found = {link.source_root for link in combined if link.target is not None}
    assert root in combined_found

    # Fix round 4, Finding #1 regression: the project dir is reachable via
    # both the outer root's own descent (shallow, budget exhausted before
    # reaching it) and the nested root's fresh walk (which does reach it) —
    # dedup must collapse this to exactly one DiscoveredLink, not yield the
    # same (source_root, target) pair twice.
    combined_matches = [
        link for link in combined if link.target is not None and link.source_root == root
    ]
    assert len(combined_matches) == 1


def test_overlapping_roots_identical_path_repeat_deduped_not_aliases(env):
    """A TRUE identical-raw-path repeat (as opposed to the "genuinely only
    reachable via the nested root" case above): with enough budget, the
    SAME literal directory is yielded twice — once during outer's own
    descent, and again when the nested dir is scanned a second time as its
    own root with a fresh budget.

    nested_root = outer/"a". project = outer/"a"/"b" (2 levels below outer,
    1 level below nested_root). depth=3 is enough for outer's own descent
    to reach "b" directly (outer -> "a" [remaining=2] -> "b" [remaining=1]),
    so "b" is yielded once from outer's pass. Then nested_root ("a") is
    processed as its own scan root: prior_budget = visited["a"] = 2, and
    2 >= 3 is False, so it is NOT skipped and re-walks from "a" with a
    fresh budget of 3, yielding "b" a second time — same raw path
    `outer/a/b` both times, not merely the same resolved target.

    This must collapse to exactly one DiscoveredLink (Fix round 4, Finding
    #1's raw-path dedup key), unlike the distinct-alias-dirs case in
    test_discovery.py::test_duplicate_collection_two_discovered_symlinks_same_target
    (two different raw paths, same target) which must still yield two
    separate DiscoveredLinks / a `multiple_symlinks_same_target` report row —
    that companion test is unaffected by this dedup since its two source
    roots have different raw paths and therefore different dedup keys."""
    nested_root = env.scan_root / "a"
    root = env.add_repo("o.proj", "o", "proj", "a/b")

    combined = discover_filesystem(
        [env.scan_root, nested_root], [env.scan_root, nested_root], depth=3
    )
    combined_matches = [
        link for link in combined if link.target is not None and link.source_root == root
    ]
    assert len(combined_matches) == 1


def test_overlapping_roots_nested_first_skips_re_walk_at_equal_or_greater_budget(env):
    """[nested, outer] order (reversed from the test above): "workspace" is
    processed FIRST as its own scan root, recording the FULL depth (4) in
    `visited`. When outer's own descent later reaches "workspace" as a
    CHILD, its recursion budget there is depth-1 (3) — the prior recorded
    budget (4) is >= that (3), so the subtree is NOT re-walked (though
    "workspace" itself is still yielded as a candidate, per the
    never-suppress-yielding rule). Net effect: the deep project under
    "workspace" is discovered only ONCE — the opposite of the outer-first
    ordering above, and the case the previous version of this test suite's
    comment mischaracterized."""
    nested_root = env.scan_root / "workspace"
    nested_root.mkdir(parents=True, exist_ok=True)
    root = env.add_repo("o.proj2", "o", "proj2", "workspace/proj")

    discovered = discover_filesystem([nested_root, env.scan_root], [env.scan_root, nested_root])
    matches = [link for link in discovered if link.source_root == root and link.target is not None]
    assert len(matches) == 1


# ---------------------------------------------------------------------------
# symlink-escape / traversal guards
# ---------------------------------------------------------------------------


def test_symlinked_project_dir_not_discovered(env, tmp_path):
    real_root = tmp_path / "real-project"
    real_root.mkdir()
    collection = env.mesh_root / "graphify" / "s" / "linked"
    collection.mkdir(parents=True)
    (collection / "graph.json").write_text("{}", encoding="utf-8")
    (real_root / "graphify-out").symlink_to(collection, target_is_directory=True)

    # The project dir itself is a symlink sitting directly under scan_root.
    (env.scan_root / "linked-project").symlink_to(real_root, target_is_directory=True)

    discovered = discover_filesystem([env.scan_root], [env.scan_root, tmp_path])
    found_roots = {link.source_root for link in discovered}
    assert (env.scan_root / "linked-project") not in found_roots


def test_candidate_dir_resolving_outside_approved_roots_is_rejected(env, tmp_path):
    # Real (non-symlinked) project dir under scan_root, but approved_roots is
    # scoped to a different tree entirely -> the resolved project dir itself
    # escapes every approved root (Fix 1's symmetric guard, independent of
    # the graphify-out target check).
    outside = tmp_path / "outside"
    outside.mkdir()
    real_project = env.add_repo("t.real", "t", "real", "real-project")

    discovered = discover_filesystem([env.scan_root], [outside])
    rejected = [link for link in discovered if link.rejected_traversal]
    assert any(link.source_root == real_project for link in rejected)


def test_target_outside_approved_roots_rejected(env, tmp_path):
    outside = tmp_path / "outside-target"
    outside.mkdir()
    collection = outside / "collection"
    collection.mkdir()
    (collection / "graph.json").write_text("{}", encoding="utf-8")

    root = env.scan_root / "proj-with-outside-target"
    root.mkdir()
    (root / "graphify-out").symlink_to(collection, target_is_directory=True)

    discovered = discover_filesystem([env.scan_root], [env.scan_root])
    rejected = [link for link in discovered if link.rejected_traversal]
    assert any(link.source_root == root for link in rejected)


def test_target_under_second_approved_root_accepted(env, tmp_path):
    root_a = tmp_path / "root-a"
    root_a.mkdir()
    root_b = tmp_path / "root-b"
    root_b.mkdir()
    collection = root_b / "collection"
    collection.mkdir()
    (collection / "graph.json").write_text("{}", encoding="utf-8")

    link_dir = root_a / "proj"
    link_dir.mkdir()
    (link_dir / "graphify-out").symlink_to(collection, target_is_directory=True)

    discovered = discover_filesystem([root_a], [root_a, root_b])
    accepted = [
        link for link in discovered if link.target is not None and not link.rejected_traversal
    ]
    assert any(link.source_root == link_dir for link in accepted)


# ---------------------------------------------------------------------------
# hidden dirs / IGNORED_DIR_NAMES pruning
# ---------------------------------------------------------------------------


def test_hidden_and_ignored_dirs_not_descended(env):
    collection = env.mesh_root / "graphify" / "h" / "hidden"
    collection.mkdir(parents=True)
    (collection / "graph.json").write_text("{}", encoding="utf-8")

    hidden_root = env.scan_root / ".hidden" / "proj"
    hidden_root.mkdir(parents=True)
    (hidden_root / "graphify-out").symlink_to(collection, target_is_directory=True)

    ignored_root = env.scan_root / "node_modules" / "proj"
    ignored_root.mkdir(parents=True)
    (ignored_root / "graphify-out").symlink_to(collection, target_is_directory=True)

    discovered = discover_filesystem([env.scan_root], [env.scan_root])
    found_roots = {link.source_root for link in discovered}
    assert hidden_root not in found_roots
    assert ignored_root not in found_roots


# ---------------------------------------------------------------------------
# OSError tolerance
# ---------------------------------------------------------------------------


def test_nonexistent_root_skipped_others_scanned(env, tmp_path):
    # Not a warn+skip candidate case (that's covered by the deterministic
    # monkeypatch tests below) — a nonexistent scan root hits the plain
    # FileNotFoundError/"not a directory" path in discover_filesystem's own
    # root loop (root.is_dir() is False), also logged at warning level.
    missing_root = tmp_path / "does-not-exist"
    root = env.add_repo("e.exists", "e", "exists", "exists-project")

    discovered = discover_filesystem([missing_root, env.scan_root], [missing_root, env.scan_root])
    found_roots = {link.source_root for link in discovered if link.target is not None}
    assert root in found_roots


def test_unreadable_dir_skipped(env, caplog):
    if os.geteuid() == 0:
        pytest.skip("chmod 000 has no effect for root")

    caplog.set_level(logging.WARNING)
    unreadable = env.scan_root / "locked"
    unreadable.mkdir()
    reachable_root = env.add_repo("u.ok", "u", "ok", "ok-project")
    try:
        unreadable.chmod(0o000)
        discovered = discover_filesystem([env.scan_root], [env.scan_root])
    finally:
        unreadable.chmod(0o755)

    found_roots = {link.source_root for link in discovered if link.target is not None}
    assert reachable_root in found_roots
    # The actual skip-with-warning behavior: os.scandir(unreadable) raises
    # PermissionError (an OSError), caught in _iter_candidate_dirs, which
    # logs "cannot list ..." and returns without yielding anything from
    # that subtree — not a silent skip.
    assert any(
        "cannot list" in record.message and str(unreadable) in record.message
        for record in caplog.records
    )


def test_lstat_oserror_on_one_candidate_skips_it_others_discovered(env, monkeypatch, caplog):
    """Deterministic version of the warn+skip path guarding os.lstat(link_path)
    (Fix round 3, Finding #4): force os.lstat to raise a genuine OSError
    (ELOOP) for exactly one candidate's graphify-out entry, leaving every
    other path's real os.lstat call untouched. That candidate must be
    skipped (not silently, not crashing the run) with a warning logged
    naming it, while its sibling is still discovered normally."""
    import graphify_mesh.sync.discovery as discovery_module

    caplog.set_level(logging.WARNING)
    root_bad = env.add_repo("l.bad", "l", "bad", "lstat-bad-project")
    root_ok = env.add_repo("l.ok", "l", "ok", "lstat-ok-project")
    bad_link = root_bad / "graphify-out"

    real_lstat = os.lstat

    def fake_lstat(path, *args, **kwargs):
        if Path(path) == bad_link:
            raise OSError(errno.ELOOP, "too many levels of symbolic links")
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(discovery_module.os, "lstat", fake_lstat)

    discovered = discover_filesystem([env.scan_root], [env.scan_root])
    linked = {link.source_root for link in discovered if link.target is not None}

    assert root_ok in linked
    assert root_bad not in linked
    assert any("cannot lstat" in record.message for record in caplog.records)


def test_stat_oserror_on_symlink_target_skips_candidate(env, monkeypatch, caplog):
    """Deterministic version of the warn+skip path guarding os.stat(link_path)
    (the follow-the-symlink dangling-target probe): force os.stat to raise a
    genuine OSError (ESTALE) for exactly one candidate's graphify-out target,
    leaving its sibling untouched."""
    import graphify_mesh.sync.discovery as discovery_module

    caplog.set_level(logging.WARNING)
    root_bad = env.add_repo("s.bad", "s", "bad", "stat-bad-project")
    root_ok = env.add_repo("s.ok", "s", "ok", "stat-ok-project")
    bad_link = root_bad / "graphify-out"

    real_stat = os.stat

    def fake_stat(path, *args, **kwargs):
        if Path(path) == bad_link:
            raise OSError(errno.ESTALE, "stale file handle")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(discovery_module.os, "stat", fake_stat)

    discovered = discover_filesystem([env.scan_root], [env.scan_root])
    linked = {link.source_root for link in discovered if link.target is not None}

    assert root_ok in linked
    assert root_bad not in linked
    assert any("cannot stat graphify-out target" in record.message for record in caplog.records)


def test_resolve_oserror_on_symlink_target_skips_candidate(env, monkeypatch, caplog):
    """Deterministic version of the warn+skip path guarding
    link_path.resolve() (resolving the graphify-out symlink's real target):
    force Path.resolve to raise a genuine OSError (ELOOP) for exactly one
    candidate's graphify-out link, leaving every other Path.resolve() call
    (including the walker's own root/candidate-dir resolves) untouched."""
    caplog.set_level(logging.WARNING)
    root_bad = env.add_repo("r.bad", "r", "bad", "resolve-bad-project")
    root_ok = env.add_repo("r.ok", "r", "ok", "resolve-ok-project")
    bad_link = root_bad / "graphify-out"

    real_resolve = Path.resolve

    def fake_resolve(self, *args, **kwargs):
        if self == bad_link:
            raise OSError(errno.ELOOP, "too many levels of symbolic links")
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fake_resolve)

    discovered = discover_filesystem([env.scan_root], [env.scan_root])
    linked = {link.source_root for link in discovered if link.target is not None}

    assert root_ok in linked
    assert root_bad not in linked
    assert any("cannot resolve graphify-out target" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# reconcile canonical pick
# ---------------------------------------------------------------------------


def test_reconcile_prefers_registry_known_root_over_lexicographic_fallback(env):
    collection = env.collection_path("r", "canon")
    collection.mkdir(parents=True)
    (collection / "graph.json").write_text("{}", encoding="utf-8")

    # The registry-known root sorts AFTER the alias lexicographically, so a
    # plain sorted-fallback tie-break would pick the alias instead.
    known_root = env.scan_root / "z-known-root"
    alias_root = env.scan_root / "a-alias-root"
    for root in (known_root, alias_root):
        root.mkdir()
        (root / "graphify-out").symlink_to(collection, target_is_directory=True)

    env._repos.append(
        {
            "repo_id": "r.canon",
            "root": str(known_root),
            "collection_path": str(collection),
            "enabled": True,
        }
    )
    env.write_registry()

    discovered = discover_filesystem([env.scan_root], [env.scan_root])
    registry = load_registry(env.registry_path)
    report = reconcile(discovered, registry, env.mesh_root)

    assert "r.canon" in report.registered
    assert not report.renamed


# ---------------------------------------------------------------------------
# assert_registry_containment
# ---------------------------------------------------------------------------


def test_assert_registry_containment_entry_under_second_approved_root_passes(env, tmp_path):
    second_root = tmp_path / "second-approved"
    second_root.mkdir()
    collection = second_root / "collection"
    collection.mkdir()

    env._repos.append(
        {
            "repo_id": "c.ok",
            "root": str(second_root / "proj"),
            "collection_path": str(collection),
            "enabled": True,
        }
    )
    env.write_registry()
    registry = load_registry(env.registry_path)

    # Must not raise: collection resolves under the second approved root.
    assert_registry_containment(registry, [env.scan_root, second_root])


def test_assert_registry_containment_outside_all_without_external_roots_raises(env, tmp_path):
    outside = tmp_path / "totally-outside"
    outside.mkdir()
    collection = outside / "collection"
    collection.mkdir()

    env._repos.append(
        {
            "repo_id": "c.bad",
            "root": str(outside / "proj"),
            "collection_path": str(collection),
            "enabled": True,
        }
    )
    env.write_registry()
    registry = load_registry(env.registry_path)

    with pytest.raises(ValueError, match="c.bad"):
        assert_registry_containment(registry, [env.scan_root])


def test_assert_registry_containment_entry_under_external_root_passes(env, tmp_path):
    # `external` is NOT in approved_roots at all — only registry.external_roots
    # covers it, exercising the `allowed_roots = resolved_approved +
    # external_roots` union rather than the approved_roots branch alone.
    external = tmp_path / "external-approved"
    external.mkdir()
    collection = external / "collection"
    collection.mkdir()

    env._repos.append(
        {
            "repo_id": "c.ext",
            "root": str(external / "proj"),
            "collection_path": str(collection),
            "enabled": True,
        }
    )
    env.write_registry(external_roots=[str(external)])
    registry = load_registry(env.registry_path)

    # Must not raise: collection resolves under external_roots even though
    # approved_roots (env.scan_root only) doesn't cover it.
    assert_registry_containment(registry, [env.scan_root])


def test_assert_registry_containment_outside_approved_and_external_raises(env, tmp_path):
    outside = tmp_path / "outside-everything"
    outside.mkdir()
    collection = outside / "collection"
    collection.mkdir()
    # A real external_roots entry is declared, but it points elsewhere — the
    # union of approved_roots + external_roots still doesn't cover `outside`.
    other_external = tmp_path / "unrelated-external"
    other_external.mkdir()

    env._repos.append(
        {
            "repo_id": "c.bad2",
            "root": str(outside / "proj"),
            "collection_path": str(collection),
            "enabled": True,
        }
    )
    env.write_registry(external_roots=[str(other_external)])
    registry = load_registry(env.registry_path)

    with pytest.raises(ValueError, match="c.bad2"):
        assert_registry_containment(registry, [env.scan_root])
