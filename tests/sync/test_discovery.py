from __future__ import annotations

from graphify_mesh.sync.discovery import discover_filesystem, reconcile
from graphify_mesh.sync.registry import load_registry


def test_broken_symlink_reported_not_crashed(env):
    root = env.add_repo(
        "example-org.styleguide", "example-org", "styleguide", "styleguide.example-org.dev.lo"
    )
    env.write_registry()

    # Break the symlink: point it at a target that no longer exists.
    link = root / "graphify-out"
    ghost = env.mesh_root / "graphify" / "example-org" / "ghost-target"
    link.unlink()
    link.symlink_to(ghost, target_is_directory=True)

    discovered = discover_filesystem(env.scan_root, env.scan_root)
    registry = load_registry(env.registry_path)
    report = reconcile(discovered, registry, env.mesh_root)

    assert "example-org.styleguide" in report.broken
    assert "example-org.styleguide" not in report.removed
    assert "example-org.styleguide" not in report.registered


def test_duplicate_collection_two_registry_entries_same_path(env):
    env.add_repo(
        "example-org.styleguide", "example-org", "styleguide", "styleguide.example-org.dev.lo"
    )
    # Second registry entry pointing at the SAME collection_path via a
    # different discovered root (no separate collection dir created).
    env.add_repo(
        "example-org.styleguide-dupe",
        "example-org",
        "styleguide",
        "styleguide-dupe.example-org.dev.lo",
        make_collection=False,
    )
    env.write_registry()

    discovered = discover_filesystem(env.scan_root, env.scan_root)
    registry = load_registry(env.registry_path)
    report = reconcile(discovered, registry, env.mesh_root)

    reasons = {d["reason"] for d in report.duplicates}
    assert "registry_duplicate_collection_path" in reasons
    assert "multiple_symlinks_same_target" in reasons


def test_duplicate_collection_two_discovered_symlinks_same_target(env):
    collection = env.collection_path("example-org", "styleguide")
    collection.mkdir(parents=True)
    (collection / "graph.json").write_text("{}", encoding="utf-8")

    root1 = env.scan_root / "styleguide.example-org.dev.lo"
    root2 = env.scan_root / "styleguide-mirror.example-org.dev.lo"
    for r in (root1, root2):
        r.mkdir(parents=True)
        (r / "graphify-out").symlink_to(collection, target_is_directory=True)

    env._repos.append(
        {
            "repo_id": "example-org.styleguide",
            "root": str(root1),
            "collection_path": str(collection),
            "enabled": True,
        }
    )
    env.write_registry()

    discovered = discover_filesystem(env.scan_root, env.scan_root)
    registry = load_registry(env.registry_path)
    report = reconcile(discovered, registry, env.mesh_root)

    dupe = [d for d in report.duplicates if d["reason"] == "multiple_symlinks_same_target"]
    assert len(dupe) == 1
    assert len(dupe[0]["source_roots"]) == 2


def test_rename_project_detected_not_duplicated(env):
    root = env.add_repo(
        "example-org.styleguide", "example-org", "styleguide", "styleguide.example-org.dev.lo"
    )
    env.write_registry()

    collection = env.collection_path("example-org", "styleguide")
    new_root = env.scan_root / "styleguide-renamed.example-org.dev.lo"
    new_root.mkdir(parents=True)
    (new_root / "graphify-out").symlink_to(collection, target_is_directory=True)
    # Old root's symlink removed (simulating the project directory having moved).
    (root / "graphify-out").unlink()
    root.rmdir()

    discovered = discover_filesystem(env.scan_root, env.scan_root)
    registry = load_registry(env.registry_path)
    report = reconcile(discovered, registry, env.mesh_root)

    assert len(report.renamed) == 1
    assert report.renamed[0]["repo_id"] == "example-org.styleguide"
    assert report.renamed[0]["new_root"] == str(new_root)
    assert "example-org.styleguide" not in report.registered
    assert "example-org.styleguide" not in report.removed
    assert not report.duplicates


def test_nested_depth2_discovery_nested_workspace_style(env):
    env.add_repo(
        "example-org.assets",
        "example-org",
        "assets",
        "workspace/assets",
        nested=True,
    )
    env.write_registry()

    discovered = discover_filesystem(env.scan_root, env.scan_root)
    registry = load_registry(env.registry_path)
    report = reconcile(discovered, registry, env.mesh_root)

    assert "example-org.assets" in report.registered


def test_remove_project_pruned_and_flagged(env):
    env.add_repo(
        "example-org.styleguide", "example-org", "styleguide", "styleguide.example-org.dev.lo"
    )
    env.write_registry()

    # Project fully vanishes: root dir + symlink gone entirely.
    import shutil as _shutil

    _shutil.rmtree(env.scan_root / "styleguide.example-org.dev.lo")

    discovered = discover_filesystem(env.scan_root, env.scan_root)
    registry = load_registry(env.registry_path)
    report = reconcile(discovered, registry, env.mesh_root)

    assert "example-org.styleguide" in report.removed
    assert "example-org.styleguide" not in report.registered
