"""Filesystem discovery + reconciliation against the registry (WS1 item 1).

Scans `<scan_root>/*/graphify-out` and one level of nesting
(`<scan_root>/*/*/graphify-out`, e.g. AgentSpaceX's layout), resolves
symlinks to real paths, and rejects anything that resolves outside the
approved root (C16 path-traversal guard). Reconciles the result against
`registry.json` (the source of truth for repo identity) to produce a report
of registered / renamed / missing / broken / removed / duplicate projects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from graphify_mesh.sync.registry import Registry


@dataclass
class DiscoveredLink:
    source_root: Path
    link_path: Path
    target: Path | None
    broken: bool = False
    rejected_traversal: bool = False


@dataclass
class ReconciliationReport:
    registered: list[str] = field(default_factory=list)
    renamed: list[dict] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    broken: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    duplicates: list[dict] = field(default_factory=list)
    unregistered_discovered: list[str] = field(default_factory=list)
    auto_add: list[str] = field(default_factory=list)
    rejected_traversal: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "registered": self.registered,
            "renamed": self.renamed,
            "missing": self.missing,
            "broken": self.broken,
            "removed": self.removed,
            "duplicates": self.duplicates,
            "unregistered_discovered": self.unregistered_discovered,
            "auto_add": self.auto_add,
            "rejected_traversal": self.rejected_traversal,
        }


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def discover_filesystem(scan_root: Path, approved_root: Path) -> list[DiscoveredLink]:
    """Scan for `graphify-out` symlinks/dirs at depth 1 and depth 2 under scan_root."""
    scan_root = scan_root.resolve()
    approved_root = approved_root.resolve()
    results: list[DiscoveredLink] = []
    if not scan_root.is_dir():
        return results

    candidate_dirs: list[Path] = []
    for child in sorted(scan_root.iterdir()):
        if not child.is_dir():
            continue
        candidate_dirs.append(child)
        for grandchild in sorted(child.iterdir()) if child.is_dir() else []:
            if grandchild.is_dir():
                candidate_dirs.append(grandchild)

    for project_dir in candidate_dirs:
        link_path = project_dir / "graphify-out"
        if not link_path.is_symlink() and not link_path.is_dir():
            continue
        if not link_path.exists():
            # Dangling symlink (target missing) — broken, do not crash.
            results.append(
                DiscoveredLink(source_root=project_dir, link_path=link_path, target=None, broken=True)
            )
            continue
        target = link_path.resolve()
        if not _is_under(target, approved_root):
            # Path traversal guard: resolved symlink target escapes the
            # approved root (the configured scan root). A legitimate
            # graphify-out target always resolves inside approved_root;
            # anything else is rejected.
            results.append(
                DiscoveredLink(
                    source_root=project_dir, link_path=link_path, target=target, rejected_traversal=True
                )
            )
            continue
        results.append(DiscoveredLink(source_root=project_dir, link_path=link_path, target=target))
    return results


def reconcile(
    discovered: list[DiscoveredLink],
    registry: Registry,
    mesh_root: Path,
) -> ReconciliationReport:
    report = ReconciliationReport()
    mesh_root = mesh_root.resolve()

    # Guard: a discovered target must resolve under the mesh tree (that's the
    # only legitimate destination for a graphify-out symlink in this design).
    valid_discovered = []
    for d in discovered:
        if d.rejected_traversal:
            report.rejected_traversal.append(str(d.link_path))
            continue
        if d.target is not None and not _is_under(d.target, mesh_root):
            report.rejected_traversal.append(str(d.link_path))
            continue
        valid_discovered.append(d)

    # Registry-internal duplicate collection_path detection.
    seen_collection_paths: dict[str, list[str]] = {}
    for entry in registry.repos:
        seen_collection_paths.setdefault(str(entry.collection_path), []).append(entry.repo_id)
    for collection_path, repo_ids in seen_collection_paths.items():
        if len(repo_ids) > 1:
            report.duplicates.append(
                {"reason": "registry_duplicate_collection_path", "collection_path": collection_path, "repo_ids": repo_ids}
            )

    # Map resolved target -> list of discovered links pointing at it (dedup
    # detection for two discovered symlinks resolving to the same real path).
    by_target: dict[str, list[DiscoveredLink]] = {}
    for d in valid_discovered:
        if d.target is None:
            continue
        by_target.setdefault(str(d.target), []).append(d)
    for target, links in by_target.items():
        if len(links) > 1:
            report.duplicates.append(
                {
                    "reason": "multiple_symlinks_same_target",
                    "collection_path": target,
                    "source_roots": sorted(str(l.source_root) for l in links),
                }
            )

    broken_by_root = {str(d.source_root): d for d in valid_discovered if d.broken}

    registered_collection_paths = set(seen_collection_paths.keys())

    for entry in registry.repos:
        if entry.repo_id in registry.disabled or not entry.enabled:
            continue
        cp = str(entry.collection_path)
        matches = by_target.get(cp, [])

        if not matches:
            broken_link = broken_by_root.get(str(entry.root))
            if broken_link is not None:
                report.broken.append(entry.repo_id)
            elif not entry.collection_path.exists():
                report.missing.append(entry.repo_id)
            else:
                report.removed.append(entry.repo_id)
        else:
            canonical = sorted(matches, key=lambda d: str(d.source_root))[0]
            if canonical.source_root == entry.root:
                report.registered.append(entry.repo_id)
            else:
                report.renamed.append(
                    {"repo_id": entry.repo_id, "old_root": str(entry.root), "new_root": str(canonical.source_root)}
                )

        graph_file = entry.collection_path / "graph.json"
        if entry.collection_path.exists() and not graph_file.exists():
            report.auto_add.append(entry.repo_id)

    for d in valid_discovered:
        if d.target is None:
            continue
        if str(d.target) not in registered_collection_paths:
            report.unregistered_discovered.append(str(d.source_root))

    return report
