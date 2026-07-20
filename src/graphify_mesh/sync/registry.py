"""Registry (repo-identity map) load/parse.

Registry is the source of truth for repo identity. A stray `.graphify_root`
file on disk (some of which just contain "." junk, per plan M7) must never
override it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RepoEntry:
    repo_id: str
    root: Path
    collection_path: Path
    enabled: bool = True


@dataclass
class Registry:
    repos: list[RepoEntry] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)
    external_roots: list[str] = field(default_factory=list)

    def by_collection_path(self) -> dict[str, RepoEntry]:
        return {str(r.collection_path): r for r in self.repos}

    def by_repo_id(self) -> dict[str, RepoEntry]:
        return {r.repo_id: r for r in self.repos}


def load_registry(path: Path) -> Registry:
    if not path.exists():
        return Registry()
    raw = json.loads(path.read_text(encoding="utf-8"))
    repos = [
        RepoEntry(
            repo_id=entry["repo_id"],
            root=Path(entry["root"]),
            collection_path=Path(entry["collection_path"]),
            enabled=bool(entry.get("enabled", True)),
        )
        for entry in raw.get("repos", [])
    ]
    return Registry(
        repos=repos,
        disabled=list(raw.get("disabled", [])),
        external_roots=list(raw.get("external_roots", [])),
    )


def registry_hash(path: Path) -> str:
    """Stable content hash of the registry file, for the generation manifest (C28)."""
    import hashlib

    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
