"""Per-project source manifest state (untracked, under graphify/global/state/).

Used to decide `graphify update` (code-only change) vs `graphify extract`
(semantic/docs/config change) per WS1 item 2, and to detect dirty worktrees
(WS1 item 4) without ever running a mutating git command.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from graphify_mesh.sync.config import IGNORED_DIR_NAMES, categorize_file


@dataclass
class SourceDigest:
    code_hash: str
    semantic_hash: str
    file_count: int

    def to_dict(self) -> dict:
        return {"code_hash": self.code_hash, "semantic_hash": self.semantic_hash, "file_count": self.file_count}


def compute_source_manifest(root: Path) -> SourceDigest:
    """Walk root, hashing (relpath, size, mtime_ns) separately per category."""
    code_entries: list[str] = []
    semantic_entries: list[str] = []
    count = 0
    if not root.is_dir():
        return SourceDigest(code_hash="empty", semantic_hash="empty", file_count=0)

    for path in sorted(root.rglob("*")):
        if any(part in IGNORED_DIR_NAMES for part in path.parts):
            continue
        if not path.is_file():
            continue
        category = categorize_file(path)
        if category == "ignore":
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        rel = str(path.relative_to(root))
        entry = f"{rel}:{stat.st_size}:{stat.st_mtime_ns}"
        count += 1
        if category == "code":
            code_entries.append(entry)
        else:
            semantic_entries.append(entry)

    def _hash(entries: list[str]) -> str:
        h = hashlib.sha256()
        for e in entries:
            h.update(e.encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()[:16]

    return SourceDigest(code_hash=_hash(code_entries), semantic_hash=_hash(semantic_entries), file_count=count)


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(state_path)


def is_worktree_dirty(root: Path) -> bool:
    """Read-only `git status --porcelain` check. Never mutates the target repo."""
    if not (root / ".git").exists():
        return False
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return bool(result.stdout.strip())


def file_content_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def graph_node_edge_counts(path: Path) -> tuple[int, int] | None:
    """Cheap count via json.load — never used on real production-sized graphs
    from the exploring agent's context (subprocess/engine-internal only)."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return len(data.get("nodes", [])), len(data.get("links", data.get("edges", [])))
