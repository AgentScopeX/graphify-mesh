"""Per-project source manifest state (untracked, under graphify/global/state/).

Used to decide `graphify update` (code-only change) vs `graphify extract`
(semantic/docs/config change) per WS1 item 2, and to detect dirty worktrees
(WS1 item 4) without ever running a mutating git command.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from graphify_mesh.sync.config import IGNORED_DIR_NAMES, categorize_file

log = logging.getLogger("graphify_mesh.sync")


@dataclass
class SourceDigest:
    code_hash: str
    semantic_hash: str
    file_count: int

    def to_dict(self) -> dict:
        return {
            "code_hash": self.code_hash,
            "semantic_hash": self.semantic_hash,
            "file_count": self.file_count,
        }


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

    return SourceDigest(
        code_hash=_hash(code_entries), semantic_hash=_hash(semantic_entries), file_count=count
    )


def load_state(state_path: Path) -> dict:
    """Load the per-project source-manifest state. A corrupt or unreadable
    state file (torn write after power loss, manual tampering) must never
    brick every subsequent sync run — treat it exactly like a missing file:
    start from empty state. Worst case is one full re-extract cycle, which
    is self-healing; an unhandled JSONDecodeError here would require manual
    cleanup before any run could succeed again."""
    if not state_path.exists():
        return {}
    try:
        loaded = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(
            "state file %s is unreadable/corrupt (%s) — starting from empty state; "
            "all projects will be treated as changed this run",
            state_path,
            exc,
        )
        return {}
    if not isinstance(loaded, dict):
        log.warning(
            "state file %s does not contain a JSON object — starting from empty state",
            state_path,
        )
        return {}
    return loaded


def save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    # fsync the file DATA before the rename: without it, a power loss can
    # make the rename durable while the contents are not, leaving a
    # truncated/empty state file behind (the exact corruption load_state
    # tolerates above — but better to not produce it in the first place).
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(state, indent=2, sort_keys=True))
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(state_path)


def is_worktree_dirty(root: Path) -> bool:
    """Read-only `git status --porcelain` check. Never mutates the target repo."""
    if not (root / ".git").exists():
        return False
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607 - system git from PATH, read-only status
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
