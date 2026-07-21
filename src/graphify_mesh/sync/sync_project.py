"""Per-project decision + invocation + shrink-guard defense (WS1 items 2-4).

Decision (`decide_action`) and outcome classification
(`_classify_post_invoke`) both use dict-dispatch instead of if/elif chains,
per project code style rules.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from graphify_mesh.sync import graphify_cli
from graphify_mesh.sync.state import (
    SourceDigest,
    file_content_hash,
    graph_hash_and_counts,
    is_worktree_dirty,
)

ACTION_SKIP = "skip"
ACTION_UPDATE = "update"
ACTION_EXTRACT = "extract"
ACTION_BOOTSTRAP = "bootstrap"

STATUS_UNCHANGED = "unchanged"
STATUS_UPDATED = "updated"
STATUS_BOOTSTRAPPED = "bootstrapped"
STATUS_BOOTSTRAP_FAILED = "bootstrap_failed"
STATUS_FAILED = "failed"
STATUS_SHRINK_REFUSED = "shrink_refused"
STATUS_NOOP = "noop"


@dataclass
class ProjectOutcome:
    repo_id: str
    action: str
    status: str
    reason: str = ""
    dirty_worktree: bool = False
    new_manifest: SourceDigest | None = None
    # sha256 of the repo's graph.json AS LEFT ON DISK by this outcome, when
    # known. Publish reuses it for the manifest's repo_input_hashes instead
    # of re-reading every per-repo graph.json; None means "not computed here,
    # hash at publish time" (skip/broken/failed paths).
    graph_content_hash: str | None = None


def decide_action(prior_state: dict | None, current_manifest: SourceDigest, has_graph: bool) -> str:
    if not has_graph:
        return ACTION_BOOTSTRAP
    if prior_state is None:
        # First time this repo's manifest is observed but a graph already
        # exists on disk (e.g. pre-existing curated graph). Seed state with a
        # cheap AST refresh rather than forcing an LLM extract every run.
        return ACTION_UPDATE
    if current_manifest.semantic_hash != prior_state.get("semantic_hash"):
        return ACTION_EXTRACT
    if current_manifest.code_hash != prior_state.get("code_hash"):
        return ACTION_UPDATE
    return ACTION_SKIP


def _classify_shrink(
    old_hash: str, new_hash: str, old_counts: tuple[int, int], new_counts: tuple[int, int]
) -> str:
    """C21 shrink-guard defense: never trust CLI exit code alone.

    Mirrors the real CLI's own shrink-guard semantics (export.py:270-286,
    "new graph has N nodes but existing graph.json has M") — compare
    structural node/edge counts, not raw file byte size (byte size is
    sensitive to incidental re-serialization/formatting differences and is
    not a reliable growth signal on its own).
    """
    if new_hash == old_hash:
        return STATUS_NOOP
    new_nodes, new_edges = new_counts
    old_nodes, old_edges = old_counts
    if new_nodes < old_nodes or new_edges < old_edges:
        return STATUS_SHRINK_REFUSED
    return STATUS_UPDATED


_INVOKERS = {
    ACTION_UPDATE: graphify_cli.run_update,
    ACTION_EXTRACT: graphify_cli.run_extract,
    ACTION_BOOTSTRAP: graphify_cli.run_extract,
}


def apply_action(
    repo_id: str,
    graphify_bin: str,
    root: Path,
    collection_path: Path,
    action: str,
    current_manifest: SourceDigest,
) -> ProjectOutcome:
    graph_path = collection_path / "graph.json"
    dirty = is_worktree_dirty(root)

    if action == ACTION_SKIP:
        return ProjectOutcome(
            repo_id, action, STATUS_UNCHANGED, dirty_worktree=dirty, new_manifest=current_manifest
        )

    snapshot_path: Path | None = None
    # Single read of graph.json: hash + counts from the same buffer.
    old_hash, old_counts_opt = graph_hash_and_counts(graph_path)
    old_counts = old_counts_opt or (0, 0)
    if graph_path.exists():
        fd, tmp_name = tempfile.mkstemp(prefix="graphify-mesh-sync-snapshot-", suffix=".json")
        os.close(fd)
        snapshot_path = Path(tmp_name)
        shutil.copy2(graph_path, snapshot_path)

    invoker = _INVOKERS[action]
    result = invoker(graphify_bin, root)

    if not result.ok:
        _restore_snapshot(snapshot_path, graph_path)
        status = STATUS_BOOTSTRAP_FAILED if action == ACTION_BOOTSTRAP else STATUS_FAILED
        _cleanup_snapshot(snapshot_path)
        return ProjectOutcome(
            repo_id,
            action,
            status,
            reason=f"exit={result.returncode}: {result.stderr.strip()[:300]}",
            dirty_worktree=dirty,
        )

    if action == ACTION_BOOTSTRAP:
        if not graph_path.exists():
            _cleanup_snapshot(snapshot_path)
            return ProjectOutcome(
                repo_id,
                action,
                STATUS_BOOTSTRAP_FAILED,
                reason="cli exited 0 but no graph.json was produced",
                dirty_worktree=dirty,
            )
        _cleanup_snapshot(snapshot_path)
        return ProjectOutcome(
            repo_id,
            action,
            STATUS_BOOTSTRAPPED,
            dirty_worktree=dirty,
            new_manifest=current_manifest,
            graph_content_hash=file_content_hash(graph_path),
        )

    if not graph_path.exists():
        _restore_snapshot(snapshot_path, graph_path)
        _cleanup_snapshot(snapshot_path)
        return ProjectOutcome(
            repo_id,
            action,
            STATUS_FAILED,
            reason="cli exited 0 but graph.json disappeared",
            dirty_worktree=dirty,
        )

    # Single read of the freshly written graph.json: hash + counts from the
    # same buffer.
    new_hash, new_counts_opt = graph_hash_and_counts(graph_path)
    new_counts = new_counts_opt or (0, 0)

    if old_hash is None:
        # No prior file existed even though has_graph was assumed true
        # upstream (race) — accept whatever was produced.
        _cleanup_snapshot(snapshot_path)
        return ProjectOutcome(
            repo_id,
            action,
            STATUS_UPDATED,
            dirty_worktree=dirty,
            new_manifest=current_manifest,
            graph_content_hash=new_hash,
        )

    outcome_status = _classify_shrink(old_hash, new_hash or "", old_counts, new_counts)

    if outcome_status == STATUS_SHRINK_REFUSED:
        _restore_snapshot(snapshot_path, graph_path)
        _cleanup_snapshot(snapshot_path)
        return ProjectOutcome(
            repo_id,
            action,
            STATUS_SHRINK_REFUSED,
            reason=(
                f"cli reported success but node/edge counts did not grow "
                f"(old={old_counts} new={new_counts}); last-good graph.json restored"
            ),
            dirty_worktree=dirty,
        )

    _cleanup_snapshot(snapshot_path)
    # State advances to current_manifest for BOTH remaining statuses here
    # (STATUS_UPDATED and STATUS_NOOP): a NOOP means the source changed but
    # the CLI produced byte-identical output — the source digest must still
    # advance, or every subsequent run would re-invoke the CLI for the same
    # no-op change. The two branches were always identical; the old
    # conditional was dead.
    return ProjectOutcome(
        repo_id,
        action,
        outcome_status,
        dirty_worktree=dirty,
        new_manifest=current_manifest,
        graph_content_hash=new_hash,
    )


def _restore_snapshot(snapshot_path: Path | None, graph_path: Path) -> None:
    if snapshot_path is None:
        return
    shutil.copy2(snapshot_path, graph_path)


def _cleanup_snapshot(snapshot_path: Path | None) -> None:
    if snapshot_path is not None and snapshot_path.exists():
        snapshot_path.unlink()
