"""Versioned generation publish + atomic `current` symlink flip (WS1 item 8).

Publish never deletes a previous generation. If validation fails or the
stale-repo threshold is exceeded, the caller simply never calls `publish()`
— "rollback" is defined as `current` never having been moved, which is why
this module never touches `current` except in the final atomic rename.

WS6: for the manual rollback procedure (reverting to an OLDER generation
than whatever `current` presently points at, past the automatic cases
above), see `bin/graphify_mesh.sync/ROLLBACK.md`.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path


def make_generation_id(output_hash: str) -> str:
    # Second-precision timestamp + content-hash prefix is not sufficient to
    # guarantee uniqueness on its own: two runs within the same second that
    # merge to byte-identical output (e.g. nothing changed) would otherwise
    # collide on generation_id. Add a short random nonce.
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    nonce = os.urandom(3).hex()
    return f"{ts}-{output_hash[:8]}-{nonce}"


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_atomic(path: Path, data: dict, **dumps_kwargs) -> None:
    """Write via tmp-file + rename so an interrupted write can never leave a
    truncated .json in a generation dir. `current` is only flipped after all
    writes, so a partial file was never *served* — but a stranded generation
    dir with half a graph in it is a trap for the manual-rollback procedure
    (ROLLBACK.md points operators at old generation dirs directly)."""
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(data, **dumps_kwargs), encoding="utf-8")
    os.rename(str(tmp_path), str(path))


def write_generation(
    generations_dir: Path, generation_id: str, graph_data: dict, manifest: dict
) -> Path:
    gen_dir = generations_dir / generation_id
    gen_dir.mkdir(parents=True, exist_ok=False)
    graph_path = gen_dir / "global-graph.json"
    manifest_path = gen_dir / "generation-manifest.json"
    _write_json_atomic(graph_path, graph_data, indent=2)
    _write_json_atomic(manifest_path, manifest, indent=2, sort_keys=True)
    _fsync_dir(gen_dir)
    _fsync_dir(generations_dir)
    return gen_dir


def write_overlay(gen_dir: Path, overlay_data: dict) -> Path:
    """WS4: stage the cross-project overlay artifact inside the SAME
    generation dir as `global-graph.json`/`generation-manifest.json`, so it
    flips atomically with everything else on `flip_current` — but as an
    entirely separate file, never merged into `global-graph.json` (C5)."""
    overlay_path = gen_dir / "cross-project-overlay.json"
    _write_json_atomic(overlay_path, overlay_data, indent=2, sort_keys=True)
    _fsync_dir(gen_dir)
    return overlay_path


def write_lexical_index(gen_dir: Path, lexical_data: dict) -> Path:
    """WS5: stage the lexical-index bundle artifact inside the SAME
    generation dir as `global-graph.json`/`cross-project-overlay.json`, so it
    flips atomically with everything else on `flip_current`. Consumed
    read-only by the graphify-mesh companion MCP server, never written to at
    query time."""
    lexical_path = gen_dir / "lexical-index.json"
    _write_json_atomic(lexical_path, lexical_data, indent=2, sort_keys=True)
    _fsync_dir(gen_dir)
    return lexical_path


def read_current_lexical_index(global_dir: Path) -> dict | None:
    current = global_dir / "current"
    if not current.exists():
        return None
    lexical_path = current / "lexical-index.json"
    if not lexical_path.exists():
        return None
    try:
        return json.loads(lexical_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_current_overlay(global_dir: Path) -> dict | None:
    current = global_dir / "current"
    if not current.exists():
        return None
    overlay_path = current / "cross-project-overlay.json"
    if not overlay_path.exists():
        return None
    try:
        return json.loads(overlay_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def flip_current(global_dir: Path, gen_dir: Path) -> None:
    """Atomically flip `current` -> gen_dir via rename of a temp symlink."""
    current = global_dir / "current"
    tmp_link = global_dir / ".current.tmp"
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    tmp_link.symlink_to(gen_dir.resolve(), target_is_directory=True)
    os.rename(str(tmp_link), str(current))
    _fsync_dir(global_dir)


def prune_old_generations(generations_dir: Path, current: Path, keep: int = 2) -> list[str]:
    """Delete all generation dirs under `generations_dir` except the one
    `current` points at and the `keep - 1` most recent others (by directory
    name, which sorts chronologically — `make_generation_id` is a
    zero-padded UTC timestamp prefix).

    Each generation is a full copy of the merged graph + overlay + lexical
    index (global-graph.json alone can be tens of MB, lexical-index.json can
    exceed 100MB) — with no GC, a generation this large published on every
    scheduled run (e.g. hourly) accumulates without bound. `write_generation`
    never deletes anything itself (by design, so a crash mid-write never
    corrupts a previous good generation) — pruning only ever runs here,
    AFTER `flip_current` has already succeeded, so a crash during pruning
    can strand extra generation dirs (wasted disk) but can never remove the
    one `current` needs.

    Also removes generation dirs that never finished publishing (e.g. a
    dangling `<name>/lexical-index.json.tmp` with no matching `.json` —
    the process was killed between `write_lexical_index`'s tmp-write and its
    rename) — these were never `current` and are safe to delete outright,
    keep-count aside.
    """
    if not generations_dir.is_dir():
        return []
    current_name = os.path.basename(os.path.realpath(current)) if current.exists() else None
    all_names = sorted(p.name for p in generations_dir.iterdir() if p.is_dir())

    def _is_incomplete(name: str) -> bool:
        gen_dir = generations_dir / name
        return any(gen_dir.glob("*.tmp"))

    incomplete = [n for n in all_names if n != current_name and _is_incomplete(n)]
    complete = [n for n in all_names if n not in incomplete]
    # Keep the most recent `keep` complete generations (current is always
    # among the most recent, but pin it explicitly in case clock skew ever
    # makes it sort out of the tail).
    keep_set = set(complete[-keep:]) if keep > 0 else set()
    if current_name is not None:
        keep_set.add(current_name)
    to_remove = incomplete + [n for n in complete if n not in keep_set]

    removed = []
    for name in to_remove:
        target = generations_dir / name
        shutil.rmtree(target, ignore_errors=True)
        removed.append(name)
    return removed


def output_hash(graph_data: dict) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(graph_data, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def read_current_manifest(global_dir: Path) -> dict | None:
    current = global_dir / "current"
    if not current.exists():
        return None
    manifest_path = current / "generation-manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_current_global_graph(global_dir: Path) -> dict | None:
    """Load the previously PUBLISHED global graph (global-graph.json behind
    the `current` symlink), not just its manifest.

    Used by the WS2 naming stage's degraded-mode fallback (C23): when Ollama
    is down and this generation's fresh clustering/labeling is skipped
    entirely, `community_name` is restored per-node only from this
    previously-published GLOBAL artifact — never from the current merge's
    per-project inputs — so a transient outage can't leak a per-project
    community name into the global graph.
    """
    current = global_dir / "current"
    if not current.exists():
        return None
    graph_path = current / "global-graph.json"
    if not graph_path.exists():
        return None
    try:
        return json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
