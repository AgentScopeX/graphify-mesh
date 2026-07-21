"""Shared bounded cache of source-file line lists for snippet building.

Both snippet consumers (`embedding.build_snippet`, used per-node by the WS3
embed stage AND per-node again by the WS1.6 lexical-index stage via
`lexical_index.build_lexical_index`) used to open and line-scan the same
source file once per node, per stage — a file contributing 50 nodes was
opened ~100 times per sync run. This module reads a file's lines ONCE (up to
the caller's line cap, mirroring `SNIPPET_READ_LINE_CAP`) and lets every
snippet window slice out of the cached tuple instead.

Memory bounds (both are hard caps, so a pathological repo can't blow RSS):

  * per file: at most ``line_cap`` lines are ever read or stored — the same
    cap the old streaming reader enforced, so cached content is exactly the
    prefix the per-call scan used to see;
  * total: at most ``SOURCE_CACHE_MAX_FILES`` files are resident at once
    (``functools.lru_cache`` eviction).

Callers keep full responsibility for path validation (absolute-path and
``..``-traversal rejection against ``source_root``) BEFORE consulting the
cache — the cache is keyed on the already-resolved path and never resolves
or validates anything itself.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# Upper bound on distinct files resident in the cache at once. 256 files x
# SNIPPET_READ_LINE_CAP (4000) lines is a small, fixed ceiling regardless of
# how many nodes or repos a run touches.
SOURCE_CACHE_MAX_FILES = 256


@lru_cache(maxsize=SOURCE_CACHE_MAX_FILES)
def _read_capped_lines(
    path_str: str, st_mtime_ns: int, st_size: int, line_cap: int
) -> tuple[str, ...] | None:
    """Reads up to ``line_cap`` newline-stripped lines of ``path_str``.

    Keyed on file identity + version (``st_mtime_ns``/``st_size`` from the
    caller's ``os.stat``) so a long-lived process (the MCP server) never
    serves pre-edit lines after the file changes on disk.

    Encoding/error handling is byte-for-byte the old per-call reader's:
    ``utf-8`` with ``errors="replace"``, each line ``rstrip("\\n")``-ed.
    Returns ``None`` (cached, like any other result) when the file cannot be
    read — callers treat that exactly like the old reader's OSError path
    (empty snippet, never an exception)."""
    lines: list[str] = []
    try:
        with open(path_str, encoding="utf-8", errors="replace") as fh:
            for idx, raw_line in enumerate(fh):
                if idx >= line_cap:
                    break
                lines.append(raw_line.rstrip("\n"))
    except OSError:
        return None
    return tuple(lines)


def get_source_lines(path: Path, line_cap: int) -> tuple[str, ...] | None:
    """Cached line list for an already-validated, already-resolved source
    path. ``None`` means the file was unreadable (missing, permissions, ...)
    — the caller degrades to an empty snippet, same as before the cache.

    One ``os.stat`` per call keys the cache on file version; a failed stat
    returns the unreadable sentinel WITHOUT touching the cache, so no bogus
    key is ever cached for a missing file."""
    path_str = str(path)
    try:
        stat_result = os.stat(path_str)
    except OSError:
        return None
    return _read_capped_lines(path_str, stat_result.st_mtime_ns, stat_result.st_size, line_cap)


def clear_source_cache() -> None:
    """Drops every cached file. Exposed for tests and for long-lived callers
    that want a fresh view of the filesystem between logical runs."""
    _read_capped_lines.cache_clear()
