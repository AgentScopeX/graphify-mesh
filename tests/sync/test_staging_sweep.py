"""Startup sweep of stale staging dirs left behind by SIGKILLed runs.

`run()`'s `finally` never executes on SIGKILL/OOM, so
`graphify-mesh-sync-staging-*` dirs accumulate in the temp dir; the sweep
(_sweep_stale_staging, called under the transaction lock) is what reclaims
them on the next run."""

from __future__ import annotations

import os
import time

from graphify_mesh.sync.pipeline import (
    STAGING_PREFIX,
    STALE_STAGING_MAX_AGE_SECONDS,
    _sweep_stale_staging,
)


def _make_staging(parent, name, age_seconds):
    d = parent / (STAGING_PREFIX + name)
    d.mkdir(parents=True)
    (d / "merged-graph.json").write_text("{}", encoding="utf-8")
    stale_mtime = time.time() - age_seconds
    os.utime(d, (stale_mtime, stale_mtime))
    return d


def test_sweep_removes_only_stale_siblings(tmp_path):
    own = tmp_path / (STAGING_PREFIX + "own")
    own.mkdir()
    stale = _make_staging(tmp_path, "stale", STALE_STAGING_MAX_AGE_SECONDS + 60)
    fresh = _make_staging(tmp_path, "fresh", 60)
    unrelated = tmp_path / "some-other-tempdir"
    unrelated.mkdir()

    removed = _sweep_stale_staging(own)

    assert removed == [stale.name]
    assert not stale.exists()
    assert own.exists()
    assert fresh.exists()
    assert unrelated.exists()


def test_sweep_never_removes_own_staging_even_if_old(tmp_path):
    own = _make_staging(tmp_path, "own-but-old", STALE_STAGING_MAX_AGE_SECONDS + 60)
    removed = _sweep_stale_staging(own)
    assert removed == []
    assert own.exists()


def test_sweep_ignores_prefix_matching_files(tmp_path):
    own = tmp_path / (STAGING_PREFIX + "own")
    own.mkdir()
    not_a_dir = tmp_path / (STAGING_PREFIX + "file")
    not_a_dir.write_text("x", encoding="utf-8")
    stale_mtime = time.time() - STALE_STAGING_MAX_AGE_SECONDS - 60
    os.utime(not_a_dir, (stale_mtime, stale_mtime))

    removed = _sweep_stale_staging(own)

    assert removed == []
    assert not_a_dir.exists()
