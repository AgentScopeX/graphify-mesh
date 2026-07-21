"""Whole-transaction lock (sync/locking.py) + the CLI's lock-held exit path."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from graphify_mesh.sync import cli
from graphify_mesh.sync.locking import LockHeldError, transaction_lock

SRC_DIR = Path(__file__).resolve().parents[2] / "src"

# Child process that acquires the transaction lock, signals readiness by
# creating the `ready` file, then holds the lock until it is killed (or the
# safety timeout expires so a broken test can never leak a sleeper).
_HOLDER_SCRIPT = """
import sys
import time
from pathlib import Path

from graphify_mesh.sync.locking import transaction_lock

lock_path = Path(sys.argv[1])
ready_path = Path(sys.argv[2])

with transaction_lock(lock_path):
    ready_path.write_text("ok", encoding="utf-8")
    time.sleep(60)
"""


def _spawn_lock_holder(lock_path: Path, ready_path: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", _HOLDER_SCRIPT, str(lock_path), str(ready_path)],
        env={"PYTHONPATH": str(SRC_DIR), "PATH": "/usr/bin:/bin"},
    )


def _wait_for_file(path: Path, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"lock-holder subprocess never signalled readiness via {path}")


def test_lock_held_by_another_process_raises_lock_held_error(tmp_path):
    lock_path = tmp_path / "locks" / "sync.lock"
    ready_path = tmp_path / "ready"
    holder = _spawn_lock_holder(lock_path, ready_path)
    try:
        _wait_for_file(ready_path)
        with pytest.raises(LockHeldError):
            with transaction_lock(lock_path):
                pass
    finally:
        holder.kill()
        holder.wait(timeout=10)


def test_lock_released_when_with_block_body_raises(tmp_path):
    lock_path = tmp_path / "sync.lock"

    with pytest.raises(RuntimeError, match="boom"):
        with transaction_lock(lock_path):
            raise RuntimeError("boom")

    # flock is held per open file description, so a second open()+flock in
    # this same process WOULD conflict if the first acquisition leaked.
    # Re-acquiring cleanly proves the finally-path released and closed it.
    with transaction_lock(lock_path):
        pass


def test_lock_released_after_clean_exit(tmp_path):
    lock_path = tmp_path / "sync.lock"
    with transaction_lock(lock_path):
        pass
    with transaction_lock(lock_path):
        pass


def test_cli_exits_3_when_lock_is_held(tmp_path, monkeypatch, capsys):
    """cli.main's LockHeldError branch: it must print to stderr and return
    exit code 3, never propagate the exception or return 0."""

    def _run_raising_lock_held(settings):
        raise LockHeldError("another graphify-mesh-sync run holds the lock")

    # cli.py does `from graphify_mesh.sync.pipeline import run`, so the name
    # to patch lives in the cli module namespace itself.
    monkeypatch.setattr(cli, "run", _run_raising_lock_held)

    exit_code = cli.main(
        [
            "--once",
            "--mesh-root",
            str(tmp_path / "mesh"),
            "--scan-root",
            str(tmp_path / "www"),
        ]
    )

    assert exit_code == 3
    captured = capsys.readouterr()
    assert "another graphify-mesh-sync run holds the lock" in captured.err
