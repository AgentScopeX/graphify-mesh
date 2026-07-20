from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

from graphify_mesh.sync import backend
from graphify_mesh.sync.config import PINNED_CLUSTERING_BACKEND


def _make_executable(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _build_interpreter_without_graspologic(tmp_path: Path) -> Path:
    """A `#!/bin/sh` shim that just execs the real system interpreter —
    graspologic is not importable there (confirmed: neither the pipx venv
    nor ~/.local graphify install nor system python3 has it installed)."""
    interp = tmp_path / "interp_no_graspologic.sh"
    return _make_executable(interp, f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')


def _build_interpreter_with_graspologic(tmp_path: Path) -> Path:
    """A `#!/bin/sh` shim that execs the system interpreter with PYTHONPATH
    pointed at a directory containing a stub `graspologic` package (just an
    empty `__init__.py`) — enough for `importlib.util.find_spec` to resolve
    it, without installing anything real or touching global/system state."""
    stub_root = tmp_path / "stub_site"
    stub_pkg = stub_root / "graspologic"
    stub_pkg.mkdir(parents=True)
    (stub_pkg / "__init__.py").write_text("", encoding="utf-8")
    interp = tmp_path / "interp_with_graspologic.sh"
    return _make_executable(
        interp,
        f'#!/bin/sh\nexec env PYTHONPATH="{stub_root}" "{sys.executable}" "$@"\n',
    )


def _build_graphify_bin(tmp_path: Path, name: str, interpreter: Path) -> Path:
    """A fake `graphify_bin` whose only relevant property is its shebang
    line — `_resolve_interpreter` never executes this file, only reads its
    first line."""
    script = tmp_path / name
    return _make_executable(script, f"#!{interpreter}\nprint('fake graphify')\n")


def test_assert_pinned_backend_matches_when_graspologic_absent(tmp_path):
    interp = _build_interpreter_without_graspologic(tmp_path)
    graphify_bin = _build_graphify_bin(tmp_path, "graphify_louvain_only", interp)

    result = backend.assert_pinned_backend(str(graphify_bin))

    assert PINNED_CLUSTERING_BACKEND == backend.LOUVAIN_BACKEND
    assert result.backend == backend.LOUVAIN_BACKEND
    assert result.matches_pinned


def test_assert_pinned_backend_raises_on_mismatch_when_graspologic_importable(tmp_path):
    interp = _build_interpreter_with_graspologic(tmp_path)
    graphify_bin = _build_graphify_bin(tmp_path, "graphify_leiden_capable", interp)

    with pytest.raises(backend.BackendMismatchError):
        backend.assert_pinned_backend(str(graphify_bin))


def test_detect_actual_backend_without_raising_for_match(tmp_path):
    interp = _build_interpreter_without_graspologic(tmp_path)
    graphify_bin = _build_graphify_bin(tmp_path, "graphify", interp)

    assert backend.detect_actual_backend(str(graphify_bin)) == backend.LOUVAIN_BACKEND
