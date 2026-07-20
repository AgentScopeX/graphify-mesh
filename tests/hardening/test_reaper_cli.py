"""graphify-mesh-reap CLI wiring — argument parsing and output format only;
the actual scan logic is covered by test_reaper.py against
graphify_mesh.sync.reaper directly."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from graphify_mesh.sync import reaper_cli as _cli  # noqa: E402


def test_default_is_report_only_not_kill(monkeypatch):
    captured = {}

    def fake_run_reaper(kill=False):
        captured["kill"] = kill
        return []

    monkeypatch.setattr(_cli, "run_reaper", fake_run_reaper)
    rc = _cli.main([])
    assert rc == 0
    assert captured["kill"] is False


def test_kill_flag_is_forwarded(monkeypatch):
    captured = {}

    def fake_run_reaper(kill=False):
        captured["kill"] = kill
        return []

    monkeypatch.setattr(_cli, "run_reaper", fake_run_reaper)
    rc = _cli.main(["--kill"])
    assert rc == 0
    assert captured["kill"] is True


def test_json_output_is_valid_json(monkeypatch, capsys):
    from graphify_mesh.sync.reaper import ReapCandidate

    monkeypatch.setattr(
        _cli,
        "run_reaper",
        lambda kill=False: [
            ReapCandidate(pid=1, ppid=1, args="python -m graphify.serve x", reason="ppid==1")
        ],
    )
    rc = _cli.main(["--json"])
    assert rc == 0
    out = capsys.readouterr().out
    import json

    parsed = json.loads(out)
    assert parsed[0]["pid"] == 1


def test_scan_failure_returns_nonzero(monkeypatch, capsys):
    def boom(kill=False):
        raise RuntimeError("ps not found")

    monkeypatch.setattr(_cli, "run_reaper", boom)
    rc = _cli.main([])
    assert rc == 1
    assert "scan failed" in capsys.readouterr().err
