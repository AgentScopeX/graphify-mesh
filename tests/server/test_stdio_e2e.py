"""End-to-end stdio transport test: launches the REAL `graphify_mesh.server.server` module
entrypoint as a subprocess (synthetic fixture mesh root only — never a real
a real project), talks newline-delimited JSON-RPC 2.0 to it exactly like a
real MCP client would, and confirms:
  * `initialize` / `tools/list` / `tools/call` all work over the real pipe
    (not just via `handle_message` in-process, which the other tests use).
  * the process exits cleanly and promptly when stdin is closed (WS6).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"


def _write_registry(mesh_root: Path) -> None:
    registry_path = mesh_root / "bin" / "registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps({"repos": [], "disabled": [], "external_roots": []}), encoding="utf-8"
    )


def _spawn(mesh_root: Path) -> subprocess.Popen:
    env = {
        "GRAPHIFY_MESH_ROOT": str(mesh_root),
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(SRC_DIR),
    }
    return subprocess.Popen(
        [sys.executable, "-m", "graphify_mesh.server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        bufsize=1,
    )


def _send(proc: subprocess.Popen, message: dict) -> dict:
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    assert line, "server produced no response (check stderr for a crash)"
    return json.loads(line)


def test_stdio_initialize_and_tools_list_over_real_subprocess(tmp_path):
    _write_registry(tmp_path)
    proc = _spawn(tmp_path)
    try:
        init_resp = _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert init_resp["result"]["serverInfo"]["name"] == "graphify-mesh"

        list_resp = _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = {t["name"] for t in list_resp["result"]["tools"]}
        assert names == {"search", "cross_project", "find_similar", "project_map", "context_pack"}
    finally:
        proc.stdin.close()
        proc.wait(timeout=5)


def test_stdio_tool_call_degrades_gracefully_with_no_published_generation(tmp_path):
    _write_registry(tmp_path)
    proc = _spawn(tmp_path)
    try:
        resp = _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "search", "arguments": {"q": "anything", "scope": "all"}},
            },
        )
        assert resp["result"]["isError"] is True
    finally:
        proc.stdin.close()
        proc.wait(timeout=5)


def test_process_exits_cleanly_on_stdin_close(tmp_path):
    """WS6: 'companion server must exit cleanly on stdin close' — the
    observed leak was 10 stale `graphify.serve` processes that never did
    this. Verify exit code 0 and a bounded wait, not a hang."""
    _write_registry(tmp_path)
    proc = _spawn(tmp_path)
    proc.stdin.close()
    returncode = proc.wait(timeout=5)
    assert returncode == 0
