"""JSON-RPC 2.0 protocol-error conformance for the stdio transport
(server/protocol.py + server/server.py).

Spec under test (may land after this file — these tests encode the CONTRACT,
not whatever the current implementation happens to do):

  * unparseable JSON line        -> error response, code -32700, id null
  * non-dict message / batch []  -> error response, code -32600
  * handler raising              -> error response, code -32603
  * notification (no "id")       -> no response written at all
  * tools/call with "params": [] -> invalid-params error (-32602) or the
                                    server's typed ToolError surface — NEVER
                                    a generic internal error

Frames are fed over the same real-subprocess stdio harness that
tests/server/test_stdio_e2e.py uses (raw newline-delimited lines on the
actual pipe), except the handler-raise case, which needs an injected
raising handler and therefore drives `protocol.serve` with in-memory
streams.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from graphify_mesh.server import protocol

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


def _send_raw(proc: subprocess.Popen, raw_line: str) -> None:
    proc.stdin.write(raw_line + "\n")
    proc.stdin.flush()


def _read_response(proc: subprocess.Popen) -> dict:
    line = proc.stdout.readline()
    assert line, "server produced no response line (check stderr for a crash)"
    return json.loads(line)


@pytest.fixture()
def server_proc(tmp_path):
    _write_registry(tmp_path)
    proc = _spawn(tmp_path)
    yield proc
    proc.stdin.close()
    proc.wait(timeout=5)


def test_unparseable_json_line_gets_parse_error_with_null_id(server_proc):
    """A line that isn't JSON at all must be answered with a JSON-RPC parse
    error (-32700) whose id is null — not silently dropped. The follow-up
    ping proves ordering: the FIRST response on the pipe must be the parse
    error, not the ping's."""
    _send_raw(server_proc, "{this is not json")
    _send_raw(server_proc, json.dumps({"jsonrpc": "2.0", "id": 77, "method": "ping"}))

    first = _read_response(server_proc)
    assert first.get("id") is None
    assert first["error"]["code"] == -32700


def test_non_dict_message_gets_invalid_request(server_proc):
    """Valid JSON that is not an object (e.g. a bare number) is not a valid
    JSON-RPC request -> -32600."""
    _send_raw(server_proc, "42")
    _send_raw(server_proc, json.dumps({"jsonrpc": "2.0", "id": 78, "method": "ping"}))

    first = _read_response(server_proc)
    assert first["error"]["code"] == -32600


def test_batch_array_gets_invalid_request(server_proc):
    """This server does not support JSON-RPC batch arrays -> -32600."""
    _send_raw(
        server_proc,
        json.dumps([{"jsonrpc": "2.0", "id": 1, "method": "ping"}]),
    )
    _send_raw(server_proc, json.dumps({"jsonrpc": "2.0", "id": 79, "method": "ping"}))

    first = _read_response(server_proc)
    assert first["error"]["code"] == -32600


def test_notification_without_id_gets_no_response(server_proc):
    """JSON-RPC 2.0 forbids responding to notifications. Send one, then a
    ping with an id: the first (and only) response on the pipe must belong
    to the ping."""
    _send_raw(server_proc, json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}))
    _send_raw(server_proc, json.dumps({"jsonrpc": "2.0", "id": 99, "method": "ping"}))

    first = _read_response(server_proc)
    assert first.get("id") == 99
    assert "error" not in first


def test_idless_known_method_is_a_notification_and_gets_no_response(server_proc):
    """The no-response rule applies to EVERY id-less message, including a
    KNOWN method the server would otherwise answer — `notifications/initialized`
    alone doesn't prove the contract, because the server special-cases it.
    An id-less `ping` is a notification per JSON-RPC 2.0: no response may be
    written for it, not even one with id null. The first response on the
    pipe must therefore be the id'd ping's."""
    _send_raw(server_proc, json.dumps({"jsonrpc": "2.0", "method": "ping"}))
    _send_raw(server_proc, json.dumps({"jsonrpc": "2.0", "id": 100, "method": "ping"}))

    first = _read_response(server_proc)
    assert first.get("id") == 100
    assert "error" not in first


def test_handler_exception_maps_to_internal_error(tmp_path):
    """A handler that raises must produce a -32603 response for the request
    (id echoed), with no exception text leaking to the client. Uses
    in-memory streams because the fault has to be injected into the
    handler itself."""

    def _raising_handler(message: dict) -> dict | None:
        raise RuntimeError("secret internal detail that must not leak")

    in_stream = io.StringIO(json.dumps({"jsonrpc": "2.0", "id": 5, "method": "ping"}) + "\n")
    out_stream = io.StringIO()

    protocol.serve(_raising_handler, in_stream, out_stream)

    lines = [line for line in out_stream.getvalue().splitlines() if line]
    assert len(lines) == 1
    response = json.loads(lines[0])
    assert response["id"] == 5
    assert response["error"]["code"] == -32603
    assert "secret internal detail" not in json.dumps(response)


def test_handler_exception_on_notification_writes_nothing(tmp_path):
    """A raising handler on a NOTIFICATION (no id) must still produce no
    response at all — errors on notifications are swallowed per JSON-RPC 2.0."""

    def _raising_handler(message: dict) -> dict | None:
        raise RuntimeError("boom")

    in_stream = io.StringIO(json.dumps({"jsonrpc": "2.0", "method": "whatever"}) + "\n")
    out_stream = io.StringIO()

    protocol.serve(_raising_handler, in_stream, out_stream)

    assert out_stream.getvalue() == ""


def test_tools_call_with_list_params_is_invalid_params_not_internal_error(server_proc):
    """`"params": []` on tools/call is malformed (params must be an object
    with name/arguments). The server must answer with a typed
    invalid-params surface — JSON-RPC error -32602, or its ToolError shape
    (result.isError=true with a specific message) — NEVER the generic
    -32700/-32603 fallback and never a crash."""
    _send_raw(
        server_proc,
        json.dumps({"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": []}),
    )

    response = _read_response(server_proc)
    assert response.get("id") == 7

    error = response.get("error")
    if error is not None:
        assert error["code"] == -32602
        return

    # ToolError surface: a tools/call result marked as an error, carrying a
    # real validation message — not the generic internal-error text.
    result = response["result"]
    assert result["isError"] is True
    text = json.dumps(result)
    assert "internal error" not in text
