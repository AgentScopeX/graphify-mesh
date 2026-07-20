"""Newline-delimited JSON-RPC 2.0 stdio transport (WS5).

The MCP stdio transport is exactly one complete JSON-RPC 2.0 message per
line on stdin/stdout — no Content-Length framing (unlike LSP), no
multi-line messages. Stdlib `json` + `sys` implement this completely; see
`graphify_mesh.server/__init__.py` for the full dependency-decision rationale
(no `mcp` SDK package, no third-party transport library).

Exits cleanly on stdin EOF (WS6: "companion server must exit cleanly on
stdin close" — the observed 10-stale-`graphify.serve`-process leak,
~1.6GB RSS, was from sessions that did NOT do this). `serve()` is a plain
`for line in stream` loop: it returns (never raises, never hangs) the
moment the client closes its end of the pipe.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable, Iterator

log = logging.getLogger("graphify_mesh.server.protocol")

# Hard per-message size cap for the stdio transport. Without a bound,
# `stream.readline()`/`for line in stream` buffers an entire line in memory —
# a single giant (or newline-less) line from a misbehaving client OOMs the
# process. 10 MB is far above any legitimate JSON-RPC message this server
# handles; longer lines are drained and skipped, never buffered whole.
MAX_LINE_BYTES = 10 * 1024 * 1024

# `handler(message) -> response-dict | None`. `None` means "this message was
# a JSON-RPC notification (no `id`), so JSON-RPC 2.0 forbids a response" —
# never write a response for those.
JsonRpcHandler = Callable[[dict], dict | None]


def _drain_line(stream) -> None:
    """Consume (and discard) the remainder of an oversized line in bounded
    chunks, stopping at the next newline or EOF. Never accumulates the data."""
    while True:
        chunk = stream.readline(MAX_LINE_BYTES + 1)
        if not chunk:
            return
        if chunk.endswith("\n"):
            return


def read_messages(stream=None) -> Iterator[dict]:
    stream = stream if stream is not None else sys.stdin
    while True:
        raw_line = stream.readline(MAX_LINE_BYTES + 1)
        if not raw_line:
            return  # EOF: clean exit (WS6 contract)
        if len(raw_line) > MAX_LINE_BYTES and not raw_line.endswith("\n"):
            # Oversized line: drain the rest without buffering it, skip the
            # message, keep serving subsequent messages.
            _drain_line(stream)
            log.warning("dropped oversized stdin line (> %d bytes)", MAX_LINE_BYTES)
            continue
        line = raw_line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(message, dict):
            yield message


def write_message(message: dict, stream=None) -> None:
    stream = stream if stream is not None else sys.stdout
    stream.write(json.dumps(message) + "\n")
    stream.flush()


def error_response(request_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def result_response(request_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def serve(handler: JsonRpcHandler, in_stream=None, out_stream=None) -> None:
    """Blocking dispatch loop. Returns (does not raise) the instant
    `in_stream` hits EOF — the only clean-exit contract this transport
    needs (WS6)."""
    for message in read_messages(in_stream):
        try:
            response = handler(message)
        except Exception:
            # Full traceback goes to stderr only; the client gets a generic
            # JSON-RPC internal error — never exception text, paths, or
            # stack frames. Notifications (no "id") get no response at all.
            log.exception("handler raised while processing message")
            if "id" in message:
                write_message(
                    error_response(message.get("id"), -32603, "internal error"), out_stream
                )
            continue
        if response is not None:
            write_message(response, out_stream)
