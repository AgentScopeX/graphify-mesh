"""graphify-mesh-reap — leak-reaper CLI.

Finds `graphify.serve` / `graphify extract` processes whose parent has
already exited (see graphify_mesh/sync/reaper.py for the full orphan
definition) and reports them. Report-only (dry-run) by default; pass
--kill to actually SIGTERM the confirmed orphans. A process with a live
parent (a real editor/agent session or its shell) is NEVER touched, with or
without --kill.

Exit code is always 0 on a successful scan (finding zero or many orphans
is not a failure); a non-zero exit means the scan itself could not run
(e.g. `ps` unavailable).
"""
from __future__ import annotations

import argparse
import json
import sys

from graphify_mesh.sync.reaper import run_reaper


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kill", action="store_true", help="Send SIGTERM to confirmed orphans (default: report only).")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a text table.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    try:
        candidates = run_reaper(kill=args.kill)
    except Exception as exc:  # noqa: BLE001 - surface any ps/scan failure clearly
        print(f"graphify-mesh-reap: scan failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([candidate.__dict__ for candidate in candidates], indent=2))
        return 0

    if not candidates:
        print("graphify-mesh-reap: no orphaned graphify processes found")
        return 0

    action = "KILLED" if args.kill else "would kill (report-only, pass --kill to act)"
    for candidate in candidates:
        print(f"[{action}] pid={candidate.pid} ppid={candidate.ppid} reason={candidate.reason}")
        print(f"    cmd: {candidate.args}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
