"""graphify-mesh-sync — sync-pipeline entrypoint.

Discovers all registered/discoverable per-project graphify collections under
the configured scan root, decides update vs extract per project, rebuilds the
global graph from empty via `graphify merge-graphs` (never `graphify global
add` — see graphify_mesh/sync/__init__.py for the merge-semantics evidence),
validates the result, and atomically publishes a new generation.

Intended to be invoked by a scheduler (e.g. systemd Type=oneshot on a timer;
see examples/systemd/graphify-mesh-sync.{service,timer}) — this is a single
run (`--once` is the only supported mode; there is no daemon loop).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from graphify_mesh.sync.config import Settings
from graphify_mesh.sync.locking import LockHeldError
from graphify_mesh.sync.pipeline import run


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print every action; write nothing outside a private staging dir.")
    parser.add_argument("--once", action="store_true", help="Single run (the only supported mode; no daemon loop).")
    parser.add_argument("--mesh-root", type=Path, default=None, help="Override the graph-mesh repo root (testing).")
    parser.add_argument("--scan-root", type=Path, default=None, help="Override the scan root (testing).")
    parser.add_argument("--registry", type=Path, default=None, help="Override the registry.json path (testing).")
    parser.add_argument("--skip-labeling", action="store_true", default=True, help="Skip the non-placeholder community_name check. Default: on.")
    parser.add_argument("--no-skip-labeling", dest="skip_labeling", action="store_false", help="Enforce the community_name check.")
    parser.add_argument("--skip-embedding", action="store_true", default=True, help="Log-skip the embedding stage. Default: on.")
    parser.add_argument("--allow-shrink", action="store_true", help="Explicitly authorize a smaller published graph than the previous generation.")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = Settings.from_env(
        mesh_root=args.mesh_root,
        scan_root=args.scan_root,
        registry_path=args.registry,
        dry_run=args.dry_run,
        skip_labeling=args.skip_labeling,
        skip_embedding=args.skip_embedding,
        allow_shrink=args.allow_shrink,
    )

    try:
        report = run(settings)
    except LockHeldError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    print(json.dumps(
        {
            "dry_run": report.dry_run,
            "reconciliation": report.reconciliation,
            "project_actions": report.project_actions,
            "stale_repos": report.stale_repos,
            "dirty_repos": report.dirty_repos,
            "merge_ok": report.merge_ok,
            "merge_error": report.merge_error,
            "validation_ok": report.validation_ok,
            "validation_errors": report.validation_errors,
            "published": report.published,
            "publish_blocked_reason": report.publish_blocked_reason,
            "generation_id": report.generation_id,
            "skipped_stages": report.skipped_stages,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
