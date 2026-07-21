# Changelog

## 0.0.4 (0.0.3 tag exists but was never fully released; the fixes below are the
first that actually land in an installable state — `__version__` had been left
at 0.0.2 across the v0.0.3/v0.0.4 tags, fixed in this release)

- Fixed: unbounded disk growth — structural generation directories
  (`global-graph.json` + overlay + lexical-index, 200-500MB each) had zero GC.
  A scheduled sync accumulates one full generation per run forever. Added
  `publish.prune_old_generations` (keeps last 2, matches the embeddings GC
  convention), also detects and removes generations left dangling by an
  interrupted publish (e.g. an OOM-killed run mid-write).
- Fixed: real OOM kills in production — `build_lexical_index` stored every
  postings entry as a `{"repo","key","field","weight"}` dict (weight is pure
  redundant lookup of a 3-entry constant) and held both the raw accumulation
  structure and the fully-materialized output simultaneously in memory.
  Switched to compact `[repo, key, field]` arrays, dedup-during-accumulation,
  and pop-as-converted instead of holding both copies — measured ~43% peak
  memory reduction and ~18-19% smaller on-disk index on fixture benchmarks.
  Lexical-index schema bumped to v2; the MCP server now rejects a
  stale-shaped (v1) index instead of silently misreading it.
- Hardened: `_write_json_atomic` now fsyncs file data before the rename, so
  an interrupted write can't leave a durable-rename but garbage-content file
  behind an already-flipped `current`.

## 0.0.2

Fixes found and verified via a real production run against a live Ollama backend.

- Fixed: `--skip-labeling` never actually gated the naming stage — it only
  affected a log message and a validation bypass, so `graphify cluster-only`/
  `label` ran regardless of the flag. The skip path now bypasses the naming
  call entirely, guaranteeing zero network calls when passed.
- Fixed: a mid-run network failure (e.g. a DNS blip) during `graphify label`
  or an `/api/embed` batch call raised an uncaught exception and crashed the
  whole pipeline, discarding already-completed work. Both stages now degrade
  gracefully instead:
  - `naming.py`: falls back to `LABELING_DEGRADED`, keeping names already on
    disk rather than crashing.
  - `embedding.py`: new `EMBED_PARTIAL` status — repos already embedded this
    run keep their real vectors; the failed repo and any not-yet-reached
    repos fall back to their previous published shard.
- Fixed: `graphify cluster-only`/`label` can hit their own internal
  shrink-guard (e.g. after a malformed node causes a node-count mismatch),
  print "Done" and exit 0, yet silently write zero `community_name` values.
  `run_naming` now verifies real names actually landed before reporting
  `LABELING_OK`, degrading instead of returning a false success.
- Added: real per-repo and per-batch progress logging for the sync/naming/
  embedding stages (previously silent for the full run duration).
- Regression tests added for all three fixes above, each verified to fail
  against the pre-fix code and pass against the fix.

## 0.0.1

Initial extraction from knowledge-base's in-repo sync engine into a
standalone installable package.
