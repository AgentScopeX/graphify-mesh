# Changelog

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
