# Changelog

## Unreleased

### Performance

- Changed: all large published artifacts (`global-graph.json`,
  `cross-project-overlay.json`, `lexical-index.json`) are written via streamed
  `JSONEncoder.iterencode` with compact separators instead of a whole-artifact
  `json.dumps` string — no full-artifact string in RAM and smaller files on
  disk; only `generation-manifest.json` keeps pretty-printing. `output_hash`
  streams the same encoding into the hasher (digest byte-identical to the
  previous implementation).
- Added: shared bounded source-line cache (`sync/source_cache.py`), keyed on
  path + mtime + size, used by every snippet builder — each source file is
  read at most once per pipeline run instead of once per node per stage.
- Changed: source-tree scans (`compute_source_manifest`, overlay API
  extraction) use a single pruned `os.walk` per repo instead of `rglob`
  passes that materialized ignored trees (`node_modules`, `.git`, vendor);
  manifest digests are byte-identical for unchanged trees. Overlay API
  extraction also shares file text between provider/consumer passes and
  finds enclosing classes via a precomputed position list + bisect instead
  of rescanning the file prefix per route.
- Changed: per-repo `graph.json` files are read and hashed once
  (hash + parse + node/edge counts from one buffer); publish reuses hashes
  computed during the per-project stage instead of re-reading every graph.
- Changed: server vector search takes per-repo `np.argpartition` shortlists
  instead of a full Python sort over every embedded node per query; MMR
  selection tracks running max-similarity incrementally (output verified
  identical); `find_similar` and its exact-label fallback use lazy
  per-generation indexes instead of full overlay/node scans per call;
  `context_pack` builds evidence cards lazily inside the token-budget loop
  (no snippet I/O past the cutoff); registry entries are cached keyed on
  mtime + size + inode.

### Memory

- Changed: `VectorShards.normalized()` no longer materializes a float32 copy
  of every mmap'd shard — the matrix stays memory-mapped and only per-row
  norms are cached (the copy previously multiplied across one server process
  per client session).
- Changed: the pipeline drops the pre-naming merged graph and the per-repo
  graph map as soon as their last consumers finish, instead of holding
  multiple whole-graph copies through embed/overlay/lexical; degraded naming
  restore mutates the graph in place instead of copying every node while the
  previous generation is also resident.

### Reliability and correctness

- Added: `generation-manifest.json` now records `artifact_sha256` (sha256 of
  each artifact's raw bytes, hashed as written) and `sync_started_at`. The
  server verifies artifacts by hashing raw bytes in one pass — an artifact
  present on disk without a hash entry is a consistency error — and falls
  back to the legacy canonical hash only for older generations. The manifest
  is written last, so an interrupted publish is always detectably incomplete.
- Fixed: the server read artifacts through the live `current` symlink, so a
  publish flipping mid-load could mix files from two generations; all
  artifacts are now read from the pinned resolved path with generation-id
  cross-checks for the overlay and embeddings (mismatched embeddings drop
  the vector channel with a degraded marker instead of serving wrong data).
- Fixed: store-level degraded markers (e.g. serving the previous generation
  after a rejected reload) were never surfaced; every tool response's
  `degraded` list now includes them.
- Fixed: JSON-RPC protocol gaps — unparseable input now gets `-32700`,
  non-object/batch messages `-32600`, mistyped `params`/tool arguments
  `-32602`/typed tool errors instead of generic internal errors, and
  notifications (including id-less known methods) never receive a response.
- Fixed: embedding batch failures retry once with backoff before degrading,
  and the fallback reuses the already-loaded previous shard; `--dry-run` no
  longer performs live embedding calls.
- Fixed: `prune_old_generations` completeness is manifest-aware — it removes
  generations that a crash left without declared artifacts, but no longer
  deletes legacy-shaped generations (graph + manifest only) that are valid
  rollback targets.
- Fixed: context-pack snippets are read from the live working tree; cards
  now carry `snippet_source`/`snippet_stale` provenance (staleness measured
  against `sync_started_at`) instead of presenting possibly-shifted lines as
  generation data.
- Fixed: validation error lists are capped (50 + summary line) so a broken
  merge cannot write millions of strings into `status.json`; a dead
  positional-identity branch in validation and a dead conditional in the
  state-advancement path were removed.

### Tooling and tests

- Added: tests for the transaction lock (cross-process contention, release on
  exception, CLI exit code 3), crash-mid-publish atomicity (before and inside
  the symlink flip), and the JSON-RPC error contract.
- Changed: CI uses pip caching, cancels superseded runs, measures coverage,
  and enforces `ruff format`; the release workflow no longer installs the
  upstream dependency from an unpinned git HEAD and no longer force-pushes
  over `gh-pages` history.
- Added: `dev` extra (`test` + `lint`); shared test fixtures consolidated
  under `tests/fixtures/`.

## 0.0.4

- Fixed: `__version__` had been left at 0.0.2 across the v0.0.3/v0.0.4 tags;
  corrected in this release.
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
