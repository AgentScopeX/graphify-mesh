"""`provides_api` / `consumes_api` overlay edges.

Extractor survey across a representative multi-repo set:
  - Symfony `#[Route(...)]` PHP attributes are used across the composer-based
    backends (e.g. a controller such as
    src/Controller/Rest/NotificationsController.php —
    `#[Route('/api/v1/notifications/users/{user}/events', name: "...",
    requirements: [...], methods: ['GET'])]`, spanning multiple lines).
  - If no swagger/openapi spec file (`*.yaml`/`*.json` named swagger/openapi)
    is checked out at any registered repo root, a dedicated OpenAPI/swagger
    extractor has no input to exercise, so only the Symfony attribute
    extractor is implemented here. Nelmio `#[OA\\...]` attributes may exist
    alongside `#[Route]` in the same controllers but describe the same routes
    `#[Route]` already captures — not a distinct source of provider identity.

Canonical API identity = `service :: METHOD :: normalized-path-template`
(plan WS4 item 3). Provenance for every extracted-from-code edge is
`EXTRACTED_CONFIG`. Per plan: "only emit a consumes edge where the URL is
actually resolvable to a provides edge — no guessing/fuzzy matching across
repos" — implemented here as an exact string match on the normalized path
template between what a controller declares (provider) and what a literal
string in another repo's source references alongside an HTTP-call-shaped
expression (consumer). An unresolvable consumer produces zero edges, never a
best-effort guess. Because an edge always needs two parties, a provided
route with no matching consumer this run does not itself appear alone —
`provides_api` and `consumes_api` are emitted together, exactly once each,
only for a matched (provider, consumer) pair.

Performance note: provider and consumer extraction share a single
`os.walk` pass per repo (ignored directories pruned in place) and a single
read per source file; the shared result is memoized per (repo_id, root)
behind a stat-based (path, mtime, size) fingerprint so the two public entry
points — called separately by overlay.py — do not re-read the tree, and an
unchanged repo skips re-extraction entirely within a long-lived process.
Gating extraction on the sync pipeline's own per-repo manifest digest is
deferred: no change signal reaches these functions' arguments, and wiring
one through would require editing overlay.py/pipeline.py.
"""

from __future__ import annotations

import os
import re
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

from graphify_mesh.sync.config import IGNORED_DIR_NAMES
from graphify_mesh.sync.overlay_refs import LogicalRef, OverlayEdge

PROVENANCE_EXTRACTED_CONFIG = "EXTRACTED_CONFIG"
CONFIDENCE_API_MATCH = 0.85

_ROUTE_ATTR_RE = re.compile(r"#\[Route\((.*?)\)\]", re.DOTALL)
_PATH_LITERAL_RE = re.compile(r"""^\s*['"](/[^'"]*)['"]""", re.MULTILINE)
_METHODS_RE = re.compile(r"methods\s*:\s*\[([^\]]*)\]")
_METHOD_NAME_RE = re.compile(r"\bfunction\s+(\w+)\s*\(")
_CLASS_NAME_RE = re.compile(r"\bclass\s+(\w+)")
_METHOD_TOKEN_RE = re.compile(r"'([A-Za-z]+)'")

# Consumer-side: an HTTP-client-shaped call followed (within the same
# expression) by a string literal path. Kept intentionally narrow — this is
# not a general-purpose HTTP-call detector, only enough to anchor a literal
# path so it can be checked for an exact match against a known provider.
_CONSUMER_CALL_RE = re.compile(
    r"(?:->request\(\s*['\"](?P<method>[A-Za-z]+)['\"]\s*,\s*['\"](?P<path1>/[^'\"]*)['\"]"
    r"|->(?:get|post|put|patch|delete)\(\s*['\"](?P<path2>/[^'\"]*)['\"]"
    r"|(?:axios|fetch)\s*\(\s*['\"](?P<path3>/[^'\"]*)['\"])",
    re.IGNORECASE,
)

_PROVIDER_SUFFIX = ".php"
# Suffix order is load-bearing: consumer candidates are emitted grouped by
# suffix in this order (each group path-sorted), matching the historical
# per-suffix scan order so downstream edge ordering stays stable.
_SOURCE_SUFFIXES = (".php", ".ts", ".tsx", ".js", ".jsx")

_ConsumerCandidate = tuple[str, str, str, str]

# Most-recent extraction per (repo_id, root), guarded by a stat fingerprint
# of every walked source file. A stale fingerprint always falls back to a
# full re-walk + re-extract, so this can serve stale data only if a file
# changes without its mtime_ns/size changing.
_FACTS_CACHE: dict[tuple[str, str], tuple[tuple, list[ApiProvider], list[_ConsumerCandidate]]] = {}
_FACTS_CACHE_MAX = 64


@dataclass(frozen=True)
class ApiProvider:
    repo: str
    http_method: str
    path_template: str
    source_file: str
    qualified_label: str


def normalize_path_template(path: str) -> str:
    """Canonical form for matching: strip query string, force a single
    leading slash, drop a trailing slash (except root), replace every
    `{paramName}` placeholder with the generic `{param}` so provider and
    consumer only need to agree on shape, not parameter names."""
    path = path.split("?", 1)[0].strip()
    if not path.startswith("/"):
        path = "/" + path
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    return re.sub(r"\{[^}/]+\}", "{param}", path)


def _walk_source_files(root: Path) -> dict[str, list[Path]]:
    """One os.walk pass collecting every suffix needed by provider and
    consumer extraction, pruning ignored directories in place instead of
    filtering each path after a full rglob per suffix."""
    by_suffix: dict[str, list[Path]] = {suffix: [] for suffix in _SOURCE_SUFFIXES}
    if not root.is_dir():
        return by_suffix
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIR_NAMES]
        for filename in filenames:
            for suffix in _SOURCE_SUFFIXES:
                if not filename.endswith(suffix):
                    continue
                path = Path(dirpath) / filename
                if path.is_file():
                    by_suffix[suffix].append(path)
                break
    for paths in by_suffix.values():
        paths.sort()
    return by_suffix


def _stat_fingerprint(by_suffix: dict[str, list[Path]]) -> tuple:
    entries: list[tuple[str, int, int]] = []
    for suffix in _SOURCE_SUFFIXES:
        for path in by_suffix[suffix]:
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append((str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(entries)


def _providers_from_text(repo_id: str, rel: str, text: str) -> list[ApiProvider]:
    providers: list[ApiProvider] = []
    # Class-declaration positions/names precomputed once per file; each
    # route then finds its enclosing class via bisect instead of re-scanning
    # the whole file prefix per route match.
    class_ends: list[int] = []
    class_names: list[str] = []
    for class_match in _CLASS_NAME_RE.finditer(text):
        class_ends.append(class_match.end())
        class_names.append(class_match.group(1))

    for match in _ROUTE_ATTR_RE.finditer(text):
        body = match.group(1)
        path_match = _PATH_LITERAL_RE.search(body)
        if not path_match:
            continue
        raw_path = path_match.group(1)
        methods_match = _METHODS_RE.search(body)
        http_methods = (
            _METHOD_TOKEN_RE.findall(methods_match.group(1)) if methods_match else ["GET"]
        )

        # Enclosing method name (qualified_label = Class::method), best
        # effort: nearest preceding `class` + nearest following `function`.
        class_idx = bisect_right(class_ends, match.start()) - 1
        class_name = "UnknownClass"
        if class_idx >= 0:
            class_name = class_names[class_idx]
        method_match = _METHOD_NAME_RE.search(text, match.end())
        method_name = method_match.group(1) if method_match else "unknownMethod"
        qualified_label = f"{class_name}::{method_name}"

        template = normalize_path_template(raw_path)
        for http_method in http_methods or ["GET"]:
            providers.append(
                ApiProvider(
                    repo=repo_id,
                    http_method=http_method.upper(),
                    path_template=template,
                    source_file=rel,
                    qualified_label=qualified_label,
                )
            )
    return providers


def _consumers_from_text(repo_id: str, rel: str, text: str) -> list[_ConsumerCandidate]:
    candidates: list[_ConsumerCandidate] = []
    for match in _CONSUMER_CALL_RE.finditer(text):
        raw_path = match.group("path1") or match.group("path2") or match.group("path3")
        if not raw_path:
            continue
        method = (match.group("method") or "GET").upper()
        candidates.append((repo_id, method, normalize_path_template(raw_path), rel))
    return candidates


def _extract_repo_api_facts(
    repo_id: str, root: Path
) -> tuple[list[ApiProvider], list[_ConsumerCandidate]]:
    """Walk the repo once, read each source file at most once, and run both
    provider and consumer extraction over the shared text. Memoized behind a
    stat fingerprint so the second public entry point (and unchanged repos on
    later runs in the same process) skip the reads entirely."""
    by_suffix = _walk_source_files(root)
    fingerprint = _stat_fingerprint(by_suffix)
    cache_key = (repo_id, str(root))
    cached = _FACTS_CACHE.get(cache_key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1], cached[2]

    providers: list[ApiProvider] = []
    candidates: list[_ConsumerCandidate] = []
    for suffix in _SOURCE_SUFFIXES:
        for path in by_suffix[suffix]:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            rel = str(path.relative_to(root))
            if suffix == _PROVIDER_SUFFIX:
                providers.extend(_providers_from_text(repo_id, rel, text))
            candidates.extend(_consumers_from_text(repo_id, rel, text))

    _FACTS_CACHE[cache_key] = (fingerprint, providers, candidates)
    while len(_FACTS_CACHE) > _FACTS_CACHE_MAX:
        _FACTS_CACHE.pop(next(iter(_FACTS_CACHE)))
    return providers, candidates


def extract_symfony_route_providers(repo_id: str, root: Path) -> list[ApiProvider]:
    """Grep-based Symfony `#[Route(...)]` attribute extractor. AST tooling
    is not available at this layer (no embeddings/AST pass over PHP in the
    sync pipeline) — grep is explicitly acceptable per plan WS4 item 3."""
    providers, _ = _extract_repo_api_facts(repo_id, root)
    return list(providers)


def extract_consumer_literal_paths(repo_id: str, root: Path) -> list[_ConsumerCandidate]:
    """[(repo, http_method_or_GET, normalized_path_template, source_file)]
    for every literal path string found alongside an HTTP-call-shaped
    expression, across both PHP and JS/TS sources (a consumer calling
    another registered repo's API can live in either ecosystem)."""
    _, candidates = _extract_repo_api_facts(repo_id, root)
    return list(candidates)


def match_provides_consumes_edges(
    providers_by_repo: dict[str, list[ApiProvider]],
    consumer_candidates_by_repo: dict[str, list[_ConsumerCandidate]],
) -> list[OverlayEdge]:
    """Emit a matched (provides_api, consumes_api) pair only where a
    consumer's exact (method, normalized path template) resolves to a
    *different* repo's declared provider. No match => zero edges for that
    candidate — never a fuzzy/best-effort guess."""
    provider_index: dict[tuple[str, str], ApiProvider] = {}
    for providers in providers_by_repo.values():
        for provider in providers:
            provider_index.setdefault((provider.http_method, provider.path_template), provider)

    edges: list[OverlayEdge] = []
    emitted_pairs: set[tuple] = set()
    for consumer_repo, candidates in consumer_candidates_by_repo.items():
        for method, template, source_file in ((c[1], c[2], c[3]) for c in candidates):
            matched = provider_index.get((method, template))
            if matched is None or matched.repo == consumer_repo:
                continue
            pair_key = (
                consumer_repo,
                source_file,
                matched.repo,
                matched.qualified_label,
                method,
                template,
            )
            if pair_key in emitted_pairs:
                continue
            emitted_pairs.add(pair_key)

            provider_ref = LogicalRef(
                repo=matched.repo,
                source_file=matched.source_file,
                qualified_label=matched.qualified_label,
            )
            consumer_ref = LogicalRef(
                repo=consumer_repo, source_file=source_file, qualified_label=f"{method} {template}"
            )
            evidence = f"{method} {template} matched: provider {matched.repo}:{matched.source_file}"
            edges.append(
                OverlayEdge(
                    type="provides_api",
                    source=provider_ref,
                    target=consumer_ref,
                    provenance=PROVENANCE_EXTRACTED_CONFIG,
                    confidence=CONFIDENCE_API_MATCH,
                    evidence=evidence,
                )
            )
            edges.append(
                OverlayEdge(
                    type="consumes_api",
                    source=consumer_ref,
                    target=provider_ref,
                    provenance=PROVENANCE_EXTRACTED_CONFIG,
                    confidence=CONFIDENCE_API_MATCH,
                    evidence=evidence,
                )
            )
    return edges
