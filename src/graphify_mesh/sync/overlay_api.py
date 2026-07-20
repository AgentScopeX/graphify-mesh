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
"""
from __future__ import annotations

import re
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


def _iter_source_files(root: Path, suffix: str) -> list[Path]:
    if not root.is_dir():
        return []
    return [
        p
        for p in sorted(root.rglob(f"*{suffix}"))
        if p.is_file() and not any(part in IGNORED_DIR_NAMES for part in p.relative_to(root).parts)
    ]


def extract_symfony_route_providers(repo_id: str, root: Path) -> list[ApiProvider]:
    """Grep-based Symfony `#[Route(...)]` attribute extractor. AST tooling
    is not available at this layer (no embeddings/AST pass over PHP in the
    sync pipeline) — grep is explicitly acceptable per plan WS4 item 3."""
    providers: list[ApiProvider] = []
    for path in _iter_source_files(root, ".php"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = str(path.relative_to(root))
        for match in _ROUTE_ATTR_RE.finditer(text):
            body = match.group(1)
            path_match = _PATH_LITERAL_RE.search(body)
            if not path_match:
                continue
            raw_path = path_match.group(1)
            methods_match = _METHODS_RE.search(body)
            http_methods = _METHOD_TOKEN_RE.findall(methods_match.group(1)) if methods_match else ["GET"]

            # Enclosing method name (qualified_label = Class::method), best
            # effort: nearest preceding `class` + nearest following `function`.
            class_match = None
            for candidate in _CLASS_NAME_RE.finditer(text, 0, match.start()):
                class_match = candidate
            class_name = class_match.group(1) if class_match else "UnknownClass"
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


def extract_consumer_literal_paths(repo_id: str, root: Path) -> list[tuple[str, str, str, str]]:
    """[(repo, http_method_or_GET, normalized_path_template, source_file)]
    for every literal path string found alongside an HTTP-call-shaped
    expression, across both PHP and JS/TS sources (a consumer calling
    another registered repo's API can live in either ecosystem)."""
    candidates: list[tuple[str, str, str, str]] = []
    for suffix in (".php", ".ts", ".tsx", ".js", ".jsx"):
        for path in _iter_source_files(root, suffix):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            rel = str(path.relative_to(root))
            for match in _CONSUMER_CALL_RE.finditer(text):
                raw_path = match.group("path1") or match.group("path2") or match.group("path3")
                if not raw_path:
                    continue
                method = (match.group("method") or "GET").upper()
                candidates.append((repo_id, method, normalize_path_template(raw_path), rel))
    return candidates


def match_provides_consumes_edges(
    providers_by_repo: dict[str, list[ApiProvider]],
    consumer_candidates_by_repo: dict[str, list[tuple[str, str, str, str]]],
) -> list[OverlayEdge]:
    """Emit a matched (provides_api, consumes_api) pair only where a
    consumer's exact (method, normalized path template) resolves to a
    *different* repo's declared provider. No match => zero edges for that
    candidate — never a fuzzy/best-effort guess."""
    provider_index: dict[tuple[str, str], ApiProvider] = {}
    for repo_id, providers in providers_by_repo.items():
        for provider in providers:
            provider_index.setdefault((provider.http_method, provider.path_template), provider)

    edges: list[OverlayEdge] = []
    emitted_pairs: set[tuple] = set()
    for consumer_repo, candidates in consumer_candidates_by_repo.items():
        for method, template, source_file in ((c[1], c[2], c[3]) for c in candidates):
            provider = provider_index.get((method, template))
            if provider is None or provider.repo == consumer_repo:
                continue
            pair_key = (consumer_repo, source_file, provider.repo, provider.qualified_label, method, template)
            if pair_key in emitted_pairs:
                continue
            emitted_pairs.add(pair_key)

            provider_ref = LogicalRef(
                repo=provider.repo, source_file=provider.source_file, qualified_label=provider.qualified_label
            )
            consumer_ref = LogicalRef(
                repo=consumer_repo, source_file=source_file, qualified_label=f"{method} {template}"
            )
            evidence = f"{method} {template} matched: provider {provider.repo}:{provider.source_file}"
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
