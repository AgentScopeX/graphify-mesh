"""Pre-publish validation of the merged global graph (WS1 item 7).

All checks return (ok: bool, errors: list[str]) so the caller can aggregate
everything before deciding to publish or not.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from graphify_mesh.sync.config import FORBIDDEN_OVERLAY_RELATION_TYPES

PLACEHOLDER_PREFIX = "Community "


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)


def validate_schema(data: dict) -> ValidationResult:
    errors = []
    if not isinstance(data.get("nodes"), list):
        errors.append("schema: 'nodes' missing or not a list")
    if not isinstance(data.get("links", data.get("edges")), list):
        errors.append("schema: 'links'/'edges' missing or not a list")
    for node in data.get("nodes", []):
        if not isinstance(node, dict) or "id" not in node:
            errors.append(f"schema: node missing 'id': {node!r}")
            break
    for link in data.get("links", data.get("edges", [])):
        if not isinstance(link, dict) or "source" not in link or "target" not in link:
            errors.append(f"schema: link missing 'source'/'target': {link!r}")
            break
    return ValidationResult(ok=not errors, errors=errors)


def validate_shrink_guard(
    new_counts: tuple[int, int], previous_counts: tuple[int, int] | None, allow_shrink: bool
) -> ValidationResult:
    if previous_counts is None or allow_shrink:
        return ValidationResult(ok=True)
    new_nodes, new_edges = new_counts
    prev_nodes, prev_edges = previous_counts
    errors = []
    if new_nodes < prev_nodes:
        errors.append(f"shrink-guard: new global graph has {new_nodes} nodes, previous published had {prev_nodes}")
    if new_edges < prev_edges:
        errors.append(f"shrink-guard: new global graph has {new_edges} edges, previous published had {prev_edges}")
    return ValidationResult(ok=not errors, errors=errors)


def validate_dangling_ids(data: dict) -> ValidationResult:
    """C20: every edge endpoint must resolve to a node that exists in the merged graph."""
    node_ids = {n["id"] for n in data.get("nodes", []) if isinstance(n, dict) and "id" in n}
    errors = []
    for link in data.get("links", data.get("edges", [])):
        if not isinstance(link, dict):
            continue
        src, dst = link.get("source"), link.get("target")
        if src not in node_ids:
            errors.append(f"dangling-id: edge source {src!r} not in merged node set")
        if dst not in node_ids:
            errors.append(f"dangling-id: edge target {dst!r} not in merged node set")
    return ValidationResult(ok=not errors, errors=errors)


def _repo_prefix(node_id: object) -> str | None:
    """Extract the `<repo_id>` half of a merged `<repo_id>::<local_id>` node
    id, or None if the id doesn't follow that convention (external/bare node)."""
    if not isinstance(node_id, str) or "::" not in node_id:
        return None
    repo_id, _, _ = node_id.partition("::")
    return repo_id


def validate_forbidden_edges(data: dict) -> ValidationResult:
    """The structural merged output must never contain CROSS-REPO overlay-only
    edges (cross_repo:true, or a cross-repo overlay relation type like
    semantically_similar_to) — those belong exclusively in the WS4 overlay
    artifact (C5).

    Same-repo edges sharing one of the overlay's relation-type strings are NOT
    forbidden: upstream `graphify`'s own semantic extraction can legitimately
    emit a same-repo `depends_on` edge (e.g. a Helm `Chart.yaml` subchart
    dependency, a package.json same-repo reference) as normal EXTRACTED data.
    The invariant this guards against is a cross-repo relation leaking into
    structural truth, not the relation-type string appearing at all — so the
    check is scoped to edges whose endpoints resolve to two DIFFERENT repo
    prefixes (or where either endpoint has no repo prefix at all, since that
    can only originate from the overlay's external-node handling, never from
    a same-repo per-project graph).
    """
    errors = []
    for link in data.get("links", data.get("edges", [])):
        if not isinstance(link, dict):
            continue
        if link.get("cross_repo") is True:
            errors.append(f"forbidden-edge: cross_repo:true edge {link.get('source')}->{link.get('target')}")
            continue
        rel = link.get("relation") or link.get("type")
        if rel not in FORBIDDEN_OVERLAY_RELATION_TYPES:
            continue
        src_repo = _repo_prefix(link.get("source"))
        dst_repo = _repo_prefix(link.get("target"))
        if src_repo is not None and src_repo == dst_repo:
            continue  # same-repo edge, legitimate upstream-extracted data
        errors.append(f"forbidden-edge: cross-repo overlay relation {rel!r} on {link.get('source')}->{link.get('target')}")
    return ValidationResult(ok=not errors, errors=errors)


def validate_community_names(data: dict, skip_labeling: bool) -> ValidationResult:
    """Every clustered node must have a non-placeholder community_name.

    Labeling is WS2/WS3 work; WS1 implements the check but allows bypassing
    it via --skip-labeling (logged reason), since no labeling stage runs yet.
    """
    if skip_labeling:
        return ValidationResult(ok=True, errors=["community-name check skipped: --skip-labeling (labeling not wired until WS2)"])
    errors = []
    for node in data.get("nodes", []):
        if not isinstance(node, dict):
            continue
        if node.get("community") is None:
            continue
        name = node.get("community_name")
        if not name or name.startswith(PLACEHOLDER_PREFIX):
            errors.append(f"placeholder community_name on node {node.get('id')!r}: {name!r}")
    return ValidationResult(ok=not errors, errors=errors)


def validate_generation_manifest(manifest: dict) -> ValidationResult:
    """C28: generation manifest internal-consistency check."""
    required_keys = {
        "generation_id",
        "created_at",
        "repo_input_hashes",
        "registry_hash",
        "config_hash",
        "output_node_count",
        "output_edge_count",
        "output_hash",
        "clustering_backend",
        "embedding_model",
        "labeling",
        "stale_repos",
    }
    missing = required_keys - set(manifest.keys())
    errors = [f"generation-manifest: missing key {k!r}" for k in sorted(missing)]
    if not missing and not isinstance(manifest["repo_input_hashes"], dict):
        errors.append("generation-manifest: repo_input_hashes must be an ordered mapping")
    return ValidationResult(ok=not errors, errors=errors)


def run_all(
    data: dict,
    previous_counts: tuple[int, int] | None,
    allow_shrink: bool,
    skip_labeling: bool,
) -> ValidationResult:
    schema = validate_schema(data)
    if not schema.ok:
        # Downstream checks assume well-formed data; stop early.
        return schema

    new_counts = (len(data.get("nodes", [])), len(data.get("links", data.get("edges", []))))
    checks = [
        schema,
        validate_shrink_guard(new_counts, previous_counts, allow_shrink),
        validate_dangling_ids(data),
        validate_forbidden_edges(data),
        validate_community_names(data, skip_labeling),
    ]
    all_errors: list[str] = []
    hard_fail = False
    for check in checks:
        if check is checks[4] and skip_labeling:
            # Community-name check errors are informational-only in skip mode.
            all_errors.extend(check.errors)
            continue
        if not check.ok:
            hard_fail = True
        all_errors.extend(check.errors)
    return ValidationResult(ok=not hard_fail, errors=all_errors)
