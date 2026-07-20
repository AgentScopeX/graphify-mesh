"""`context_pack(goal, scope, token_budget)` tool (WS5 tool 5): evidence
cards built from `retrieval.rank`'s ranked hits, each carrying an explicit
`[repo:path:line]` citation (same shape as everywhere else in this
package), a snippet, and a confidence flag. Truncates strictly by whole-card
boundary — a card is either fully included or fully excluded, it is never
sliced mid-way (plan WS5 tool contract: "budgets").

Confidence flag is always `ranking.CONFIDENCE_EXTRACTED` here:
`context_pack` does not expose an `include_inferred` toggle the way
`rank()` itself does, so every hit this function can ever surface is, by
construction, drawn from the EXTRACTED-only default structural traversal
plus lexical/vector matches — there is no code path by which an
INFERRED-sourced hit reaches this function today. The field is still
carried explicitly (not hardcoded as a bare string) so a future
`include_inferred` passthrough only needs to plumb the real value through,
not invent the field.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from graphify_mesh.sync.embedding import build_snippet

from graphify_mesh.server import ranking
from graphify_mesh.server.retrieval import EmbedQueryFn, Hit, rank
from graphify_mesh.server.scope import RegistryEntry
from graphify_mesh.server.store import Generation

# Rough, dependency-free token estimate — no tokenizer dependency added just
# for a budget heuristic. ~4 chars/token is the commonly-cited English-text
# average; deliberately errs on the side of UNDER-counting cards (budget
# runs out slightly early rather than ever overshooting the caller's stated
# token_budget).
CHARS_PER_TOKEN_ESTIMATE = 4

# How many ranked candidates to consider building cards from before token
# budget truncation. Independent of the caller's k (context_pack has no k
# parameter) — kept generous since truncation is budget-driven, not count-driven.
CONTEXT_PACK_CANDIDATE_K = 30


@dataclass
class EvidenceCard:
    citation: str  # "[repo:path:line]"
    repo: str
    label: str
    source_file: str
    line: int | None
    community_name: str | None
    confidence: str
    snippet: str
    score: float

    def estimated_tokens(self) -> int:
        # citation + label are always present and cheap; snippet dominates
        # the estimate. +1 keeps a zero-length snippet from costing 0 tokens.
        return max(1, (len(self.snippet) + len(self.label) + len(self.citation)) // CHARS_PER_TOKEN_ESTIMATE)


@dataclass
class ContextPackResult:
    goal: str
    cards: list = field(default_factory=list)
    truncated: bool = False
    degraded: list = field(default_factory=list)


def _root_for_repo(repo_id: str, registry_entries: list[RegistryEntry]) -> Path | None:
    for entry in registry_entries:
        if entry.repo_id == repo_id:
            return entry.root
    return None


def _card_from_hit(hit: Hit, generation: Generation, registry_entries: list[RegistryEntry]) -> EvidenceCard:
    node = generation.node_by_id.get(hit.node_id, {})
    line = node.get("line")
    root = _root_for_repo(hit.repo, registry_entries)
    snippet = build_snippet(root, hit.source_file, line) if hit.source_file else ""
    line_part = line if line is not None else "?"
    citation = f"[{hit.repo}:{hit.source_file}:{line_part}]"
    return EvidenceCard(
        citation=citation,
        repo=hit.repo,
        label=hit.label,
        source_file=hit.source_file,
        line=line,
        community_name=hit.community_name,
        confidence=ranking.CONFIDENCE_EXTRACTED,
        snippet=snippet,
        score=hit.score,
    )


def build_context_pack(
    goal: str,
    generation: Generation,
    repo_filter: frozenset[str] | None,
    registry_entries: list[RegistryEntry],
    token_budget: int,
    embed_query_fn: EmbedQueryFn,
) -> ContextPackResult:
    ranked = rank(goal, generation, repo_filter, CONTEXT_PACK_CANDIDATE_K, embed_query_fn)
    cards = [_card_from_hit(h, generation, registry_entries) for h in ranked.hits]

    selected: list[EvidenceCard] = []
    remaining = token_budget
    truncated = False
    for card in cards:
        cost = card.estimated_tokens()
        if cost > remaining:
            truncated = True
            break
        selected.append(card)
        remaining -= cost

    return ContextPackResult(goal=goal, cards=selected, truncated=truncated, degraded=ranked.degraded)
