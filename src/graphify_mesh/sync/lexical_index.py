"""WS1.6 / WS5 lexical index: a bundle artifact built ONCE per generation by
the sync pipeline (not per MCP-session), consumed read-only by the `graphify-mesh`
companion MCP server (WS5).

Runs as a new pipeline stage AFTER overlay-resolve and BEFORE validate/publish
(plan WS1 item 6 order: `... -> overlay resolve -> lexical index -> validate
-> atomic publish`). Slots into the previously-unwired gap noted in
pipeline.py's `RunReport.skipped_stages`.

Why this exists instead of reusing graphify's own query-time tokenizer
(serve.py): verified against the installed graphify 0.9.20 package
(`graphify/serve.py:86` and `:180`) — its tokenizer is a single
`re.findall(r"\\w+", text.lower())`. That is Unicode-word-character-only: it
never splits `Namespace\\Class::method` into (`Namespace`, `Class`, `method`),
never splits camelCase/PascalCase/snake_case identifiers into subtokens, and
treats a leading-backslash FQCN (`\\App\\Service\\Foo`) and its
non-backslash-leading form (`App\\Service\\Foo`) as different token streams
(the leading `\\` is simply dropped as non-word, but so is every other `\\`,
so `\\Foo\\Bar` and `Foo\\Bar` both tokenize to `["foo","bar"]` today — the
insufficiency the plan calls out is that this happens to work by accident for
the *backslash* case but not for the camelCase/snake_case/`Class::method`
pairing case, which `\\w+` cannot address at all since `:` is non-word and
`getUserName`/`get_user_name` are each a single `\\w+` match with no subtoken
split). This module fixes the subtoken gap and makes the alias/exact-id path
an explicit O(1) table instead of substring scanning.

Normalization reuse (per plan instruction: "don't reinvent label
normalization from scratch if it already exists"): the base
lowercase+diacritic-stripping step reuses graphify's own exported
`norm_label`-equivalent, `graphify.export._strip_diacritics`, rather than
rewriting Unicode normalization here.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from graphify_mesh.sync.embedding import build_snippet, node_key

try:
    from graphify.export import _strip_diacritics as _graphify_strip_diacritics
except ImportError:  # pragma: no cover - defensive: graphify package shape changed upstream

    def _graphify_strip_diacritics(text: str | None) -> str:
        if not text:
            return ""
        return "".join(
            c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
        )


LEXICAL_INDEX_FILENAME = "lexical-index.json"

# Bumped whenever tokenization/alias-extraction rules change in a way that
# would make an old index's postings/aliases inconsistent with a fresh build.
# The graphify-mesh MCP server (WS5) reads this and can refuse a stale-shaped
# index rather than silently misinterpreting it.
TOKENIZER_VERSION = "gm-lex-v1"

# Field boosts (label > path > snippet), per plan WS5 lexical-index bullet.
# Named constants, not magic numbers, per project style rule.
FIELD_BOOST_LABEL = 3.0
FIELD_BOOST_PATH = 1.5
FIELD_BOOST_SNIPPET = 1.0
FIELD_BOOSTS = {
    "label": FIELD_BOOST_LABEL,
    "path": FIELD_BOOST_PATH,
    "snippet": FIELD_BOOST_SNIPPET,
}

# camelCase/PascalCase subtoken boundary: before an uppercase letter that
# follows a lowercase letter/digit, or before the last uppercase letter of a
# run that is followed by a lowercase letter (handles acronym runs like
# "XMLHttpRequest" -> XML, Http, Request).
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
# Splits on any run of non-alphanumeric characters (namespace separators
# `\` and `::`, path separators `/`, dots, dashes, whitespace, etc.) — this is
# the PHP/JS-aware substitute for graphify's plain `\w+` match: segments
# produced here are then further split on camelCase/snake_case boundaries.
# Segment split keeps `_` attached (it is itself a subtoken boundary handled
# separately below) so namespace/path/punctuation separators (`\`, `::`,
# `/`, `.`, `-`, whitespace) delimit segments without prematurely destroying
# snake_case boundaries.
_SEGMENT_SPLIT = re.compile(r"[^A-Za-z0-9_]+")


def _base_normalize(text: str) -> str:
    """Diacritic-stripped, case-PRESERVING. Callers that need the final
    lowercase search/alias form should use `_norm_lower` instead — case must
    survive until after camelCase-boundary detection (see `tokenize_text`)."""
    return _graphify_strip_diacritics(text or "")


def _norm_lower(text: str) -> str:
    return _base_normalize(text).lower()


def normalize_alias_query(text: str) -> str:
    """Public wrapper so callers outside this module (the graphify-mesh MCP
    server's exact-alias bypass, WS5) can normalize a raw query string using
    the IDENTICAL normalization `extract_alias_forms` used to build the
    `alias_exact` table, without duplicating the diacritic-stripping/case
    rules here a second time."""
    return _norm_lower(text)


def _camel_split(segment: str) -> list[str]:
    if not segment:
        return []
    return [p for p in _CAMEL_BOUNDARY.split(segment) if p]


def tokenize_text(text: str | None) -> list[str]:
    """PHP/JS-aware tokenizer (verified insufficiency of graphify's plain
    `\\w+` regex above). Produces, for each raw non-alnum-delimited segment:
    the segment itself (lowercased) AND its camelCase/snake_case subtokens.
    Namespace segments (`App`, `Service`, `Foo` from `App\\Service\\Foo`) and
    `Class::method` pairs (`Class`, `method` from `Class::method`) fall out
    naturally from splitting on non-alphanumeric runs; leading-backslash FQCN
    variants are handled by `extract_alias_forms` below (this function only
    produces search tokens, not exact-alias keys).

    Case/underscore splitting must happen BEFORE lowercasing — lowercasing
    first would destroy the camelCase boundary signal entirely."""
    if not text:
        return []
    diacritics_stripped = _base_normalize(text)
    tokens: list[str] = []
    for raw_segment in _SEGMENT_SPLIT.split(diacritics_stripped):
        if not raw_segment:
            continue
        tokens.append(raw_segment.lower())
        underscore_parts = [p for p in raw_segment.split("_") if p]
        for part in underscore_parts:
            tokens.append(part.lower())
            camel_parts = _camel_split(part)
            if len(camel_parts) > 1:
                tokens.extend(p.lower() for p in camel_parts)
    # De-dup while preserving nothing order-sensitive (postings sort later);
    # a set is fine and keeps doc-frequency accounting simple (one increment
    # per distinct term per document, not per raw occurrence).
    return sorted(set(tokens))


def extract_alias_forms(label: str, source_file: str | None) -> set[str]:
    """Exact-alias forms for O(1) lookup: the label as-is (normalized), the
    label with a leading namespace-separator stripped (leading-backslash
    variant per plan bullet), the bare method name of a `Class::method` pair,
    the bare class name of the same pair, and the file basename."""
    aliases: set[str] = set()
    if label:
        norm = _norm_lower(label)
        aliases.add(norm)
        aliases.add(norm.lstrip("\\").lstrip("/"))
        if "::" in label:
            cls, _, method = label.partition("::")
            if cls:
                aliases.add(_norm_lower(cls))
            if method:
                aliases.add(_norm_lower(method))
        if "\\" in label:
            aliases.add(_norm_lower(label.split("\\")[-1]))
    if source_file:
        aliases.add(_norm_lower(Path(source_file).name))
    aliases.discard("")
    return aliases


@dataclass
class LexicalIndexStats:
    documents: int = 0
    terms: int = 0
    aliases: int = 0

    def to_dict(self) -> dict:
        return {"documents": self.documents, "terms": self.terms, "aliases": self.aliases}


@dataclass
class LexicalIndexResult:
    data: dict = field(default_factory=dict)
    stats: LexicalIndexStats = field(default_factory=LexicalIndexStats)


def build_lexical_index(
    graphs_by_repo: dict[str, dict],
    repo_roots_by_id: dict[str, Path],
) -> LexicalIndexResult:
    """Builds the full lexical-index bundle artifact for one generation.

    Deterministic by construction: repos are iterated in sorted order, every
    postings list and alias list is sorted before being placed in the output
    dict, and doc-frequency counters are plain per-term integers — nothing
    here depends on dict iteration order or wall-clock time.
    """
    postings: dict[str, list[tuple[str, str, str]]] = {}
    alias_exact: dict[str, list[tuple[str, str]]] = {}
    node_id_index: dict[str, str] = {}
    doc_freq_global: dict[str, int] = {}
    doc_freq_per_repo: dict[str, dict[str, int]] = {}
    documents = 0

    for repo_id in sorted(graphs_by_repo.keys()):
        graph_data = graphs_by_repo[repo_id]
        source_root = repo_roots_by_id.get(repo_id)
        repo_df = doc_freq_per_repo.setdefault(repo_id, {})

        for node in graph_data.get("nodes", []):
            if not isinstance(node, dict):
                continue
            key = node_key(repo_id, node)
            if key is None:
                continue
            label = node.get("label") or ""
            source_file = node.get("source_file") or ""
            snippet = build_snippet(source_root, source_file, node.get("line"))

            field_tokens = {
                "label": set(tokenize_text(label)),
                "path": set(tokenize_text(source_file)),
                "snippet": set(tokenize_text(snippet)) if snippet else set(),
            }

            documents += 1
            all_terms_this_doc: set[str] = set()
            for field_name, terms in field_tokens.items():
                for term in terms:
                    postings.setdefault(term, []).append((repo_id, key, field_name))
                    all_terms_this_doc.add(term)
            for term in all_terms_this_doc:
                doc_freq_global[term] = doc_freq_global.get(term, 0) + 1
                repo_df[term] = repo_df.get(term, 0) + 1

            for alias in extract_alias_forms(label, source_file):
                alias_exact.setdefault(alias, []).append((repo_id, key))

            raw_id = node.get("id")
            if raw_id:
                node_id_index[f"{repo_id}\x1f{str(raw_id).lower()}"] = key

    sorted_postings = {
        term: [
            {"repo": r, "key": k, "field": f, "weight": FIELD_BOOSTS[f]}
            for r, k, f in sorted(set(entries), key=lambda e: (e[0], e[1], e[2]))
        ]
        for term, entries in sorted(postings.items())
    }
    sorted_aliases = {
        alias: [{"repo": r, "key": k} for r, k in sorted(set(entries))]
        for alias, entries in sorted(alias_exact.items())
    }

    data = {
        "schema_version": 1,
        "tokenizer_version": TOKENIZER_VERSION,
        "field_boosts": FIELD_BOOSTS,
        "postings": sorted_postings,
        "doc_freq": {
            "global": dict(sorted(doc_freq_global.items())),
            "per_repo": {
                rid: dict(sorted(df.items())) for rid, df in sorted(doc_freq_per_repo.items())
            },
        },
        "alias_exact": sorted_aliases,
        "node_id_index": dict(sorted(node_id_index.items())),
        "document_count": documents,
    }
    stats = LexicalIndexStats(
        documents=documents, terms=len(sorted_postings), aliases=len(sorted_aliases)
    )
    return LexicalIndexResult(data=data, stats=stats)
