from __future__ import annotations

import json

from graphify_mesh.sync import lexical_index
from graphify_mesh.sync.pipeline import run


def test_tokenize_splits_namespace_and_double_colon():
    tokens = lexical_index.tokenize_text(r"App\Service\Foo::barMethod")
    assert "app" in tokens
    assert "service" in tokens
    assert "foo" in tokens
    assert "barmethod" in tokens
    # camelCase subtoken split
    assert "bar" in tokens
    assert "method" in tokens


def test_tokenize_snake_case_subtokens():
    tokens = lexical_index.tokenize_text("get_user_name")
    assert "get_user_name" in tokens
    assert "get" in tokens
    assert "user" in tokens
    assert "name" in tokens


def test_tokenize_plain_word_regex_would_miss_subtokens():
    """Documents the verified insufficiency: graphify's `\\w+` tokenizer
    would return a single token here; ours must split it."""
    import re

    plain = re.findall(r"\w+", "getUserName".lower())
    assert plain == ["getusername"]
    ours = lexical_index.tokenize_text("getUserName")
    assert "get" in ours and "user" in ours and "name" in ours


def test_alias_forms_include_leading_backslash_variant():
    aliases = lexical_index.extract_alias_forms(r"\App\Service\Foo", None)
    assert "\\app\\service\\foo" in aliases
    assert "app\\service\\foo" in aliases


def test_alias_forms_class_method_pair():
    aliases = lexical_index.extract_alias_forms("TimeService::calculate", "src/TimeService.php")
    assert "timeservice::calculate" in aliases
    assert "timeservice" in aliases
    assert "calculate" in aliases
    assert "timeservice.php" in aliases


def test_build_lexical_index_field_boosts_and_determinism():
    graphs_by_repo = {
        "repo.a": {
            "nodes": [
                {"id": "n1", "label": "AlphaClass", "source_file": "src/alpha.py"},
                {"id": "n2", "label": "BetaClass", "source_file": "src/beta.py"},
            ]
        }
    }
    result1 = lexical_index.build_lexical_index(graphs_by_repo, {})
    result2 = lexical_index.build_lexical_index(graphs_by_repo, {})
    assert result1.data == result2.data  # deterministic given identical input
    assert result1.stats.documents == 2

    # schema_version 2: compact `[repo, key, field]` arrays, no per-entry
    # "weight" — weight is derived from `field_boosts` at read time.
    label_postings = result1.data["postings"]["alphaclass"]
    assert isinstance(label_postings[0], list)
    assert len(label_postings[0]) == 3
    assert label_postings[0][2] == "label"
    assert result1.data["field_boosts"]["label"] == lexical_index.FIELD_BOOST_LABEL

    path_postings = result1.data["postings"]["alpha"]
    assert any(p[2] == "path" for p in path_postings)


def test_build_lexical_index_alias_exact_and_node_id_index():
    graphs_by_repo = {
        "repo.a": {
            "nodes": [{"id": "n1", "label": "TimeService", "source_file": "src/TimeService.php"}]
        }
    }
    result = lexical_index.build_lexical_index(graphs_by_repo, {})
    assert "timeservice" in result.data["alias_exact"]
    entry = result.data["alias_exact"]["timeservice"][0]
    # schema_version 2: compact `[repo, key]` array, not `{"repo": ..., "key": ...}`.
    assert isinstance(entry, list)
    assert len(entry) == 2
    assert entry[0] == "repo.a"
    assert "repo.a\x1fn1" in result.data["node_id_index"]


def test_build_lexical_index_no_dict_entries_or_weight_key():
    """Regression guard (production OOM fix): postings/alias_exact entries
    must be compact arrays, and `weight` must never appear anywhere in a
    built index — a future accidental regression back to per-entry dicts
    would reintroduce the peak-RSS/on-disk-size blowup that caused real
    OOM kills of graphify-mesh-sync."""
    graphs_by_repo = {
        "repo.a": {
            "nodes": [
                {"id": "n1", "label": "AlphaClass", "source_file": "src/alpha.py"},
                {"id": "n2", "label": "TimeService", "source_file": "src/TimeService.php"},
            ]
        }
    }
    result = lexical_index.build_lexical_index(graphs_by_repo, {})
    assert result.data["schema_version"] == lexical_index.LEXICAL_SCHEMA_VERSION

    for entries in result.data["postings"].values():
        for entry in entries:
            assert isinstance(entry, list)
            assert len(entry) == 3
            assert "weight" not in entry

    for entries in result.data["alias_exact"].values():
        for entry in entries:
            assert isinstance(entry, list)
            assert len(entry) == 2
            assert "weight" not in entry


def test_pipeline_publishes_lexical_index_artifact(env):
    env.add_repo("acme.a", "acme", "a", "acme-a", graph_fixture="repo_a.json")
    env.write_registry()
    settings = env.settings()

    report = run(settings)

    assert report.published, report.publish_blocked_reason
    assert not report.skipped_stages  # lexical-index no longer skipped
    assert report.lexical_index_stats["documents"] > 0

    lexical_path = settings.global_dir / "current" / lexical_index.LEXICAL_INDEX_FILENAME
    assert lexical_path.is_file()
    data = json.loads(lexical_path.read_text(encoding="utf-8"))
    assert data["tokenizer_version"] == lexical_index.TOKENIZER_VERSION
    assert data["schema_version"] == lexical_index.LEXICAL_SCHEMA_VERSION
    assert data["postings"]

    manifest = json.loads(
        (settings.global_dir / "current" / "generation-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["lexical_index_tokenizer_version"] == lexical_index.TOKENIZER_VERSION
    assert manifest["lexical_index_schema_version"] == lexical_index.LEXICAL_SCHEMA_VERSION
