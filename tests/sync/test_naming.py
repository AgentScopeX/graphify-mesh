from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

from graphify_mesh.sync import backend, graphify_cli, naming, publish
from graphify_mesh.sync.config import Settings
from graphify_mesh.sync.pipeline import run

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FAKE_GRAPHIFY = FIXTURES_DIR / "fake_graphify" / "graphify"


def _merged_graph() -> dict:
    return {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": [
            {
                "id": "example-org.styleguide::a1",
                "label": "AlphaClass",
                "repo_tag": "example-org.styleguide",
                "community": 0,
                "community_name": "Alpha Domain",
            },
            {
                "id": "example-org.styleguide::a2",
                "label": "alpha_helper",
                "repo_tag": "example-org.styleguide",
                "community": 0,
                "community_name": "Alpha Domain",
            },
            {
                "id": "example-org.services::b1",
                "label": "BetaService",
                "repo_tag": "example-org.services",
                "community": 0,
                "community_name": "Beta Domain",
            },
            {
                "id": "example-org.services::b2",
                "label": "beta_util",
                "repo_tag": "example-org.services",
                "community": 0,
                "community_name": "Beta Domain",
            },
        ],
        "links": [
            {
                "source": "example-org.styleguide::a1",
                "target": "example-org.styleguide::a2",
                "relation": "calls",
            },
            {
                "source": "example-org.services::b1",
                "target": "example-org.services::b2",
                "relation": "calls",
            },
        ],
    }


def _settings(tmp_path: Path, **overrides) -> Settings:
    mesh_root = tmp_path / "mesh"
    return Settings.from_env(
        mesh_root=mesh_root,
        scan_root=tmp_path / "www",
        registry_path=mesh_root / "bin" / "registry.json",
        graphify_bin=str(FAKE_GRAPHIFY),
        **overrides,
    )


# ---------------------------------------------------------------------------
# strip_project_community_attrs
# ---------------------------------------------------------------------------


def test_strip_project_community_attrs_removes_both_keys():
    graph = _merged_graph()
    stripped = naming.strip_project_community_attrs(graph)
    for node in stripped["nodes"]:
        assert "community" not in node
        assert "community_name" not in node
    # in-place contract: same object returned, original graph is mutated too
    assert stripped is graph
    assert "community_name" not in graph["nodes"][0]


def test_strip_mutates_in_place_and_returns_same_object():
    graph = {"nodes": [{"id": "n1", "community": 3, "community_name": "X", "label": "L"}]}
    result = naming.strip_project_community_attrs(graph)
    assert result is graph
    assert "community" not in graph["nodes"][0]
    assert "community_name" not in graph["nodes"][0]
    assert graph["nodes"][0]["label"] == "L"


def test_strip_skips_non_dict_nodes():
    graph = {"nodes": ["junk", {"id": "n1", "community": 1}]}
    naming.strip_project_community_attrs(graph)
    assert graph["nodes"][0] == "junk"
    assert "community" not in graph["nodes"][1]


# ---------------------------------------------------------------------------
# run_naming unit tests (bypassing pipeline.run for direct control)
# ---------------------------------------------------------------------------


def test_degraded_mode_skips_cluster_only_and_label_entirely(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_GRAPHIFY_CONTROL", str(tmp_path / "control.json"))
    call_log = tmp_path / "call-log.jsonl"
    monkeypatch.setenv("FAKE_GRAPHIFY_CALL_LOG", str(call_log))

    settings = _settings(tmp_path)
    stripped = naming.strip_project_community_attrs(_merged_graph())

    result = naming.run_naming(
        str(FAKE_GRAPHIFY),
        tmp_path / "naming",
        tmp_path / "naming-home",
        stripped,
        settings,
        health_check=lambda *a, **kw: False,
    )

    assert result.labeling == naming.LABELING_DEGRADED
    assert result.graph_data == stripped
    assert not call_log.exists()  # no cluster-only/label invocation happened at all


def test_cluster_only_silent_no_op_success_is_detected_and_degrades(tmp_path, monkeypatch):
    """Regression test for a REAL observed upstream failure mode: `graphify
    cluster-only` can hit its own internal shrink-guard (e.g. a malformed
    node causes a node-count mismatch against the input), print
    "Done - N communities" and exit 0, yet silently write ZERO
    community/community_name onto any node. Trusting cluster_result.ok alone
    would report LABELING_OK for a generation carrying no real names at
    all — run_naming must detect this and degrade instead."""
    monkeypatch.setenv("FAKE_GRAPHIFY_CONTROL", str(tmp_path / "control.json"))
    call_log = tmp_path / "call-log.jsonl"
    monkeypatch.setenv("FAKE_GRAPHIFY_CALL_LOG", str(call_log))

    naming_dir = tmp_path / "naming"
    out_dir = naming_dir / "graphify-out"
    (tmp_path / "control.json").write_text(
        json.dumps({str(out_dir): {"mode": "silent_success_no_names"}}), encoding="utf-8"
    )

    settings = _settings(tmp_path)
    stripped = naming.strip_project_community_attrs(_merged_graph())

    result = naming.run_naming(
        str(FAKE_GRAPHIFY),
        naming_dir,
        tmp_path / "naming-home",
        stripped,
        settings,
        health_check=lambda *a, **kw: True,
    )

    assert result.labeling == naming.LABELING_DEGRADED
    assert "no community_name" in result.reason
    # The graph is NOT silently swapped for something that looks named —
    # it falls back to the stripped (pre-naming) input, honest about having
    # no real names this generation.
    for node in result.graph_data["nodes"]:
        assert "community_name" not in node


def test_sig_unchanged_community_not_sent_to_label_on_second_run(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_GRAPHIFY_CONTROL", str(tmp_path / "control.json"))
    call_log = tmp_path / "call-log.jsonl"
    monkeypatch.setenv("FAKE_GRAPHIFY_CALL_LOG", str(call_log))

    settings = _settings(tmp_path)
    naming_dir = tmp_path / "naming"
    staging_home = tmp_path / "naming-home"
    stripped = naming.strip_project_community_attrs(_merged_graph())

    first = naming.run_naming(
        str(FAKE_GRAPHIFY),
        naming_dir,
        staging_home,
        stripped,
        settings,
        health_check=lambda *a, **kw: True,
    )
    assert first.labeling == naming.LABELING_OK
    # First run: both communities are brand-new -> both get relabeled by the LLM.
    assert sorted(first.changed_cids) == ["0", "1"]

    entries_after_first = [
        json.loads(line) for line in call_log.read_text(encoding="utf-8").splitlines()
    ]
    label_entries = [e for e in entries_after_first if e["cmd"] == "label"]
    assert len(label_entries) == 1
    assert sorted(label_entries[0]["relabeled_cids"]) == [0, 1]

    # Second run over the SAME (unchanged) graph: nothing should be relabeled.
    second = naming.run_naming(
        str(FAKE_GRAPHIFY),
        naming_dir,
        staging_home,
        first.graph_data,
        settings,
        health_check=lambda *a, **kw: True,
    )
    assert second.labeling == naming.LABELING_OK
    assert second.changed_cids == []

    entries_after_second = [
        json.loads(line) for line in call_log.read_text(encoding="utf-8").splitlines()
    ]
    new_entries = entries_after_second[len(entries_after_first) :]
    assert all(e["cmd"] != "label" for e in new_entries)  # label was never invoked this run


def test_partial_change_only_relabels_changed_community(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_GRAPHIFY_CONTROL", str(tmp_path / "control.json"))
    call_log = tmp_path / "call-log.jsonl"
    monkeypatch.setenv("FAKE_GRAPHIFY_CALL_LOG", str(call_log))

    settings = _settings(tmp_path)
    naming_dir = tmp_path / "naming"
    staging_home = tmp_path / "naming-home"
    stripped = naming.strip_project_community_attrs(_merged_graph())

    first = naming.run_naming(
        str(FAKE_GRAPHIFY),
        naming_dir,
        staging_home,
        stripped,
        settings,
        health_check=lambda *a, **kw: True,
    )
    assert first.labeling == naming.LABELING_OK

    # Mutate only the example-org.styleguide community's membership (add a node) so
    # its sig changes; example-org.services is untouched.
    mutated = json.loads(json.dumps(first.graph_data))
    mutated["nodes"].append(
        {
            "id": "example-org.styleguide::a3",
            "label": "alpha_extra",
            "repo_tag": "example-org.styleguide",
        }
    )
    mutated["links"].append(
        {
            "source": "example-org.styleguide::a1",
            "target": "example-org.styleguide::a3",
            "relation": "calls",
        }
    )

    second = naming.run_naming(
        str(FAKE_GRAPHIFY),
        naming_dir,
        staging_home,
        mutated,
        settings,
        health_check=lambda *a, **kw: True,
    )
    assert second.labeling == naming.LABELING_OK

    entries = [json.loads(line) for line in call_log.read_text(encoding="utf-8").splitlines()]
    label_entries = [e for e in entries if e["cmd"] == "label"]
    assert len(label_entries) == 2
    # Only the styleguide community was relabeled on the second run — cids
    # are assigned by sorting community keys ("example-org.services" < "example-org.styleguide"),
    # so styleguide is cid 1.
    assert label_entries[1]["relabeled_cids"] == [1]


def _fail_if_called(*args, **kwargs):
    pytest.fail("should not have been called: fingerprint-reuse path must skip this")


def test_run_naming_reuses_on_matching_fingerprint(tmp_path, monkeypatch):
    """Second run with an identical stripped graph must reuse the on-disk
    named graph wholesale: no cluster-only, no label, no health check."""
    monkeypatch.setenv("FAKE_GRAPHIFY_CONTROL", str(tmp_path / "control.json"))
    call_log = tmp_path / "call-log.jsonl"
    monkeypatch.setenv("FAKE_GRAPHIFY_CALL_LOG", str(call_log))

    settings = _settings(tmp_path)
    naming_dir = tmp_path / "naming"
    staging_home = tmp_path / "naming-home"
    stripped = naming.strip_project_community_attrs(_merged_graph())

    first = naming.run_naming(
        str(FAKE_GRAPHIFY),
        naming_dir,
        staging_home,
        stripped,
        settings,
        health_check=lambda *a, **kw: True,
    )
    assert first.labeling == naming.LABELING_OK
    fingerprint_path = naming_dir / "graphify-out" / naming.FINGERPRINT_FILENAME
    assert fingerprint_path.is_file()

    monkeypatch.setattr(graphify_cli, "run_cluster_only", _fail_if_called)
    monkeypatch.setattr(graphify_cli, "run_label", _fail_if_called)

    second = naming.run_naming(
        str(FAKE_GRAPHIFY),
        naming_dir,
        staging_home,
        stripped,
        settings,
        health_check=_fail_if_called,
    )

    assert second.labeling == naming.LABELING_REUSED
    assert any(n.get("community_name") for n in second.graph_data["nodes"])
    # The reused NamingResult still carries backend/backend_check — callers
    # (pipeline.py's manifest writer) read these unconditionally regardless
    # of which labeling outcome came back.
    assert second.backend == first.backend
    assert second.backend_check is not None
    assert second.backend_check == first.backend_check


def test_stale_sidecar_removed_before_crash_can_leave_it_behind(tmp_path, monkeypatch):
    """Regression test for the stale-sidecar crash window: a full run must
    unlink any pre-existing (now-stale) fingerprint sidecar BEFORE writing
    graph.json/running cluster-only+label, so that a crash partway through
    labeling (subprocess succeeded, python process died before the final
    OK-path sidecar write) can never leave a sidecar on disk next to a
    half-labeled graph.json for the next run's reuse gate to wrongly serve."""
    monkeypatch.setenv("FAKE_GRAPHIFY_CONTROL", str(tmp_path / "control.json"))
    call_log = tmp_path / "call-log.jsonl"
    monkeypatch.setenv("FAKE_GRAPHIFY_CALL_LOG", str(call_log))

    settings = _settings(tmp_path)
    naming_dir = tmp_path / "naming"
    staging_home = tmp_path / "naming-home"
    stripped = naming.strip_project_community_attrs(_merged_graph())

    first = naming.run_naming(
        str(FAKE_GRAPHIFY),
        naming_dir,
        staging_home,
        stripped,
        settings,
        health_check=lambda *a, **kw: True,
    )
    assert first.labeling == naming.LABELING_OK
    fingerprint_path = naming_dir / "graphify-out" / naming.FINGERPRINT_FILENAME
    assert fingerprint_path.is_file()  # stale-to-be sidecar from the "prior good run"

    real_try_reuse = naming._try_reuse_named_graph
    real_run_label = graphify_cli.run_label

    # Mutate the graph so cluster-only actually detects a changed community
    # and `label` really gets invoked this run (an unchanged graph would
    # take the sig-gated "nothing to relabel" shortcut and never call
    # `label` at all — see test_sig_unchanged_community_not_sent_to_label...).
    mutated = json.loads(json.dumps(stripped))
    mutated["nodes"].append(
        {
            "id": "example-org.styleguide::a3",
            "label": "alpha_extra",
            "repo_tag": "example-org.styleguide",
        }
    )

    # Force the full path even though a fingerprint sidecar exists on disk
    # (e.g. a transient reuse-gate miss), then let `label` genuinely write
    # names into graph.json before simulating the process dying right
    # after — before run_naming ever reaches its own OK-path sidecar write.
    monkeypatch.setattr(naming, "_try_reuse_named_graph", lambda *a, **kw: None)

    def crash_after_label(*args, **kwargs):
        result = real_run_label(*args, **kwargs)
        assert result.ok
        raise RuntimeError("simulated crash: process died right after label wrote names")

    monkeypatch.setattr(graphify_cli, "run_label", crash_after_label)

    with pytest.raises(RuntimeError, match="simulated crash"):
        naming.run_naming(
            str(FAKE_GRAPHIFY),
            naming_dir,
            staging_home,
            mutated,
            settings,
            health_check=lambda *a, **kw: True,
        )

    # graph.json really was (partially) relabeled by the real subprocess...
    written = json.loads(
        (naming_dir / "graphify-out" / "graph.json").read_text(encoding="utf-8")
    )
    assert any(n.get("community_name") for n in written["nodes"])
    # ...but the stale sidecar from the earlier good run must be gone, even
    # though the crash happened before the OK-path sidecar write ever ran.
    assert not fingerprint_path.is_file()

    # Restore real behavior; a subsequent run over the same graph must do a
    # full run (no sidecar to match) rather than ever reusing the
    # crash-truncated on-disk graph.
    monkeypatch.setattr(naming, "_try_reuse_named_graph", real_try_reuse)
    monkeypatch.setattr(graphify_cli, "run_label", real_run_label)

    entries_before = [
        json.loads(line) for line in call_log.read_text(encoding="utf-8").splitlines()
    ]
    second = naming.run_naming(
        str(FAKE_GRAPHIFY),
        naming_dir,
        staging_home,
        stripped,
        settings,
        health_check=lambda *a, **kw: True,
    )
    assert second.labeling == naming.LABELING_OK
    entries_after = [
        json.loads(line) for line in call_log.read_text(encoding="utf-8").splitlines()
    ]
    new_entries = entries_after[len(entries_before) :]
    assert any(e["cmd"] == "cluster-only" for e in new_entries)  # a real full run, not reuse


def test_reuse_skipped_for_non_object_named_graph_json(tmp_path, monkeypatch):
    """`graph.json` containing valid-but-non-object JSON (e.g. `[]`) must not
    crash `_try_reuse_named_graph` — it must fall through to a full naming
    run instead."""
    monkeypatch.setenv("FAKE_GRAPHIFY_CONTROL", str(tmp_path / "control.json"))
    call_log = tmp_path / "call-log.jsonl"
    monkeypatch.setenv("FAKE_GRAPHIFY_CALL_LOG", str(call_log))

    settings = _settings(tmp_path)
    naming_dir = tmp_path / "naming"
    staging_home = tmp_path / "naming-home"
    stripped = naming.strip_project_community_attrs(_merged_graph())

    out_dir = naming_dir / "graphify-out"
    out_dir.mkdir(parents=True)
    fingerprint = publish.output_hash(stripped)
    (out_dir / naming.FINGERPRINT_FILENAME).write_text(fingerprint + "\n", encoding="utf-8")
    (out_dir / "graph.json").write_text("[]", encoding="utf-8")

    result = naming.run_naming(
        str(FAKE_GRAPHIFY),
        naming_dir,
        staging_home,
        stripped,
        settings,
        health_check=lambda *a, **kw: True,
    )

    assert result.labeling == naming.LABELING_OK
    entries = [json.loads(line) for line in call_log.read_text(encoding="utf-8").splitlines()]
    assert any(e["cmd"] == "cluster-only" for e in entries)  # full run happened, no crash


def test_run_naming_full_run_on_changed_fingerprint(tmp_path, monkeypatch):
    """A graph that changed since the sidecar was written must take the
    normal full path — cluster-only + label invoked again."""
    monkeypatch.setenv("FAKE_GRAPHIFY_CONTROL", str(tmp_path / "control.json"))
    call_log = tmp_path / "call-log.jsonl"
    monkeypatch.setenv("FAKE_GRAPHIFY_CALL_LOG", str(call_log))

    settings = _settings(tmp_path)
    naming_dir = tmp_path / "naming"
    staging_home = tmp_path / "naming-home"
    stripped = naming.strip_project_community_attrs(_merged_graph())

    first = naming.run_naming(
        str(FAKE_GRAPHIFY),
        naming_dir,
        staging_home,
        stripped,
        settings,
        health_check=lambda *a, **kw: True,
    )
    assert first.labeling == naming.LABELING_OK

    changed = naming.strip_project_community_attrs(json.loads(json.dumps(_merged_graph())))
    changed["nodes"].append(
        {
            "id": "example-org.styleguide::a3",
            "label": "alpha_extra",
            "repo_tag": "example-org.styleguide",
        }
    )

    entries_before = [
        json.loads(line) for line in call_log.read_text(encoding="utf-8").splitlines()
    ]

    second = naming.run_naming(
        str(FAKE_GRAPHIFY),
        naming_dir,
        staging_home,
        changed,
        settings,
        health_check=lambda *a, **kw: True,
    )

    assert second.labeling == naming.LABELING_OK
    entries_after = [
        json.loads(line) for line in call_log.read_text(encoding="utf-8").splitlines()
    ]
    new_entries = entries_after[len(entries_before) :]
    assert any(e["cmd"] == "cluster-only" for e in new_entries)


def test_no_sidecar_written_on_degraded_run(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_GRAPHIFY_CONTROL", str(tmp_path / "control.json"))
    call_log = tmp_path / "call-log.jsonl"
    monkeypatch.setenv("FAKE_GRAPHIFY_CALL_LOG", str(call_log))

    settings = _settings(tmp_path)
    naming_dir = tmp_path / "naming"
    stripped = naming.strip_project_community_attrs(_merged_graph())

    result = naming.run_naming(
        str(FAKE_GRAPHIFY),
        naming_dir,
        tmp_path / "naming-home",
        stripped,
        settings,
        health_check=lambda *a, **kw: False,
    )

    assert result.labeling == naming.LABELING_DEGRADED
    fingerprint_path = naming_dir / "graphify-out" / naming.FINGERPRINT_FILENAME
    assert not fingerprint_path.is_file()


def test_reuse_refused_when_named_graph_missing(tmp_path, monkeypatch):
    """Sidecar matches but the on-disk named graph.json was deleted -> a
    full naming run happens, never a crash."""
    monkeypatch.setenv("FAKE_GRAPHIFY_CONTROL", str(tmp_path / "control.json"))
    call_log = tmp_path / "call-log.jsonl"
    monkeypatch.setenv("FAKE_GRAPHIFY_CALL_LOG", str(call_log))

    settings = _settings(tmp_path)
    naming_dir = tmp_path / "naming"
    staging_home = tmp_path / "naming-home"
    stripped = naming.strip_project_community_attrs(_merged_graph())

    first = naming.run_naming(
        str(FAKE_GRAPHIFY),
        naming_dir,
        staging_home,
        stripped,
        settings,
        health_check=lambda *a, **kw: True,
    )
    assert first.labeling == naming.LABELING_OK

    graph_path = naming_dir / "graphify-out" / "graph.json"
    graph_path.unlink()

    second = naming.run_naming(
        str(FAKE_GRAPHIFY),
        naming_dir,
        staging_home,
        stripped,
        settings,
        health_check=lambda *a, **kw: True,
    )

    assert second.labeling == naming.LABELING_OK
    assert any(n.get("community_name") for n in second.graph_data["nodes"])


def test_backend_mismatch_raises_and_never_calls_cluster_only(tmp_path, monkeypatch):
    stub_root = tmp_path / "stub_site"
    (stub_root / "graspologic").mkdir(parents=True)
    (stub_root / "graspologic" / "__init__.py").write_text("", encoding="utf-8")
    interp = tmp_path / "interp_with_graspologic.sh"
    interp.write_text(
        f'#!/bin/sh\nexec env PYTHONPATH="{stub_root}" "{sys.executable}" "$@"\n', encoding="utf-8"
    )
    interp.chmod(interp.stat().st_mode | stat.S_IEXEC)

    mismatched_bin = tmp_path / "graphify_mismatched"
    mismatched_bin.write_text(f"#!{interp}\nprint('unused')\n", encoding="utf-8")
    mismatched_bin.chmod(mismatched_bin.stat().st_mode | stat.S_IEXEC)

    monkeypatch.setenv("FAKE_GRAPHIFY_CONTROL", str(tmp_path / "control.json"))
    call_log = tmp_path / "call-log.jsonl"
    monkeypatch.setenv("FAKE_GRAPHIFY_CALL_LOG", str(call_log))

    settings = _settings(tmp_path)
    stripped = naming.strip_project_community_attrs(_merged_graph())

    with pytest.raises(backend.BackendMismatchError):
        naming.run_naming(
            str(mismatched_bin),
            tmp_path / "naming",
            tmp_path / "naming-home",
            stripped,
            settings,
            health_check=lambda *a, **kw: True,
        )
    assert not call_log.exists()  # failed before cluster-only ever ran


# ---------------------------------------------------------------------------
# End-to-end pipeline tests
# ---------------------------------------------------------------------------


def test_pipeline_strip_then_relabel_no_per_project_leakage(env):
    env.add_repo(
        "example-org.styleguide",
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        "repo_a.json",
    )
    env.add_repo(
        "example-org.services",
        "example-org",
        "services",
        "services.example-org.dev.lo",
        "repo_b.json",
    )
    env.write_registry()
    settings = env.settings(ollama_health_check=lambda *a, **kw: True)

    report = run(settings)

    assert report.published
    assert report.labeling == "ok"
    assert report.clustering_backend == "louvain"

    graph = json.loads(
        (settings.global_dir / "current" / "global-graph.json").read_text(encoding="utf-8")
    )
    names = {n.get("community_name") for n in graph["nodes"]}
    # None of the original per-project names survived into the published output.
    assert "Alpha Domain" not in names
    assert "Beta Domain" not in names
    for node in graph["nodes"]:
        assert node.get("community_name")  # every clustered node got a real name
        assert not node["community_name"].startswith("Community ")

    manifest = json.loads(
        (settings.global_dir / "current" / "generation-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["clustering_backend"] == "louvain"
    assert manifest["labeling"] == "ok"


def test_previous_global_graph_not_loaded_on_healthy_naming(env, monkeypatch):
    """The previously published global graph (a full 38K-node JSON parse in
    production) must only be loaded when naming degrades — the healthy path
    never needs it."""
    calls = []
    real = publish.read_current_global_graph

    def counting(global_dir):
        calls.append(global_dir)
        return real(global_dir)

    monkeypatch.setattr(publish, "read_current_global_graph", counting)

    env.add_repo(
        "example-org.styleguide",
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        "repo_a.json",
    )
    env.write_registry()
    settings = env.settings(ollama_health_check=lambda *a, **kw: True)

    report = run(settings)

    assert report.published
    assert report.labeling == "ok"
    assert calls == []


def test_previous_global_graph_loaded_once_on_degraded_naming(env, monkeypatch):
    """The degraded-naming restore path is the only caller of
    read_current_global_graph, and it must load it exactly once."""
    env.add_repo(
        "example-org.styleguide",
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        "repo_a.json",
    )
    env.add_repo(
        "example-org.services",
        "example-org",
        "services",
        "services.example-org.dev.lo",
        "repo_b.json",
    )
    env.write_registry()

    # First (healthy) run establishes a real published global generation to
    # later restore from — no call-counting here, only on the degraded run.
    settings_healthy = env.settings(ollama_health_check=lambda *a, **kw: True)
    first = run(settings_healthy)
    assert first.published
    assert first.labeling == "ok"

    root_a = Path([r["root"] for r in env._repos if r["repo_id"] == "example-org.styleguide"][0])
    (root_a / "touched.py").write_text("# touch\n", encoding="utf-8")

    calls = []
    real = publish.read_current_global_graph

    def counting(global_dir):
        calls.append(global_dir)
        return real(global_dir)

    monkeypatch.setattr(publish, "read_current_global_graph", counting)

    settings_degraded = env.settings(ollama_health_check=lambda *a, **kw: False)
    second = run(settings_degraded)

    assert second.published
    assert second.labeling == "degraded"
    assert len(calls) == 1


def test_pipeline_degraded_mode_no_leakage_and_restores_last_global(env):
    env.add_repo(
        "example-org.styleguide",
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        "repo_a.json",
    )
    env.add_repo(
        "example-org.services",
        "example-org",
        "services",
        "services.example-org.dev.lo",
        "repo_b.json",
    )
    env.write_registry()

    # First (healthy) run establishes a real published global generation
    # with fresh names to later restore from.
    settings_healthy = env.settings(ollama_health_check=lambda *a, **kw: True)
    first = run(settings_healthy)
    assert first.published
    assert first.labeling == "ok"
    first_graph = json.loads(
        (env.mesh_root / "graphify" / "global" / "current" / "global-graph.json").read_text(
            encoding="utf-8"
        )
    )
    first_names_by_id = {n["id"]: n.get("community_name") for n in first_graph["nodes"]}
    assert all(first_names_by_id.values())

    # Touch a repo so a new run is actually triggered, then run degraded.
    root_a = Path([r["root"] for r in env._repos if r["repo_id"] == "example-org.styleguide"][0])
    (root_a / "touched.py").write_text("# touch\n", encoding="utf-8")

    settings_degraded = env.settings(ollama_health_check=lambda *a, **kw: False)
    second = run(settings_degraded)

    assert second.published
    assert second.labeling == "degraded"

    second_graph = json.loads(
        (env.mesh_root / "graphify" / "global" / "current" / "global-graph.json").read_text(
            encoding="utf-8"
        )
    )
    names_by_id = {n["id"]: n.get("community_name") for n in second_graph["nodes"]}

    # Every node that existed in the last published GLOBAL generation keeps
    # that exact global name (restored), never the per-project seed name.
    for node_id, prior_name in first_names_by_id.items():
        if node_id in names_by_id:
            assert names_by_id[node_id] == prior_name
    # No per-project name ever leaked in, degraded or not.
    all_names = set(names_by_id.values())
    assert "Alpha Domain" not in all_names
    assert "Beta Domain" not in all_names


def test_pipeline_backend_mismatch_blocks_publish_end_to_end(env):
    """Forced backend mismatch (deliverable 1/2) must hard-fail the naming
    stage and propagate all the way out of pipeline.run() uncaught — never
    silently degrade, never publish. Mirrors the existing
    test_publish_failure_between_write_and_flip_leaves_current_untouched
    pattern (pytest.raises around run()); current must stay untouched."""
    stub_root = env.tmp_path / "stub_site"
    (stub_root / "graspologic").mkdir(parents=True)
    (stub_root / "graspologic" / "__init__.py").write_text("", encoding="utf-8")
    interp = env.tmp_path / "interp_with_graspologic.sh"
    interp.write_text(
        f'#!/bin/sh\nexec env PYTHONPATH="{stub_root}" "{sys.executable}" "$@"\n', encoding="utf-8"
    )
    interp.chmod(interp.stat().st_mode | stat.S_IEXEC)

    mismatched_bin = env.tmp_path / "graphify_mismatched"
    # Delegate every other subcommand (update/extract/merge-graphs) to the
    # real fake stub so only the backend probe (based on THIS file's own
    # shebang) disagrees with the pinned constant.
    mismatched_bin.write_text(
        f"#!{interp}\n"
        "import runpy, sys\n"
        f"sys.argv[0] = {str(FAKE_GRAPHIFY)!r}\n"
        f"runpy.run_path({str(FAKE_GRAPHIFY)!r}, run_name='__main__')\n",
        encoding="utf-8",
    )
    mismatched_bin.chmod(mismatched_bin.stat().st_mode | stat.S_IEXEC)

    env.add_repo(
        "example-org.styleguide",
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        "repo_a.json",
    )
    env.write_registry()

    settings = Settings.from_env(
        mesh_root=env.mesh_root,
        scan_root=env.scan_root,
        registry_path=env.registry_path,
        graphify_bin=str(mismatched_bin),
        ollama_health_check=lambda *a, **kw: True,
    )

    with pytest.raises(backend.BackendMismatchError):
        run(settings)

    assert not (settings.global_dir / "current").exists()


def test_pipeline_degraded_mode_never_invokes_cluster_only_or_label(env):
    env.add_repo(
        "example-org.styleguide",
        "example-org",
        "styleguide",
        "styleguide.example-org.dev.lo",
        "repo_a.json",
    )
    env.write_registry()
    settings = env.settings(ollama_health_check=lambda *a, **kw: False)

    report = run(settings)

    assert report.published
    assert report.labeling == "degraded"
    call_log = env.read_call_log()
    assert not any(e["cmd"] in ("cluster-only", "label") for e in call_log)
