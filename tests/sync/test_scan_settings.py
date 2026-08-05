"""Settings scan_roots/approved_roots/scan_depth env + CLI parsing.

Mirrors test_extract_concurrency_settings.py's style: unit-test the parsing
helpers directly, then Settings.from_env, then round-trip through the CLI's
_settings_from_cli-style harness. See src/graphify_mesh/sync/config.py for
the authoritative precedence/clamping rules this exercises.
"""

from __future__ import annotations

from graphify_mesh.sync.config import (
    SCAN_DEFAULT_DEPTH,
    SCAN_MAX_DEPTH,
    SCAN_MIN_DEPTH,
    Settings,
    _approved_roots_from_env,
    _scan_depth_from_env,
    _scan_roots_from_env,
)

# ---------------------------------------------------------------------------
# _scan_roots_from_env: roots precedence
# ---------------------------------------------------------------------------


def test_scan_roots_env_colon_split(monkeypatch, tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    monkeypatch.delenv("GRAPHIFY_MESH_SCAN_ROOT", raising=False)
    monkeypatch.setenv("GRAPHIFY_MESH_SCAN_ROOTS", f"{a}:{b}")
    assert _scan_roots_from_env(None) == [a.resolve(), b.resolve()]


def test_scan_roots_env_strips_whitespace_and_drops_empty_segments(monkeypatch, tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    monkeypatch.delenv("GRAPHIFY_MESH_SCAN_ROOT", raising=False)
    monkeypatch.setenv("GRAPHIFY_MESH_SCAN_ROOTS", f"  {a}  ::  ")
    assert _scan_roots_from_env(None) == [a.resolve()]


def test_scan_roots_env_all_empty_falls_through_to_scan_root(monkeypatch, tmp_path):
    single = tmp_path / "single"
    single.mkdir()
    monkeypatch.setenv("GRAPHIFY_MESH_SCAN_ROOTS", ":::")
    monkeypatch.setenv("GRAPHIFY_MESH_SCAN_ROOT", str(single))
    assert _scan_roots_from_env(None) == [single.resolve()]


def test_scan_roots_env_wins_over_scan_root_env_when_both_set(monkeypatch, tmp_path):
    roots_value = tmp_path / "from-roots"
    root_value = tmp_path / "from-root"
    roots_value.mkdir()
    root_value.mkdir()
    monkeypatch.setenv("GRAPHIFY_MESH_SCAN_ROOTS", str(roots_value))
    monkeypatch.setenv("GRAPHIFY_MESH_SCAN_ROOT", str(root_value))
    assert _scan_roots_from_env(None) == [roots_value.resolve()]


def test_scan_roots_arg_takes_precedence_over_env(monkeypatch, tmp_path):
    explicit = tmp_path / "explicit"
    env_root = tmp_path / "env-root"
    explicit.mkdir()
    env_root.mkdir()
    monkeypatch.setenv("GRAPHIFY_MESH_SCAN_ROOTS", str(env_root))
    assert _scan_roots_from_env([str(explicit)]) == [explicit.resolve()]


def test_scan_roots_falls_back_to_single_scan_root_env(monkeypatch, tmp_path):
    single = tmp_path / "single-root"
    single.mkdir()
    monkeypatch.delenv("GRAPHIFY_MESH_SCAN_ROOTS", raising=False)
    monkeypatch.setenv("GRAPHIFY_MESH_SCAN_ROOT", str(single))
    assert _scan_roots_from_env(None) == [single.resolve()]


def test_scan_root_env_whitespace_only_treated_as_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("GRAPHIFY_MESH_SCAN_ROOTS", raising=False)
    monkeypatch.setenv("GRAPHIFY_MESH_SCAN_ROOT", "   ")
    monkeypatch.chdir(tmp_path)
    assert _scan_roots_from_env(None) == [tmp_path.resolve()]


def test_scan_roots_both_unset_defaults_to_cwd(monkeypatch, tmp_path):
    monkeypatch.delenv("GRAPHIFY_MESH_SCAN_ROOTS", raising=False)
    monkeypatch.delenv("GRAPHIFY_MESH_SCAN_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert _scan_roots_from_env(None) == [tmp_path.resolve()]


# ---------------------------------------------------------------------------
# _approved_roots_from_env
# ---------------------------------------------------------------------------


def test_approved_roots_env_set_overrides_scan_roots(monkeypatch, tmp_path):
    scan = [tmp_path / "scan"]
    approved_dir = tmp_path / "approved"
    approved_dir.mkdir()
    monkeypatch.setenv("GRAPHIFY_MESH_APPROVED_ROOTS", str(approved_dir))
    assert _approved_roots_from_env(scan) == [approved_dir.resolve()]


def test_approved_roots_unset_defaults_to_scan_roots(monkeypatch, tmp_path):
    scan = [tmp_path / "scan-a", tmp_path / "scan-b"]
    monkeypatch.delenv("GRAPHIFY_MESH_APPROVED_ROOTS", raising=False)
    assert _approved_roots_from_env(scan) == scan


def test_approved_roots_env_colon_split(monkeypatch, tmp_path):
    a = tmp_path / "approved-a"
    b = tmp_path / "approved-b"
    a.mkdir()
    b.mkdir()
    monkeypatch.setenv("GRAPHIFY_MESH_APPROVED_ROOTS", f"{a}:{b}")
    assert _approved_roots_from_env([tmp_path / "scan"]) == [a.resolve(), b.resolve()]


def test_approved_roots_env_strips_whitespace_and_drops_empty_segments(monkeypatch, tmp_path):
    a = tmp_path / "approved-only"
    a.mkdir()
    monkeypatch.setenv("GRAPHIFY_MESH_APPROVED_ROOTS", f"  {a}  ::  ")
    assert _approved_roots_from_env([tmp_path / "scan"]) == [a.resolve()]


def test_approved_roots_env_all_empty_falls_back_to_scan_roots(monkeypatch, tmp_path):
    scan = [tmp_path / "scan-only"]
    monkeypatch.setenv("GRAPHIFY_MESH_APPROVED_ROOTS", ":::")
    assert _approved_roots_from_env(scan) == scan


# ---------------------------------------------------------------------------
# _scan_depth_from_env
# ---------------------------------------------------------------------------


def test_scan_depth_unset_uses_default(monkeypatch):
    monkeypatch.delenv("GRAPHIFY_MESH_SCAN_DEPTH", raising=False)
    resolved = _scan_depth_from_env("GRAPHIFY_MESH_SCAN_DEPTH", SCAN_DEFAULT_DEPTH)
    assert resolved == SCAN_DEFAULT_DEPTH


def test_scan_depth_valid_value_passes_through(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_MESH_SCAN_DEPTH", "3")
    assert _scan_depth_from_env("GRAPHIFY_MESH_SCAN_DEPTH", SCAN_DEFAULT_DEPTH) == 3


def test_scan_depth_garbage_falls_back_to_floor(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_MESH_SCAN_DEPTH", "not-a-number")
    assert (
        _scan_depth_from_env("GRAPHIFY_MESH_SCAN_DEPTH", SCAN_DEFAULT_DEPTH) == SCAN_MIN_DEPTH
    )


def test_scan_depth_zero_falls_back_to_floor(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_MESH_SCAN_DEPTH", "0")
    assert (
        _scan_depth_from_env("GRAPHIFY_MESH_SCAN_DEPTH", SCAN_DEFAULT_DEPTH) == SCAN_MIN_DEPTH
    )


def test_scan_depth_negative_falls_back_to_floor(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_MESH_SCAN_DEPTH", "-1")
    assert (
        _scan_depth_from_env("GRAPHIFY_MESH_SCAN_DEPTH", SCAN_DEFAULT_DEPTH) == SCAN_MIN_DEPTH
    )


def test_scan_depth_above_cap_clamped_to_ceiling(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_MESH_SCAN_DEPTH", "99")
    assert (
        _scan_depth_from_env("GRAPHIFY_MESH_SCAN_DEPTH", SCAN_DEFAULT_DEPTH) == SCAN_MAX_DEPTH
    )


def test_settings_direct_construction_scan_depth_zero_clamped(tmp_path):
    settings = Settings(
        mesh_root=tmp_path,
        scan_roots=[tmp_path],
        approved_roots=[tmp_path],
        registry_path=tmp_path / "registry.json",
        scan_depth=0,
    )
    assert settings.scan_depth == SCAN_MIN_DEPTH


def test_settings_direct_construction_scan_depth_above_cap_clamped(tmp_path):
    settings = Settings(
        mesh_root=tmp_path,
        scan_roots=[tmp_path],
        approved_roots=[tmp_path],
        registry_path=tmp_path / "registry.json",
        scan_depth=99,
    )
    assert settings.scan_depth == SCAN_MAX_DEPTH


# ---------------------------------------------------------------------------
# CLI: --scan-root (repeatable) / --scan-depth (clamped)
# ---------------------------------------------------------------------------


def _settings_from_cli(monkeypatch, tmp_path, argv):
    """Drive graphify_mesh.sync.cli.main() far enough to capture the Settings
    it builds, without running the pipeline (run() is faked)."""
    from graphify_mesh.sync import cli as cli_module
    from graphify_mesh.sync.pipeline import RunReport

    captured: dict = {}

    def fake_run(settings):
        captured["settings"] = settings
        return RunReport(dry_run=settings.dry_run, reconciliation={})

    monkeypatch.setattr(cli_module, "run", fake_run)
    cli_module.main(["--once", "--mesh-root", str(tmp_path), *argv])
    return captured["settings"]


def test_cli_repeated_scan_root_flag(monkeypatch, tmp_path):
    root_a = tmp_path / "root-a"
    root_b = tmp_path / "root-b"
    root_a.mkdir()
    root_b.mkdir()
    monkeypatch.delenv("GRAPHIFY_MESH_SCAN_ROOTS", raising=False)
    monkeypatch.delenv("GRAPHIFY_MESH_SCAN_ROOT", raising=False)
    settings = _settings_from_cli(
        monkeypatch,
        tmp_path,
        ["--scan-root", str(root_a), "--scan-root", str(root_b)],
    )
    assert settings.scan_roots == [root_a.resolve(), root_b.resolve()]


def test_cli_single_scan_root_flag_back_compat(monkeypatch, tmp_path):
    root = tmp_path / "only-root"
    root.mkdir()
    monkeypatch.delenv("GRAPHIFY_MESH_SCAN_ROOTS", raising=False)
    monkeypatch.delenv("GRAPHIFY_MESH_SCAN_ROOT", raising=False)
    settings = _settings_from_cli(monkeypatch, tmp_path, ["--scan-root", str(root)])
    assert settings.scan_roots == [root.resolve()]


def test_cli_scan_depth_zero_clamped_to_floor(monkeypatch, tmp_path):
    settings = _settings_from_cli(
        monkeypatch, tmp_path, ["--scan-root", str(tmp_path), "--scan-depth", "0"]
    )
    assert settings.scan_depth == SCAN_MIN_DEPTH


def test_cli_scan_depth_above_cap_clamped_to_ceiling(monkeypatch, tmp_path):
    settings = _settings_from_cli(
        monkeypatch, tmp_path, ["--scan-root", str(tmp_path), "--scan-depth", "99"]
    )
    assert settings.scan_depth == SCAN_MAX_DEPTH
