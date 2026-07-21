"""Settings.extract_concurrency env parsing (Task 7).

Mirrors _health_timeout_from_env's defensive-parsing style, but the floor
behavior is different: garbage/negative/zero silently falls back to the
hard floor (1 = fully sequential) rather than raising, since a bad
concurrency value should degrade to safe behavior, not crash the pipeline
at startup.
"""

from __future__ import annotations

from graphify_mesh.sync.config import (
    EXTRACT_DEFAULT_CONCURRENCY,
    EXTRACT_MIN_CONCURRENCY,
    Settings,
    _extract_concurrency_from_env,
)


def test_extract_concurrency_env_var_resolves(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_MESH_EXTRACT_CONCURRENCY", "3")
    assert (
        _extract_concurrency_from_env(
            "GRAPHIFY_MESH_EXTRACT_CONCURRENCY", EXTRACT_DEFAULT_CONCURRENCY
        )
        == 3
    )


def test_extract_concurrency_unset_uses_default(monkeypatch):
    monkeypatch.delenv("GRAPHIFY_MESH_EXTRACT_CONCURRENCY", raising=False)
    assert (
        _extract_concurrency_from_env(
            "GRAPHIFY_MESH_EXTRACT_CONCURRENCY", EXTRACT_DEFAULT_CONCURRENCY
        )
        == EXTRACT_DEFAULT_CONCURRENCY
    )


def test_extract_concurrency_garbage_falls_back_to_floor(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_MESH_EXTRACT_CONCURRENCY", "not-a-number")
    assert (
        _extract_concurrency_from_env(
            "GRAPHIFY_MESH_EXTRACT_CONCURRENCY", EXTRACT_DEFAULT_CONCURRENCY
        )
        == EXTRACT_MIN_CONCURRENCY
    )


def test_extract_concurrency_zero_falls_back_to_floor(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_MESH_EXTRACT_CONCURRENCY", "0")
    assert (
        _extract_concurrency_from_env(
            "GRAPHIFY_MESH_EXTRACT_CONCURRENCY", EXTRACT_DEFAULT_CONCURRENCY
        )
        == EXTRACT_MIN_CONCURRENCY
    )


def test_extract_concurrency_negative_falls_back_to_floor(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_MESH_EXTRACT_CONCURRENCY", "-5")
    assert (
        _extract_concurrency_from_env(
            "GRAPHIFY_MESH_EXTRACT_CONCURRENCY", EXTRACT_DEFAULT_CONCURRENCY
        )
        == EXTRACT_MIN_CONCURRENCY
    )


def test_settings_default_extract_concurrency(monkeypatch, tmp_path):
    monkeypatch.delenv("GRAPHIFY_MESH_EXTRACT_CONCURRENCY", raising=False)
    settings = Settings(
        mesh_root=tmp_path,
        scan_root=tmp_path,
        approved_root=tmp_path,
        registry_path=tmp_path / "registry.json",
    )
    assert settings.extract_concurrency == EXTRACT_DEFAULT_CONCURRENCY


def test_settings_extract_concurrency_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHIFY_MESH_EXTRACT_CONCURRENCY", "5")
    settings = Settings(
        mesh_root=tmp_path,
        scan_root=tmp_path,
        approved_root=tmp_path,
        registry_path=tmp_path / "registry.json",
    )
    assert settings.extract_concurrency == 5


def _settings_from_cli(monkeypatch, tmp_path, argv):
    """Drive graphify_mesh.sync.cli.main() far enough to capture the Settings
    it builds, without actually running the pipeline (run() is faked to
    avoid a full sync)."""
    from graphify_mesh.sync import cli as cli_module
    from graphify_mesh.sync.pipeline import RunReport

    captured: dict = {}

    def fake_run(settings):
        captured["settings"] = settings
        return RunReport(dry_run=settings.dry_run, reconciliation={})

    monkeypatch.setattr(cli_module, "run", fake_run)
    cli_module.main(
        [
            "--once",
            "--mesh-root",
            str(tmp_path),
            "--scan-root",
            str(tmp_path),
            *argv,
        ]
    )
    return captured["settings"]


def test_cli_extract_concurrency_zero_clamped_to_floor(monkeypatch, tmp_path):
    settings = _settings_from_cli(monkeypatch, tmp_path, ["--extract-concurrency", "0"])
    assert settings.extract_concurrency == EXTRACT_MIN_CONCURRENCY


def test_cli_extract_concurrency_negative_clamped_to_floor(monkeypatch, tmp_path):
    settings = _settings_from_cli(monkeypatch, tmp_path, ["--extract-concurrency", "-5"])
    assert settings.extract_concurrency == EXTRACT_MIN_CONCURRENCY


def test_cli_extract_concurrency_positive_passes_through(monkeypatch, tmp_path):
    settings = _settings_from_cli(monkeypatch, tmp_path, ["--extract-concurrency", "4"])
    assert settings.extract_concurrency == 4
