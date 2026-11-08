"""Layer 1 tests for M6 CLI help, dispatch, failures, and artifact verification."""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from steel_frame_optimization.provenance import build_artifact_manifest

pytestmark = pytest.mark.unit


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Execute the installed module entry point without invoking uv or syncing."""

    return subprocess.run(
        [sys.executable, "-m", "steel_frame_optimization", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_lists_every_m6_command() -> None:
    """Expose every required Phase 1 command in top-level help."""

    result = _run_cli("--help")
    assert result.returncode == 0
    for command in (
        "test-unit",
        "test-integration",
        "evaluate",
        "campaign",
        "report",
        "verify-artifacts",
        "provenance",
    ):
        assert command in result.stdout


@pytest.mark.parametrize(
    ("command", "capability", "phase"),
    [
        ("report", "structural report generation", "Phase 14"),
    ],
)
def test_unimplemented_commands_fail_clearly(
    command: str,
    capability: str,
    phase: str,
) -> None:
    """Return non-zero and name the missing capability and required phase."""

    result = _run_cli(command)
    assert result.returncode == 2
    assert f"not implemented: {capability} requires {phase}" in result.stderr
    assert result.stdout == ""


def test_campaign_requires_explicit_configuration_and_checkpoint_directory() -> None:
    """Prevent an accidental campaign launch without source-visible inputs and recovery path."""

    result = _run_cli("campaign")
    assert result.returncode == 2
    assert "--config" in result.stderr
    assert "--checkpoint-dir" in result.stderr


def test_artifact_verification_success_and_mutation(tmp_path: Path) -> None:
    """Verify exact bytes, then return non-zero after an artifact mutation."""

    artifact = tmp_path / "artifact.txt"
    artifact.write_text("verified", encoding="utf-8")
    manifest = build_artifact_manifest(
        paths=[artifact],
        root=tmp_path,
        run_id="cli-verification",
        created_at_utc=datetime(2026, 10, 27, tzinfo=UTC),
        config_sha256="1" * 64,
        lock_sha256="2" * 64,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")

    success = _run_cli("verify-artifacts", str(manifest_path), "--root", str(tmp_path))
    assert success.returncode == 0
    assert "verified 1 artifact(s)" in success.stdout

    artifact.write_text("mutated!", encoding="utf-8")
    failure = _run_cli("verify-artifacts", str(manifest_path), "--root", str(tmp_path))
    assert failure.returncode == 4
    assert "artifact verification failed:" in failure.stderr


def test_artifact_verification_rejects_invalid_manifest(tmp_path: Path) -> None:
    """Return a distinct non-zero status for an invalid manifest schema."""

    manifest_path = tmp_path / "invalid.json"
    manifest_path.write_text("{}", encoding="utf-8")
    result = _run_cli("verify-artifacts", str(manifest_path), "--root", str(tmp_path))
    assert result.returncode == 3
    assert "artifact verification failed:" in result.stderr
