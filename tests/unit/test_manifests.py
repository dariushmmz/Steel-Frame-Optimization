"""Layer 1 tests for artifact, seed, and provenance manifests."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from steel_frame_optimization.models.manifests import ArtifactManifest, SeedManifest
from steel_frame_optimization.provenance import (
    build_artifact_manifest,
    collect_runtime_provenance,
    collect_vcs_provenance,
    verify_artifact_manifest,
)
from steel_frame_optimization.serialization import sha256_config, sha256_file

pytestmark = pytest.mark.unit


def test_artifact_manifest_round_trip_order_and_verification(tmp_path: Path) -> None:
    """Round-trip a path-sorted manifest and verify its exact files."""

    second = tmp_path / "b.txt"
    first = tmp_path / "a.txt"
    second.write_text("second", encoding="utf-8")
    first.write_text("first", encoding="utf-8")
    manifest = build_artifact_manifest(
        paths=[second, first],
        root=tmp_path,
        run_id="round-trip",
        created_at_utc=datetime(2026, 10, 27, tzinfo=UTC),
        config_sha256=sha256_config({"case": "test"}),
        lock_sha256="0" * 64,
    )
    restored = ArtifactManifest.model_validate_json(manifest.model_dump_json())
    assert restored == manifest
    assert [artifact.path for artifact in manifest.files] == ["a.txt", "b.txt"]
    assert verify_artifact_manifest(restored, tmp_path) == ()


def test_artifact_verification_reports_mutation(tmp_path: Path) -> None:
    """Detect an artifact whose bytes change after manifest creation."""

    path = tmp_path / "artifact.txt"
    path.write_text("before", encoding="utf-8")
    manifest = build_artifact_manifest(
        paths=[path],
        root=tmp_path,
        run_id="mutation",
        created_at_utc=datetime(2026, 10, 27, tzinfo=UTC),
        config_sha256="1" * 64,
        lock_sha256="2" * 64,
    )
    path.write_text("after!", encoding="utf-8")
    assert verify_artifact_manifest(manifest, tmp_path) == ("hash mismatch: artifact.txt",)


def test_seed_manifest_is_explicit_and_delegated(project_root: Path) -> None:
    """Load the delegated seeds while preserving their pending-ratification status."""

    path = project_root / "configs" / "optimization" / "seed_manifest.json"
    manifest = SeedManifest.model_validate_json(path.read_text(encoding="utf-8"))
    assert manifest.decision_status == "agent_decided_pending_ratification_DEC-012"
    assert manifest.seeds == tuple(range(30))


def test_seed_manifest_rejects_duplicates() -> None:
    """Prevent duplicate trials from masquerading as independent seeds."""

    with pytest.raises(ValidationError, match="unique"):
        SeedManifest(
            generator="numpy.random.PCG64",
            decision_status="agent_decided_pending_ratification_DEC-012",
            purpose="invalid test fixture",
            seeds=(1, 1),
        )


def test_provenance_differs_only_in_declared_fields(project_root: Path) -> None:
    """Compare two observations after removing only time and run identity."""

    first = collect_runtime_provenance(
        project_root,
        run_id="run-a",
        created_at_utc=datetime(2026, 10, 27, 0, 0, tzinfo=UTC),
    )
    second = collect_runtime_provenance(
        project_root,
        run_id="run-b",
        created_at_utc=datetime(2026, 10, 27, 0, 1, tzinfo=UTC),
    )
    assert first.run_id != second.run_id
    assert first.created_at_utc != second.created_at_utc
    assert first.comparable_payload() == second.comparable_payload()
    assert first.uv_version == "uv 0.6.14 (a4cec56dc 2025-04-09)"
    assert first.lock_sha256 == sha256_file(project_root / "uv.lock")
    assert first.vcs.kind == "git"
    assert first.vcs.commit is not None


def test_provenance_rejects_naive_time(project_root: Path) -> None:
    """Require timezone-aware declared timestamps."""

    with pytest.raises(ValueError, match="timezone-aware"):
        collect_runtime_provenance(
            project_root,
            run_id="naive-time",
            created_at_utc=datetime(2026, 10, 27),
        )


def _run_git(repository: Path, *arguments: str) -> None:
    """Run a local-only Git command for the isolated provenance fixture."""

    subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )


def test_empty_git_status_means_clean(tmp_path: Path) -> None:
    """Treat successful empty porcelain output as clean rather than unavailable."""

    _run_git(tmp_path, "init", "-b", "main")
    _run_git(tmp_path, "config", "user.name", "M3 Test")
    _run_git(tmp_path, "config", "user.email", "m3-test@example.invalid")
    tracked_file = tmp_path / "tracked.txt"
    tracked_file.write_text("tracked\n", encoding="utf-8")
    _run_git(tmp_path, "add", "tracked.txt")
    _run_git(tmp_path, "commit", "-m", "fixture")

    vcs = collect_vcs_provenance(tmp_path)
    assert vcs.dirty is False
