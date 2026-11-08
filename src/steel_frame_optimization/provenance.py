"""Collect runtime provenance and construct or verify deterministic artifact manifests."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from steel_frame_optimization.models.manifests import (
    ArtifactFile,
    ArtifactManifest,
    RuntimeProvenance,
    VcsProvenance,
)
from steel_frame_optimization.serialization import sha256_file

PROVENANCE_PACKAGES = (
    "matplotlib",
    "mypy",
    "numpy",
    "openseespy",
    "pydantic",
    "pytest",
    "ruff",
)


def _command_output(command: Sequence[str]) -> tuple[int, str]:
    """Run a bounded read-only command and return its status and stripped combined output."""

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return 1, "unavailable"
    output = result.stdout.strip() or result.stderr.strip()
    return result.returncode, output


def collect_vcs_provenance(project_root: Path) -> VcsProvenance:
    """Return the real Git commit and dirty flag, or an explicit no-VCS record."""

    status, _ = _command_output(["git", "-C", str(project_root), "rev-parse", "--git-dir"])
    if status != 0:
        return VcsProvenance(kind="none")
    commit_status, commit = _command_output(["git", "-C", str(project_root), "rev-parse", "HEAD"])
    if commit_status != 0:
        raise RuntimeError("Git repository exists but HEAD cannot be resolved")
    worktree_status, worktree = _command_output(
        ["git", "-C", str(project_root), "status", "--porcelain"]
    )
    if worktree_status != 0:
        raise RuntimeError("Git repository exists but dirtiness cannot be determined")
    return VcsProvenance(kind="git", commit=commit, dirty=bool(worktree))


def collect_runtime_provenance(
    project_root: Path,
    *,
    run_id: str | None = None,
    created_at_utc: datetime | None = None,
) -> RuntimeProvenance:
    """Collect runtime facts, allowing injection of the only intentionally variable fields."""

    timestamp = created_at_utc or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("created_at_utc must be timezone-aware")
    package_versions: dict[str, str] = {}
    for package_name in PROVENANCE_PACKAGES:
        try:
            package_versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package_name] = "not installed"
    uv_status, uv_version = _command_output(["uv", "--version"])
    if uv_status != 0 or not uv_version:
        uv_version = "unavailable"
    return RuntimeProvenance(
        run_id=run_id or str(uuid4()),
        created_at_utc=timestamp,
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform=platform.platform(),
        machine=platform.machine(),
        uv_version=uv_version,
        package_versions=package_versions,
        lock_sha256=sha256_file(project_root / "uv.lock"),
        vcs=collect_vcs_provenance(project_root),
    )


def build_artifact_manifest(
    *,
    paths: Sequence[Path],
    root: Path,
    run_id: str,
    created_at_utc: datetime,
    config_sha256: str,
    lock_sha256: str,
) -> ArtifactManifest:
    """Build a path-sorted artifact manifest for files contained by ``root``."""

    resolved_root = root.resolve()
    files: list[ArtifactFile] = []
    for path in paths:
        resolved_path = path.resolve()
        relative_path = resolved_path.relative_to(resolved_root).as_posix()
        files.append(
            ArtifactFile(
                path=relative_path,
                size_bytes=resolved_path.stat().st_size,
                sha256=sha256_file(resolved_path),
            )
        )
    return ArtifactManifest(
        run_id=run_id,
        created_at_utc=created_at_utc,
        config_sha256=config_sha256,
        lock_sha256=lock_sha256,
        files=tuple(sorted(files, key=lambda item: item.path)),
    )


def verify_artifact_manifest(manifest: ArtifactManifest, root: Path) -> tuple[str, ...]:
    """Return deterministic mismatch messages without modifying the artifacts."""

    mismatches: list[str] = []
    resolved_root = root.resolve()
    for artifact in manifest.files:
        path = (resolved_root / artifact.path).resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError:
            mismatches.append(f"outside root: {artifact.path}")
            continue
        if not path.is_file():
            mismatches.append(f"missing: {artifact.path}")
            continue
        if path.stat().st_size != artifact.size_bytes:
            mismatches.append(f"size mismatch: {artifact.path}")
            continue
        if sha256_file(path) != artifact.sha256:
            mismatches.append(f"hash mismatch: {artifact.path}")
    return tuple(mismatches)
