"""Assemble and verify the frozen Phase 12 runtime in an existing Colab VM."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

EXPECTED_RUNTIME_SHA256 = "c382c0867130d68ba3ee81869f7cca382c09a34c3e0007712fe1fd2c731f755d"
EXPECTED_PROJECT_SHA256 = "1e4b92ae2b15bc3f5851cbd6b7ffc2ac265e207e4d2c34fa74f8e5e9a325c15a"
CONTENT_ROOT = Path("/content")
DRIVE_INPUTS = Path("/content/drive/MyDrive/steel-frame-phase12-v1/immutable-inputs")
RUNTIME_ARCHIVE = CONTENT_ROOT / "phase12-python-3.12.6-runtime.tar.gz"
PROJECT_ARCHIVE = CONTENT_ROOT / "steel-frame-phase12-project.zip"
PROJECT_ROOT = CONTENT_ROOT / "phase12-project"


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assemble_runtime() -> None:
    """Concatenate the uploaded runtime parts without altering their bytes."""

    parts = sorted(CONTENT_ROOT.glob("runtime.part.*"))
    if len(parts) != 9:
        raise RuntimeError(f"expected 9 runtime parts, found {len(parts)}")
    with RUNTIME_ARCHIVE.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def persist_verified(source: Path, expected_sha256: str) -> Path:
    """Atomically copy one verified immutable input to Drive and reread its hash."""

    DRIVE_INPUTS.mkdir(parents=True, exist_ok=True)
    destination = DRIVE_INPUTS / source.name
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    if sha256_file(temporary) != expected_sha256:
        raise RuntimeError(f"Drive copy hash differs for {source.name}")
    temporary.replace(destination)
    if sha256_file(destination) != expected_sha256:
        raise RuntimeError(f"persisted Drive hash differs for {source.name}")
    return destination


def main() -> None:
    """Verify, persist and extract the frozen runtime and project inputs."""

    assemble_runtime()
    runtime_sha256 = sha256_file(RUNTIME_ARCHIVE)
    project_sha256 = sha256_file(PROJECT_ARCHIVE)
    if runtime_sha256 != EXPECTED_RUNTIME_SHA256:
        raise RuntimeError(f"runtime SHA-256 differs: {runtime_sha256}")
    if project_sha256 != EXPECTED_PROJECT_SHA256:
        raise RuntimeError(f"project SHA-256 differs: {project_sha256}")

    runtime_destination = persist_verified(RUNTIME_ARCHIVE, EXPECTED_RUNTIME_SHA256)
    project_destination = persist_verified(PROJECT_ARCHIVE, EXPECTED_PROJECT_SHA256)

    runtime_root = Path("/kaggle/working/runtime")
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    Path("/kaggle/working").mkdir(parents=True, exist_ok=True)
    with tarfile.open(RUNTIME_ARCHIVE, mode="r:gz") as archive:
        archive.extractall(Path("/kaggle/working"), filter="fully_trusted")
    if PROJECT_ROOT.exists():
        shutil.rmtree(PROJECT_ROOT)
    PROJECT_ROOT.mkdir(parents=True)
    with zipfile.ZipFile(PROJECT_ARCHIVE) as archive:
        archive.extractall(PROJECT_ROOT)

    python = Path("/kaggle/working/runtime/venv/bin/python")
    if not python.exists():
        raise RuntimeError("frozen Python executable was not extracted")
    print(
        json.dumps(
            {
                "runtime_archive_bytes": RUNTIME_ARCHIVE.stat().st_size,
                "runtime_archive_sha256": runtime_sha256,
                "runtime_drive_path": str(runtime_destination),
                "project_archive_bytes": PROJECT_ARCHIVE.stat().st_size,
                "project_archive_sha256": project_sha256,
                "project_drive_path": str(project_destination),
                "python_path": str(python),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
