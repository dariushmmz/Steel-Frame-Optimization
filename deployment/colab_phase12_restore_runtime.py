"""Restore the frozen Phase 12 runtime from hash-verified Google Drive inputs."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

RUNTIME_SHA256 = "c382c0867130d68ba3ee81869f7cca382c09a34c3e0007712fe1fd2c731f755d"
PROJECT_SHA256 = "1e4b92ae2b15bc3f5851cbd6b7ffc2ac265e207e4d2c34fa74f8e5e9a325c15a"
DRIVE_INPUTS = Path("/content/drive/MyDrive/steel-frame-phase12-v1/immutable-inputs")


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one immutable input."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Verify and extract the exact runtime and project into a fresh Colab VM."""

    runtime_archive = DRIVE_INPUTS / "phase12-python-3.12.6-runtime.tar.gz"
    project_archive = DRIVE_INPUTS / "steel-frame-phase12-project.zip"
    if sha256_file(runtime_archive) != RUNTIME_SHA256:
        raise RuntimeError("Drive runtime archive SHA-256 differs")
    if sha256_file(project_archive) != PROJECT_SHA256:
        raise RuntimeError("Drive project archive SHA-256 differs")
    runtime_root = Path("/kaggle/working/runtime")
    project_root = Path("/content/phase12-project")
    shutil.rmtree(runtime_root, ignore_errors=True)
    shutil.rmtree(project_root, ignore_errors=True)
    Path("/kaggle/working").mkdir(parents=True, exist_ok=True)
    project_root.mkdir(parents=True)
    with tarfile.open(runtime_archive, mode="r:gz") as archive:
        archive.extractall(Path("/kaggle/working"), filter="fully_trusted")
    with zipfile.ZipFile(project_archive) as archive:
        archive.extractall(project_root)
    python = runtime_root / "venv/bin/python"
    if not python.is_file():
        raise RuntimeError("restored Python executable is missing")
    print(
        json.dumps(
            {
                "python": str(python),
                "project": str(project_root),
                "runtime_sha256": RUNTIME_SHA256,
                "project_sha256": PROJECT_SHA256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
