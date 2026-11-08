"""Build one reusable, lock-synchronized Python 3.12.6 Kaggle runtime."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Final

INPUT_ROOT: Final = Path("/kaggle/input")
WORKING: Final = Path("/kaggle/working")
RUNTIME: Final = WORKING / "runtime"
PYTHON_INSTALLS: Final = RUNTIME / "python"
VENV: Final = RUNTIME / "venv"
ARCHIVE: Final = WORKING / "phase12-python-3.12.6-runtime.tar.gz"
MANIFEST: Final = WORKING / "phase12-runtime-manifest.json"
MULTIPROCESSING_PROBE: Final = WORKING / "multiprocessing_probe.py"
EXPECTED_PYTHON: Final = "3.12.6"
EXPECTED_UV: Final = "0.6.14"


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    """Run one required command and return its standard output."""

    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.stdout:
        print(completed.stdout, end="")
    return completed.stdout.strip()


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_managed_python() -> Path:
    """Locate the exact managed interpreter installed beneath the fixed runtime root."""

    candidates = sorted(PYTHON_INSTALLS.glob("cpython-3.12.6-*/bin/python3.12"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one managed Python 3.12.6, found {candidates}")
    return candidates[0]


def locate_project_input() -> Path:
    """Locate the one mounted frozen project root without assuming Kaggle's slug spelling."""

    candidates = sorted(path.parent for path in INPUT_ROOT.rglob("uv.lock"))
    if len(candidates) != 1:
        mounted = sorted(path.as_posix() for path in INPUT_ROOT.iterdir())
        raise RuntimeError(f"expected one mounted uv.lock, found {candidates}; mounts={mounted}")
    return candidates[0]


def verify_multiprocessing(venv_python: Path) -> list[int]:
    """Verify that the frozen interpreter can use spawned worker processes."""

    MULTIPROCESSING_PROBE.write_text(
        "from concurrent.futures import ProcessPoolExecutor\n"
        "import multiprocessing as mp\n\n"
        "def square(value: int) -> int:\n"
        "    return value * value\n\n"
        "if __name__ == '__main__':\n"
        "    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('spawn')) as pool:\n"
        "        print(','.join(str(value) for value in pool.map(square, (1, 2, 3))))\n",
        encoding="utf-8",
    )
    output = run([str(venv_python), str(MULTIPROCESSING_PROBE)])
    values = [int(value) for value in output.splitlines()[-1].split(",")]
    if values != [1, 4, 9]:
        raise RuntimeError(f"spawned multiprocessing probe differed: {values}")
    return values


def main() -> None:
    """Install, verify, archive, and describe the reusable locked runtime."""

    project_input = locate_project_input()
    if RUNTIME.exists():
        shutil.rmtree(RUNTIME)
    RUNTIME.mkdir(parents=True)

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            f"uv=={EXPECTED_UV}",
        ]
    )
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv executable was not installed")

    environment = dict(os.environ)
    environment.update(
        {
            "UV_CACHE_DIR": str(RUNTIME / "uv-cache"),
            "UV_PROJECT_ENVIRONMENT": str(VENV),
            "UV_PYTHON_INSTALL_DIR": str(PYTHON_INSTALLS),
            "UV_PYTHON_DOWNLOADS": "automatic",
        }
    )
    run([uv, "python", "install", EXPECTED_PYTHON], env=environment)
    managed_python = locate_managed_python()
    run(
        [
            uv,
            "sync",
            "--frozen",
            "--no-install-project",
            "--python",
            str(managed_python),
        ],
        cwd=project_input,
        env=environment,
    )

    venv_python = VENV / "bin" / "python"
    actual_python = run(
        [str(venv_python), "-c", "import platform; print(platform.python_version())"]
    ).splitlines()[-1]
    if actual_python != EXPECTED_PYTHON:
        raise RuntimeError(f"runtime Python differs: {actual_python}")
    uv_version = run([uv, "--version"]).splitlines()[-1]
    if uv_version != f"uv {EXPECTED_UV}":
        raise RuntimeError(f"uv version differs: {uv_version}")
    opensees_version = run(
        [
            str(venv_python),
            "-c",
            "import openseespy.opensees as ops; print(ops.version())",
        ]
    ).splitlines()[-1]
    spawned_values = verify_multiprocessing(venv_python)

    shutil.rmtree(RUNTIME / "uv-cache", ignore_errors=True)
    with tarfile.open(ARCHIVE, mode="w:gz", compresslevel=6) as archive:
        archive.add(RUNTIME, arcname="runtime")

    archive_bytes = ARCHIVE.stat().st_size
    archive_sha256 = sha256_file(ARCHIVE)
    shutil.rmtree(RUNTIME)
    MULTIPROCESSING_PROBE.unlink()

    manifest = {
        "schema_version": "1.0",
        "archive": ARCHIVE.name,
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha256,
        "python_version": actual_python,
        "uv_version": uv_version,
        "opensees_version": opensees_version,
        "spawned_multiprocessing_values": spawned_values,
        "project_git_commit": "cf628c32f61b72b76fe5627ade9c40adcb25357f",
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
