"""Emit the minimum CPU-runtime facts needed for the Phase 12 Colab gate."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys


def interpreter_version(name: str) -> dict[str, str | None]:
    """Return the resolved path and version for an interpreter command."""

    executable = shutil.which(name)
    if executable is None:
        return {"path": None, "version": None}
    completed = subprocess.run(
        [executable, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "path": executable,
        "version": (completed.stdout or completed.stderr).strip(),
    }


record = {
    "cpu_count": os.cpu_count(),
    "interpreters": {
        name: interpreter_version(name)
        for name in ("python", "python3", "python3.12", "python3.12.6")
    },
    "machine": platform.machine(),
    "platform": platform.platform(),
    "python_version": sys.version,
}
print(json.dumps(record, indent=2, sort_keys=True))
