"""Emit a minimal Kaggle CPU-runtime capability record."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

candidate_names = ("python", "python3", "python3.12", "python3.12.6")
interpreters: dict[str, dict[str, str | None]] = {}
for name in candidate_names:
    executable = shutil.which(name)
    if executable is None:
        interpreters[name] = {"path": None, "version": None}
        continue
    version = subprocess.run(
        [executable, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    interpreters[name] = {
        "path": executable,
        "version": (version.stdout or version.stderr).strip(),
    }

record = {
    "cpu_count": os.cpu_count(),
    "interpreters": interpreters,
    "machine": platform.machine(),
    "platform": platform.platform(),
    "python_version": sys.version,
}
Path("kaggle_access_probe.json").write_text(
    json.dumps(record, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(json.dumps(record, sort_keys=True))
