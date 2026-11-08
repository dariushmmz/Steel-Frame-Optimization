"""Verify the frozen Kaggle runtime, checkpoint recovery, and Windows equivalence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tarfile
import traceback
from pathlib import Path
from typing import Any, Final

INPUT_ROOT: Final = Path("/kaggle/input")
WORKING: Final = Path("/kaggle/working")
RUNTIME_ARCHIVE_NAME: Final = "phase12-python-3.12.6-runtime.tar.gz"
EXPECTED_RUNTIME_SHA256: Final = "c382c0867130d68ba3ee81869f7cca382c09a34c3e0007712fe1fd2c731f755d"
EXPECTED_CHECKPOINT_SHA256: Final = (
    "837a933fce479666497aec02272718adf6d3d6c82f5940ffcde4b967402bbc05"
)
RELATIVE_TOLERANCE: Final = 1e-8
ABSOLUTE_TOLERANCE: Final = 1e-10


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_one(pattern: str) -> Path:
    """Return the unique mounted input matching one recursive pattern."""

    candidates = sorted(INPUT_ROOT.rglob(pattern))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one {pattern}, found {candidates}")
    return candidates[0]


def compare_values(
    expected: Any,
    actual: Any,
    *,
    path: str = "$",
    differences: list[dict[str, Any]],
) -> tuple[float, float]:
    """Compare nested JSON values and return maximum absolute and relative float errors."""

    maximum_absolute = 0.0
    maximum_relative = 0.0
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected != actual:
            differences.append({"path": path, "expected": expected, "actual": actual})
        return maximum_absolute, maximum_relative
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if isinstance(expected, int) and isinstance(actual, int):
            if expected != actual:
                differences.append({"path": path, "expected": expected, "actual": actual})
            return maximum_absolute, maximum_relative
        expected_float = float(expected)
        actual_float = float(actual)
        absolute = abs(expected_float - actual_float)
        scale = max(abs(expected_float), abs(actual_float))
        relative = absolute / scale if scale else 0.0
        maximum_absolute = absolute
        maximum_relative = relative
        if not math.isclose(
            expected_float,
            actual_float,
            rel_tol=RELATIVE_TOLERANCE,
            abs_tol=ABSOLUTE_TOLERANCE,
        ):
            differences.append(
                {
                    "path": path,
                    "expected": expected_float,
                    "actual": actual_float,
                    "absolute_difference": absolute,
                    "relative_difference": relative,
                }
            )
        return maximum_absolute, maximum_relative
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            differences.append(
                {
                    "path": path,
                    "expected_keys": sorted(expected),
                    "actual_keys": sorted(actual),
                }
            )
            return maximum_absolute, maximum_relative
        for key in sorted(expected):
            child_absolute, child_relative = compare_values(
                expected[key],
                actual[key],
                path=f"{path}.{key}",
                differences=differences,
            )
            maximum_absolute = max(maximum_absolute, child_absolute)
            maximum_relative = max(maximum_relative, child_relative)
        return maximum_absolute, maximum_relative
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            differences.append(
                {"path": path, "expected_length": len(expected), "actual_length": len(actual)}
            )
            return maximum_absolute, maximum_relative
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            child_absolute, child_relative = compare_values(
                expected_item,
                actual_item,
                path=f"{path}[{index}]",
                differences=differences,
            )
            maximum_absolute = max(maximum_absolute, child_absolute)
            maximum_relative = max(maximum_relative, child_relative)
        return maximum_absolute, maximum_relative
    if expected != actual:
        differences.append({"path": path, "expected": expected, "actual": actual})
    return maximum_absolute, maximum_relative


def main() -> None:
    """Execute the complete bounded compatibility gate."""

    runtime_archive = require_one(RUNTIME_ARCHIVE_NAME)
    runtime_sha256 = sha256_file(runtime_archive)
    if runtime_sha256 != EXPECTED_RUNTIME_SHA256:
        raise RuntimeError(f"runtime archive SHA-256 differs: {runtime_sha256}")
    with tarfile.open(runtime_archive, mode="r:gz") as archive:
        archive.extractall(WORKING, filter="fully_trusted")

    project_root = require_one("uv.lock").parent
    checkpoint_input = require_one("seed_18.json")
    reference_path = require_one("reference.json")
    probe_script = require_one("phase12_checkpoint_probe.py")
    checkpoint_sha256 = sha256_file(checkpoint_input)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"checkpoint SHA-256 differs: {checkpoint_sha256}")

    probe_directory = WORKING / "compatibility"
    probe_directory.mkdir()
    checkpoint = probe_directory / "seed_18.json"
    shutil.copy2(checkpoint_input, checkpoint)
    actual_path = WORKING / "kaggle_checkpoint_probe.json"
    venv_python = WORKING / "runtime/venv/bin/python"
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(project_root / "src"),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    subprocess.run(
        [
            str(venv_python),
            str(probe_script),
            "--project-root",
            str(project_root),
            "--policy",
            str(project_root / "configs/optimization/phase12_case1_no_filter_main.json"),
            "--checkpoint",
            str(checkpoint),
            "--seed",
            "18",
            "--maximum-epochs",
            "27",
            "--output",
            str(actual_path),
        ],
        cwd=project_root,
        env=environment,
        check=True,
    )

    expected = json.loads(reference_path.read_text(encoding="utf-8"))
    actual = json.loads(actual_path.read_text(encoding="utf-8"))
    differences: list[dict[str, Any]] = []
    maximum_absolute, maximum_relative = compare_values(
        expected["new_results"],
        actual["new_results"],
        path="$.new_results",
        differences=differences,
    )
    history_differences: list[dict[str, Any]] = []
    history_absolute, history_relative = compare_values(
        expected["new_history"],
        actual["new_history"],
        path="$.new_history",
        differences=history_differences,
    )
    gate = {
        "schema_version": "1.0",
        "runtime_archive_sha256": runtime_sha256,
        "checkpoint_input_sha256": checkpoint_sha256,
        "python_version": subprocess.run(
            [str(venv_python), "-c", "import platform; print(platform.python_version())"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "checkpoint_state_gate_passed": all(
            actual[field]
            for field in (
                "campaign_identity_equal",
                "evaluator_identity_equal",
                "population_restored_equal",
                "rng_state_restored_equal",
                "cache_restored_equal",
                "history_restored_equal",
                "run_history_equal_checkpoint",
                "run_population_equal_checkpoint",
            )
        ),
        "relative_tolerance": RELATIVE_TOLERANCE,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "new_result_count": len(actual["new_results"]),
        "new_result_difference_count": len(differences),
        "new_history_difference_count": len(history_differences),
        "maximum_new_result_absolute_difference": maximum_absolute,
        "maximum_new_result_relative_difference": maximum_relative,
        "maximum_history_absolute_difference": history_absolute,
        "maximum_history_relative_difference": history_relative,
        "first_result_differences": differences[:10],
        "history_differences": history_differences,
        "numerical_equivalence_passed": not differences and not history_differences,
    }
    output = WORKING / "kaggle_compatibility_gate.json"
    output.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.rmtree(WORKING / "runtime")
    print(json.dumps(gate, sort_keys=True))
    if not gate["checkpoint_state_gate_passed"] or not gate["numerical_equivalence_passed"]:
        raise RuntimeError("Kaggle compatibility gate failed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        (WORKING / "kaggle_compatibility_error.txt").write_text(error, encoding="utf-8")
        shutil.rmtree(WORKING / "runtime", ignore_errors=True)
        print(error)
