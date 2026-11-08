"""Run the bounded Phase 12 numerical compatibility gate on Google Colab."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

EXPECTED_CHECKPOINT_SHA256 = "837a933fce479666497aec02272718adf6d3d6c82f5940ffcde4b967402bbc05"
RELATIVE_TOLERANCE = 1e-8
ABSOLUTE_TOLERANCE = 1e-10


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""

    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_comparator() -> ModuleType:
    """Load the project's cross-platform comparator from the uploaded source."""

    path = Path("/content/phase12_cross_platform_compare.py")
    spec = importlib.util.spec_from_file_location("phase12_cross_platform_compare", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the cross-platform comparator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Advance the portable checkpoint by one epoch and compare all new results."""

    project_root = Path("/content/phase12-project")
    python = Path("/kaggle/working/runtime/venv/bin/python")
    source_checkpoint = Path("/content/seed_18.json")
    if sha256_file(source_checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("compatibility checkpoint SHA-256 differs")
    working = Path("/content/phase12-colab-compatibility")
    working.mkdir(exist_ok=True)
    checkpoint = working / "seed_18.json"
    shutil.copy2(source_checkpoint, checkpoint)
    actual_path = working / "actual.json"
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(project_root / "src"),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "MPLBACKEND": "Agg",
        }
    )
    completed = subprocess.run(
        [
            str(python),
            "/content/phase12_checkpoint_probe.py",
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
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"checkpoint probe failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    expected = json.loads(Path("/content/reference.json").read_text(encoding="utf-8"))
    actual = json.loads(actual_path.read_text(encoding="utf-8"))
    comparator = load_comparator()
    comparison = comparator.compare_probe_evidence(
        expected,
        actual,
        relative_tolerance=RELATIVE_TOLERANCE,
        absolute_tolerance=ABSOLUTE_TOLERANCE,
    )
    identity_fields = (
        "campaign_identity_equal",
        "evaluator_identity_equal",
        "population_restored_equal",
        "rng_state_restored_equal",
        "cache_restored_equal",
        "history_restored_equal",
        "run_history_equal_checkpoint",
        "run_population_equal_checkpoint",
    )
    evidence = {
        "schema_version": "1.0",
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "python_version": subprocess.run(
            [str(python), "-c", "import platform; print(platform.python_version())"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "opensees_version": subprocess.run(
            [
                str(python),
                "-c",
                "import importlib.metadata; print(importlib.metadata.version('openseespy'))",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "input_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "identity_and_recovery_passed": all(actual[field] for field in identity_fields),
        "comparison": comparison,
    }
    output = Path("/content/drive/MyDrive/steel-frame-phase12-v1/evidence/colab_compatibility.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not evidence["identity_and_recovery_passed"]:
        raise RuntimeError("checkpoint recovery identity gate failed")
    if not comparison["numerical_equivalence_passed"]:
        raise RuntimeError("Colab numerical compatibility gate failed")


if __name__ == "__main__":
    main()
