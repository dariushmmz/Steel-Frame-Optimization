"""Verify cross-session Kaggle checkpoint persistence without another evaluation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, cast

INPUT_ROOT: Final = Path("/kaggle/input")
WORKING: Final = Path("/kaggle/working")
EXPECTED_PRODUCED_CHECKPOINT_SHA256: Final = (
    "ff2ef2ad3b5057ce12c31b1ba35bbcf56b0df2e4ecc742e127d34116aefee27c"
)
RELATIVE_TOLERANCE: Final = 1e-8
ABSOLUTE_TOLERANCE: Final = 1e-10


def sha256_file(path: Path) -> str:
    """Return one streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_one(pattern: str) -> Path:
    """Return the unique mounted file matching a recursive pattern."""

    candidates = sorted(INPUT_ROOT.rglob(pattern))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one {pattern}, found {candidates}")
    return candidates[0]


def main() -> None:
    """Consume the prior session output and persist the final migration gate."""

    expected_path = require_one("reference.json")
    actual_path = require_one("kaggle_checkpoint_probe.json")
    comparison_module = require_one("phase12_cross_platform_compare.py")
    specification = importlib.util.spec_from_file_location(
        "phase12_cross_platform_compare", comparison_module
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load the mounted comparison policy")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    compare_probe_evidence = cast(Callable[..., dict[str, Any]], module.compare_probe_evidence)

    produced_candidates = [
        path
        for path in INPUT_ROOT.rglob("seed_18.json")
        if sha256_file(path) == EXPECTED_PRODUCED_CHECKPOINT_SHA256
    ]
    if len(produced_candidates) != 1:
        raise RuntimeError(
            f"expected one produced checkpoint with frozen hash, found {produced_candidates}"
        )
    produced_checkpoint = produced_candidates[0]
    checkpoint_content: dict[str, Any] = json.loads(produced_checkpoint.read_text(encoding="utf-8"))
    expected: dict[str, Any] = json.loads(expected_path.read_text(encoding="utf-8"))
    actual: dict[str, Any] = json.loads(actual_path.read_text(encoding="utf-8"))
    comparison = compare_probe_evidence(
        expected,
        actual,
        relative_tolerance=RELATIVE_TOLERANCE,
        absolute_tolerance=ABSOLUTE_TOLERANCE,
    )
    checkpoint_state_fields = (
        "campaign_identity_equal",
        "evaluator_identity_equal",
        "population_restored_equal",
        "rng_state_restored_equal",
        "cache_restored_equal",
        "history_restored_equal",
        "run_history_equal_checkpoint",
        "run_population_equal_checkpoint",
    )
    gate = {
        "schema_version": "1.0",
        "source_compatibility_kernel": "dariushmzx/steel-frame-phase-12-compatibility/4",
        "cross_session_checkpoint_retrieval_passed": True,
        "produced_checkpoint_sha256": sha256_file(produced_checkpoint),
        "produced_checkpoint_completed_epochs": checkpoint_content["completed_epochs"],
        "produced_checkpoint_evaluation_requests": checkpoint_content["evaluation_requests"],
        "checkpoint_state_gate_passed": all(actual[field] for field in checkpoint_state_fields),
        **comparison,
    }
    gate["migration_gate_passed"] = bool(
        gate["cross_session_checkpoint_retrieval_passed"]
        and gate["checkpoint_state_gate_passed"]
        and gate["numerical_equivalence_passed"]
        and gate["produced_checkpoint_completed_epochs"] == 27
        and gate["produced_checkpoint_evaluation_requests"] == 2700
    )
    output = WORKING / "kaggle_migration_gate.json"
    output.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(gate, sort_keys=True))
    if not gate["migration_gate_passed"]:
        raise RuntimeError("Kaggle migration gate failed")


if __name__ == "__main__":
    main()
