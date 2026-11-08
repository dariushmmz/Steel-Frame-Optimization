"""Compare Phase 12 checkpoint-probe evidence across operating systems."""

from __future__ import annotations

import math
from typing import Any


def compare_values(
    expected: Any,
    actual: Any,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
    path: str = "$",
    differences: list[dict[str, Any]],
    result_hash_differences: list[dict[str, str]],
) -> tuple[float, float]:
    """Compare nested JSON while separating hashes of platform-rounded numerical payloads."""

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
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
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
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                path=f"{path}.{key}",
                differences=differences,
                result_hash_differences=result_hash_differences,
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
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                path=f"{path}[{index}]",
                differences=differences,
                result_hash_differences=result_hash_differences,
            )
            maximum_absolute = max(maximum_absolute, child_absolute)
            maximum_relative = max(maximum_relative, child_relative)
        return maximum_absolute, maximum_relative
    if expected != actual:
        if path.endswith(".complete_result_sha256"):
            result_hash_differences.append(
                {"path": path, "expected": str(expected), "actual": str(actual)}
            )
        else:
            differences.append({"path": path, "expected": expected, "actual": actual})
    return maximum_absolute, maximum_relative


def compare_probe_evidence(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, Any]:
    """Return the cross-platform disposition for one bounded recovery probe."""

    differences: list[dict[str, Any]] = []
    result_hash_differences: list[dict[str, str]] = []
    maximum_absolute, maximum_relative = compare_values(
        expected["new_results"],
        actual["new_results"],
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        path="$.new_results",
        differences=differences,
        result_hash_differences=result_hash_differences,
    )
    history_differences: list[dict[str, Any]] = []
    history_hash_differences: list[dict[str, str]] = []
    history_absolute, history_relative = compare_values(
        expected["new_history"],
        actual["new_history"],
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        path="$.new_history",
        differences=history_differences,
        result_hash_differences=history_hash_differences,
    )
    return {
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "new_result_count": len(actual["new_results"]),
        "new_result_difference_count": len(differences),
        "complete_result_hash_difference_count": len(result_hash_differences),
        "new_history_difference_count": len(history_differences),
        "maximum_new_result_absolute_difference": maximum_absolute,
        "maximum_new_result_relative_difference": maximum_relative,
        "maximum_history_absolute_difference": history_absolute,
        "maximum_history_relative_difference": history_relative,
        "first_result_differences": differences[:10],
        "first_complete_result_hash_differences": result_hash_differences[:10],
        "history_differences": history_differences,
        "numerical_equivalence_passed": not differences and not history_differences,
    }
