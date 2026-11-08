"""Analytical and numerical checks for the nine-story nonlinear frame evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

pytestmark = pytest.mark.analytical


@pytest.fixture
def artifact(project_root: Path) -> dict[str, Any]:
    """Load the canonical Phase 12 pushover artifact."""

    return cast(
        dict[str, Any],
        json.loads(
            (project_root / "docs/evidence/phase12/nine_story_pushover.json").read_text(
                encoding="utf-8"
            )
        ),
    )


def test_nine_story_pushover_reaches_targets_and_closes_equilibrium(
    artifact: dict[str, Any],
) -> None:
    """Require target landing, equilibrium, diaphragm compatibility, and hinge consistency."""

    assert artifact["frame_id"] == "9-story"
    assert artifact["maximum_target_landing_error_m"] < 1.0e-10
    assert artifact["maximum_lateral_equilibrium_relative_residual"] < 1.0e-8
    assert artifact["maximum_vertical_equilibrium_relative_residual"] < 1.0e-8
    assert artifact["maximum_constraint_translation_residual_m"] < 1.0e-8
    assert artifact["maximum_hinge_force_mismatch_relative"] < 1.0e-8


def test_nine_story_refinement_and_repeat_are_stable(artifact: dict[str, Any]) -> None:
    """Require the predeclared refinement series and byte-stable repeated calculation."""

    assert [point["roof_increment_ratio"] for point in artifact["refinement_points"]] == [
        0.001,
        0.0005,
        0.00025,
    ]
    assert artifact["maximum_refinement_relative_spread"] < 0.02
    assert artifact["deterministic_repeat_equal"]
    assert len(set(artifact["repeat_result_sha256"])) == 1


def test_published_discrepancies_remain_diagnostic(artifact: dict[str, Any]) -> None:
    """Keep all Table 9 differences and project feasibility outcomes visible without fitting."""

    results = {item["design_id"]: item for item in artifact["baseline_design_results"]}
    assert results["case1_gwo"]["failed_primary_constraint_ids"] == ["CP:story:4:drift"]
    assert results["case1_no_filter_gwo"]["primary_feasible"] is True
    assert results["case2_gwo"]["failed_primary_constraint_ids"] == ["CP:story:4:drift"]
    comparisons = artifact["published_drift_comparisons"]
    assert len(comparisons) == 12
    assert all(item["acceptance_role"] == "research_diagnostic_only" for item in comparisons)
