"""Layer 1/2 integration tests for the complete three-story two-pass pushover."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from steel_frame_optimization.analysis.hinges import ColumnHingeMode
from steel_frame_optimization.serialization import canonical_json_dumps
from steel_frame_optimization.validation.phase7 import (
    Phase7Artifact,
    build_phase7_artifact,
    phase7_gate_failures,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def phase7_artifact() -> Phase7Artifact:
    """Build the complete deterministic Phase 7 evidence once for this module."""

    project_root = Path(__file__).resolve().parents[2]
    return build_phase7_artifact(
        frame_path=project_root / "configs" / "frames" / "three_story_frame.json",
        sections_path=project_root / "data" / "processed" / "aisc_v14_1_w_shapes.json",
        solver_policy_path=project_root / "configs" / "analysis" / "phase6_nonlinear_policy.json",
        pushover_policy_path=project_root / "configs" / "analysis" / "phase7_pushover_policy.json",
        published_designs_path=(
            project_root / "configs" / "cases" / "three_story_published_gwo_stage2.json"
        ),
    )


def test_phase7_numerical_gate_passes(phase7_artifact: Phase7Artifact) -> None:
    """All predeclared convergence, equilibrium, refinement, and repeat gates pass."""

    assert phase7_gate_failures(phase7_artifact) == ()


def test_each_published_design_runs_two_fresh_complete_passes(
    phase7_artifact: Phase7Artifact,
) -> None:
    """Every design has complete preliminary and axial-dependent final target states."""

    assert len(phase7_artifact.baseline_design_results) == 3
    for result in phase7_artifact.baseline_design_results:
        assert result.preliminary.converged
        assert result.final.converged
        assert len(result.preliminary.target_states) == 5
        assert len(result.final.target_states) == 5
        assert len(result.preliminary.column_compressions_at_cp_n) == 15
        assert len(result.final.column_compressions_at_cp_n) == 15
        assert all(
            capacity.column_mode is ColumnHingeMode.PRELIMINARY_UNREDUCED
            for capacity in result.preliminary.member_capacities
            if capacity.member_type == "column"
        )
        assert all(
            capacity.column_mode is ColumnHingeMode.FINAL_AXIAL_REDUCED
            for capacity in result.final.member_capacities
            if capacity.member_type == "column"
        )


def test_target_outputs_are_complete_and_exactly_landed(
    phase7_artifact: Phase7Artifact,
) -> None:
    """Curves, member demands, hinges, and explicit constraints exist at every target."""

    tolerance = phase7_artifact.policy.target_landing_absolute_tolerance_m
    for result in phase7_artifact.baseline_design_results:
        for state in result.final.target_states:
            assert state.landing_error_m <= tolerance
            assert len(state.member_demands) == 27
            assert len(state.hinge_demands) == 54
            assert state.base_shear_n > 0.0
        assert len(result.hierarchy_results) == 10
        assert result.constraints
        assert result.primary_feasible is (not result.failed_primary_constraint_ids)


def test_penalty_constraint_accuracy_is_measured_not_assumed(
    phase7_artifact: Phase7Artifact,
) -> None:
    """The actual maximum MP-constraint translation mismatch passes its fixed gate."""

    assert phase7_artifact.policy.constraint_handler == "Penalty"
    assert (
        phase7_artifact.maximum_constraint_translation_residual_m
        <= phase7_artifact.policy.constraint_translation_absolute_tolerance_m
    )


def test_published_drift_differences_remain_research_diagnostics(
    phase7_artifact: Phase7Artifact,
) -> None:
    """All Table 5 values are compared without turning disagreement into a false pass."""

    assert len(phase7_artifact.published_drift_comparisons) == 12
    assert all(
        item.acceptance_role == "research_diagnostic_only"
        for item in phase7_artifact.published_drift_comparisons
    )
    assert any(
        abs(difference) > 0.0
        for item in phase7_artifact.published_drift_comparisons
        for difference in item.signed_differences_percentage_points
    )


def test_committed_phase7_evidence_is_canonical_and_current(
    project_root: Path, phase7_artifact: Phase7Artifact
) -> None:
    """The checked-in evidence equals a fresh locked-environment calculation byte for byte."""

    evidence_path = project_root / "docs" / "evidence" / "phase07" / "three_story_pushover.json"
    persisted = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert persisted == json.loads(canonical_json_dumps(phase7_artifact))
