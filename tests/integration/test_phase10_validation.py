"""Regression tests for the independent three-story model-freeze evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from steel_frame_optimization.optimization.evaluator import EvaluatorPaths, StructuralEvaluator
from steel_frame_optimization.serialization import sha256_file


@pytest.fixture
def phase10_artifact(project_root: Path) -> dict[str, Any]:
    """Load the artifact that was strictly validated by its evidence builder."""

    path = project_root / "docs" / "evidence" / "phase10" / "three_story_validation.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Phase 10 artifact must be a JSON object")
    return value


@pytest.mark.integration
@pytest.mark.regression
def test_fresh_champions_and_independent_objective_match(
    phase10_artifact: dict[str, Any],
) -> None:
    """Require all 30 fresh evaluations and the independent Eq. (20) audit to pass."""

    fresh = phase10_artifact["fresh_champion_validations"]
    assert len(fresh) == 30
    assert phase10_artifact["all_fresh_champions_match"]
    assert all(item["status"] == "feasible" for item in fresh)
    objective = phase10_artifact["independent_objective_audit"]
    assert objective["passed"]
    assert objective["maximum_relative_error"] < 1e-12


@pytest.mark.integration
@pytest.mark.regression
def test_refinement_and_no_filter_ablation_are_frozen(
    phase10_artifact: dict[str, Any],
) -> None:
    """Freeze convergence and the exhaustive excluded-section boundary study."""

    assert [point["roof_increment_ratio"] for point in phase10_artifact["refinement_points"]] == [
        0.001,
        0.0005,
        0.00025,
    ]
    assert phase10_artifact["refinement_passed"]
    assert phase10_artifact["maximum_refinement_relative_spread"] < 0.02
    assert len(phase10_artifact["no_filter_outcomes"]) == 148
    assert phase10_artifact["no_filter_status_counts"] == {"stage1_infeasible": 148}
    assert phase10_artifact["no_filter_stage1_feasible_count"] == 0
    assert phase10_artifact["no_filter_stage2_feasible_count"] == 0


@pytest.mark.integration
@pytest.mark.regression
def test_model_freeze_identity_and_source_hashes(
    project_root: Path, phase10_artifact: dict[str, Any]
) -> None:
    """Keep frozen inputs while requiring an explicit compatible implementation-hash chain."""

    evaluator = StructuralEvaluator(EvaluatorPaths.from_project_root(project_root))
    frozen_identity = dict(phase10_artifact["evaluator_identity"])
    current_identity = evaluator.identity.model_dump()
    frozen_implementation = frozen_identity.pop("implementation_sha256")
    current_implementation = current_identity.pop("implementation_sha256")
    assert frozen_identity == current_identity
    compatibility = json.loads(
        (project_root / "configs" / "validation" / "handover_runtime_compatibility.json").read_text(
            encoding="utf-8"
        )
    )
    assert compatibility["change_scope"] == "campaign_approval_input_only"
    assert compatibility["structural_calculation_code_changed"] is False
    assert compatibility["frozen_phase10_implementation_sha256"] == frozen_implementation
    extension = json.loads(
        (
            project_root / "configs" / "validation" / "phase11_three_story_compatibility.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        compatibility["handover_runtime_implementation_sha256"]
        == extension["handover_runtime_implementation_sha256"]
    )
    assert extension["change_scope"] == "nine_story_extension_with_exact_three_story_regression"
    assert extension["structural_calculation_code_changed"] is True
    assert extension["three_story_configuration_changed"] is False
    assert extension["three_story_frozen_evidence_changed"] is False
    assert extension["frozen_phase10_implementation_sha256"] == frozen_implementation
    phase12 = json.loads(
        (
            project_root / "configs" / "validation" / "phase12_three_story_compatibility.json"
        ).read_text(encoding="utf-8")
    )
    assert phase12["phase11_implementation_sha256"] == extension["phase11_implementation_sha256"]
    assert phase12["change_scope"] == "nine_story_evaluator_profiles_and_phase12_validation"
    assert phase12["structural_calculation_code_changed"] is True
    assert phase12["three_story_configuration_changed"] is False
    assert phase12["three_story_frozen_evidence_changed"] is False
    assert phase12["phase12_implementation_sha256"] == current_implementation
    for name, path in EvaluatorPaths.from_project_root(project_root).model_dump().items():
        assert isinstance(path, Path)
        assert phase10_artifact["model_freeze_inputs"][name] == sha256_file(path)


@pytest.mark.integration
@pytest.mark.regression
def test_predeclared_policy_and_phase9_summary_are_immutable(project_root: Path) -> None:
    """Verify the validation policy still points to the accepted Phase 9 evidence."""

    policy_path = project_root / "configs" / "validation" / "phase10_three_story_validation.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    summary_path = project_root / policy["phase9_summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert sha256_file(summary_path) == policy["phase9_summary_sha256"]
    assert policy["phase9_completion_commit"] == ("6ebbb428703b77409c023b39249970619d26bed3")
    assert summary["evaluation_requests_total"] == 300_000
    assert summary["all_champions_independently_reevaluated_equal"] is True


@pytest.mark.integration
@pytest.mark.regression
def test_assumption_sensitivities_remain_diagnostic(
    phase10_artifact: dict[str, Any],
) -> None:
    """Keep assumption effects visible without turning them into acceptance criteria."""

    gravity = phase10_artifact["gravity_sensitivity"]
    weak_axis = phase10_artifact["weak_axis_sensitivity"]
    assert gravity["acceptance_role"] == "diagnostic_only"
    assert gravity["baseline_primary_feasible"]
    assert gravity["sensitivity_primary_feasible"]
    assert gravity["relative_base_shear_change"] == pytest.approx(0.03468796117338374)
    assert weak_axis["acceptance_role"] == "diagnostic_only"
    assert weak_axis["baseline_maximum_utilization"] == pytest.approx(0.592759237631157)
    assert weak_axis["weak_axis_maximum_utilization"] == pytest.approx(0.7868228424649709)
    assert weak_axis["weak_axis_failed_count"] == 0
