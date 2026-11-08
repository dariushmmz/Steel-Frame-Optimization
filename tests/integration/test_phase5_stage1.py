"""Integration checks for deterministic complete Stage 1 evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from steel_frame_optimization.serialization import canonical_json_dumps
from steel_frame_optimization.validation.phase5 import Stage1Artifact, build_stage1_artifact

pytestmark = pytest.mark.integration


def _paths(project_root: Path) -> dict[str, Path]:
    """Return every explicit Phase 5 input path."""

    return {
        "frame_path": project_root / "configs/frames/three_story_frame.json",
        "sections_path": project_root / "data/processed/aisc_v14_1_w_shapes.json",
        "case_path": project_root / "configs/cases/three_story_case2_gwo.json",
        "policy_path": project_root / "configs/analysis/stage1_policy.json",
        "reference_path": project_root / "configs/cases/three_story_case2_stage1_reference.json",
    }


def test_stage1_artifact_is_deterministic_complete_and_direct(project_root: Path) -> None:
    """Run all six combinations twice and verify complete constraint/result invariants."""

    first = build_stage1_artifact(**_paths(project_root))
    second = build_stage1_artifact(**_paths(project_root))
    assert first == second
    assert set(first.analyses) == {
        "strength_gravity",
        "strength_seismic_positive",
        "strength_seismic_negative",
        "service_D",
        "service_L",
        "service_D_plus_L",
    }
    assert all(result.combination_evaluated_directly for result in first.analyses.values())
    assert all(
        result.load_stages == ("gravity_converged_and_held", "lateral_converged")
        for result in first.analyses.values()
    )
    assert all(
        result.equilibrium_relative_residual <= first.policy.equilibrium_relative_tolerance
        for result in first.analyses.values()
    )
    assert len(first.effective_lengths) == 15
    assert len(first.constraints) == 125
    assert len(first.research_comparisons) == 33
    assert first.feasible is (not first.failed_constraint_ids)
    assert first.maximum_utilization == max(item.utilization for item in first.constraints)
    assert any(item.signed_difference != 0.0 for item in first.research_comparisons)
    assert all(
        item.disposition == "diagnostic_not_an_acceptance_gate"
        for item in first.research_comparisons
    )


def test_stage1_artifact_matches_committed_evidence(project_root: Path) -> None:
    """Require canonical equality with the committed Phase 5 artifact."""

    actual = build_stage1_artifact(**_paths(project_root))
    evidence_path = project_root / "docs/evidence/phase05/case2_gwo_stage1.json"
    evidence = Stage1Artifact.model_validate_json(evidence_path.read_text(encoding="utf-8"))
    assert evidence == actual
    assert evidence_path.read_text(encoding="utf-8") == canonical_json_dumps(actual)


def test_weak_candidate_returns_meaningful_infeasible_constraints(
    project_root: Path, tmp_path: Path
) -> None:
    """Ensure an analyzable weak design is infeasible instead of suppressed or treated as valid."""

    paths = _paths(project_root)
    case = json.loads(paths["case_path"].read_text(encoding="utf-8"))
    case["case_id"] = "weak-stage1-integration-fixture"
    case["source"] = "software failure-semantics fixture"
    case["sections_by_group"] = {str(group): "W6X8.5" for group in range(1, 6)}
    weak_path = tmp_path / "weak_case.json"
    weak_path.write_text(json.dumps(case), encoding="utf-8")
    paths["case_path"] = weak_path
    result = build_stage1_artifact(**paths)
    assert result.feasible is False
    assert result.failed_constraint_ids
    assert result.maximum_utilization > 1.0
    assert all(item.demand >= 0.0 and item.capacity > 0.0 for item in result.constraints)
