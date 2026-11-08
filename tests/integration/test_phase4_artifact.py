"""Integration checks for deterministic Phase 4 evidence and frozen analysis policy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from steel_frame_optimization.serialization import canonical_json_dumps
from steel_frame_optimization.validation.phase4 import (
    Phase4Artifact,
    PublishedElasticCase,
    build_phase4_artifact,
)

pytestmark = pytest.mark.integration


def _paths(project_root: Path) -> tuple[Path, Path, Path]:
    """Return the three explicit inputs for the published Case 2 evidence run."""

    return (
        project_root / "configs" / "frames" / "three_story_frame.json",
        project_root / "data" / "processed" / "aisc_v14_1_w_shapes.json",
        project_root / "configs" / "cases" / "three_story_case2_gwo.json",
    )


def test_phase4_artifact_is_deterministic_and_matches_committed_evidence(
    project_root: Path,
) -> None:
    """Run every ±E combination directly twice and require canonical persisted equality."""

    first = build_phase4_artifact(*_paths(project_root))
    second = build_phase4_artifact(*_paths(project_root))
    assert first == second
    assert first.positive_e.load_stages == (
        "gravity_converged_and_held",
        "lateral_converged",
    )
    assert first.positive_e.combination_evaluated_directly is True
    assert first.negative_e.combination_evaluated_directly is True
    assert first.positive_e.equilibrium_relative_residual <= 1e-8
    assert first.negative_e.equilibrium_relative_residual <= 1e-8
    assert first.negative_e.roof_displacement_m == pytest.approx(
        -first.positive_e.roof_displacement_m, rel=1e-12
    )
    assert first.case.analysis_policy.pdelta_refinement_sequence == (1, 2, 4, 8, 16, 32)
    assert first.case.analysis_policy.exact_beam_column_relative_tolerance == 1e-4
    assert first.case.analysis_policy.minimum_superposition_error == 0.05
    assert 0.0 < first.column_p_over_pcr_min <= first.column_p_over_pcr_max < 1.0

    evidence_path = project_root / "docs" / "evidence" / "phase04" / "case2_gwo_elastic.json"
    evidence = Phase4Artifact.model_validate_json(evidence_path.read_text(encoding="utf-8"))
    assert evidence == first
    assert evidence_path.read_text(encoding="utf-8") == canonical_json_dumps(first)


def test_phase4_case_rejects_changed_frozen_tolerance(project_root: Path) -> None:
    """Prevent later code or configuration from loosening the owner-frozen benchmark tolerance."""

    case_path = _paths(project_root)[2]
    content = json.loads(case_path.read_text(encoding="utf-8"))
    content["analysis_policy"]["numerical_relative_tolerance"] = 1e-5
    with pytest.raises(ValidationError, match="frozen"):
        PublishedElasticCase.model_validate(content)
