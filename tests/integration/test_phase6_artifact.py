"""Regression checks for the deterministic Phase 6 component evidence artifact."""

from __future__ import annotations

from pathlib import Path

import pytest

from steel_frame_optimization.models.failures import FailureCategory
from steel_frame_optimization.serialization import canonical_json_dumps
from steel_frame_optimization.validation.phase6 import Phase6Artifact, build_phase6_artifact

pytestmark = [pytest.mark.integration, pytest.mark.regression]


def _build(project_root: Path) -> Phase6Artifact:
    """Build Phase 6 evidence from the approved section and policy inputs."""

    return build_phase6_artifact(
        sections_path=project_root / "data/processed/aisc_v14_1_w_shapes.json",
        policy_path=project_root / "configs/analysis/phase6_nonlinear_policy.json",
    )


def test_phase6_component_artifact_satisfies_numerical_invariants(project_root: Path) -> None:
    """Check series mechanics, monotonic response, refinement, and controlled failure."""

    artifact = _build(project_root)
    assert artifact.series_initial_stiffness_nm_per_rad == pytest.approx(
        artifact.beam_parameters.member_stiffness_nm_per_rad, rel=1e-12
    )
    assert artifact.series_postyield_stiffness_nm_per_rad == pytest.approx(
        0.03 * artifact.beam_parameters.member_stiffness_nm_per_rad, rel=1e-12
    )
    assert max(point.relative_error for point in artifact.monotonic_points) <= 1e-8
    assert artifact.maximum_refinement_spread_rad <= 1e-12 * max(
        abs(value) for value in artifact.refinement_endpoint_rotations_rad.values()
    )
    assert artifact.controlled_failure.converged is False
    assert artifact.controlled_failure.failure is not None
    assert artifact.controlled_failure.failure.category is FailureCategory.NUMERICAL_NONCONVERGENCE


def test_phase6_artifact_is_repeatable_and_matches_evidence(project_root: Path) -> None:
    """Require repeatable calculation and byte-canonical committed evidence."""

    first = _build(project_root)
    second = _build(project_root)
    assert first == second
    evidence_path = project_root / "docs/evidence/phase06/component_benchmarks.json"
    evidence = Phase6Artifact.model_validate_json(evidence_path.read_text(encoding="utf-8"))
    assert evidence == first
    assert evidence_path.read_text(encoding="utf-8") == canonical_json_dumps(first)
    assert evidence_path.read_bytes() == canonical_json_dumps(first).encode("utf-8")
