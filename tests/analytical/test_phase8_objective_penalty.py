"""Layer 1/2 tests for REF-A objective and ASM-016 penalty formulas."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from steel_frame_optimization.objective.functions import (
    calculate_objective,
    drift_uniformity_term,
    structural_weight_n,
)
from steel_frame_optimization.objective.penalties import (
    PenaltyPolicy,
    calculate_penalty_score,
    normalized_violation,
)
from steel_frame_optimization.sections.repository import SectionRepository

pytestmark = pytest.mark.analytical


@pytest.fixture
def penalty_policy(project_root: Path) -> PenaltyPolicy:
    """Load the frozen Phase 8 penalty policy."""

    import json

    content = json.loads(
        (project_root / "configs" / "analysis" / "phase8_objective_penalty_policy.json").read_text(
            encoding="utf-8"
        )
    )
    return PenaltyPolicy.model_validate(content["penalty"])


def test_uniform_linear_displacement_profile_has_zero_uniformity() -> None:
    """A displacement profile proportional to elevation has no shape deviation."""

    assert drift_uniformity_term(
        floor_displacements_m=(0.02, 0.04, 0.06),
        floor_elevations_m=(4.0, 8.0, 12.0),
    ) == pytest.approx(0.0, abs=1e-15)


def test_uniformity_retains_ns_divisor_while_excluding_roof() -> None:
    """The printed ns divisor is not silently replaced by ns minus one."""

    actual = drift_uniformity_term(
        floor_displacements_m=(0.01, 0.03, 0.06),
        floor_elevations_m=(4.0, 8.0, 12.0),
    )
    expected = math.sqrt((0.5**2 + 0.25**2) / 3.0)
    alternative = math.sqrt((0.5**2 + 0.25**2) / 2.0)
    assert actual == pytest.approx(expected)
    assert actual != pytest.approx(alternative)


def test_structural_weight_uses_group_lengths_and_section_line_weights(
    project_root: Path,
) -> None:
    """Candidate weight is independent of modal and gravity load accounting."""

    repository = SectionRepository.from_json(
        project_root / "data" / "processed" / "aisc_v14_1_w_shapes.json"
    )
    sections = {"1": repository.get("W16X77"), "2": repository.get("W21X44")}
    actual = structural_weight_n(
        total_member_length_m_by_group={"1": 10.0, "2": 20.0},
        sections_by_group=sections,
    )
    expected = 10.0 * sections["1"].weight_n_per_m + 20.0 * sections["2"].weight_n_per_m
    assert actual == pytest.approx(expected)


def test_weighted_objective_exposes_every_component() -> None:
    """The raw objective equals its separately persisted 0.95 and 0.05 contributions."""

    result = calculate_objective(
        structural_weight=100.0,
        maximum_structural_weight=1_000.0,
        uniformity_term=0.2,
        weight_coefficient=0.95,
        uniformity_coefficient=0.05,
    )
    assert result.normalized_weight == pytest.approx(0.1)
    assert result.weight_contribution == pytest.approx(0.095)
    assert result.uniformity_contribution == pytest.approx(0.01)
    assert result.raw_objective == pytest.approx(0.105)


def test_normalized_violation_and_severity_are_proportional(
    penalty_policy: PenaltyPolicy,
) -> None:
    """A 20% strength exceedance receives exactly coefficient 100 times 0.2."""

    violation = normalized_violation(
        check_id="strength",
        stage="stage1",
        category="strength",
        utilization=1.2,
        failed=True,
        policy=penalty_policy,
    )
    assert violation is not None
    assert violation.normalized_exceedance == pytest.approx(0.2)
    assert violation.weighted_penalty == pytest.approx(20.0)


def test_strict_boundary_failure_is_retained_with_zero_exceedance(
    penalty_policy: PenaltyPolicy,
) -> None:
    """A strict constraint failing at utilization one remains visible and offset-infeasible."""

    violation = normalized_violation(
        check_id="hierarchy-equality",
        stage="stage2",
        category="capacity_hierarchy",
        utilization=1.0,
        failed=True,
        policy=penalty_policy,
    )
    assert violation is not None
    assert violation.normalized_exceedance == 0.0
    assert violation.weighted_penalty == 0.0


def test_feasibility_offset_guarantees_rank_separation(
    penalty_policy: PenaltyPolicy,
) -> None:
    """Even an arbitrarily large feasible objective ranks ahead of zero-exceedance infeasibility."""

    feasible = calculate_penalty_score(
        objective_base=1_000_000.0,
        objective_basis="full_objective",
        feasible=True,
        violations=(),
        policy=penalty_policy,
    )
    boundary = normalized_violation(
        check_id="boundary",
        stage="stage2",
        category="capacity_hierarchy",
        utilization=1.0,
        failed=True,
        policy=penalty_policy,
    )
    assert boundary is not None
    infeasible = calculate_penalty_score(
        objective_base=0.0,
        objective_basis="weight_contribution_only",
        feasible=False,
        violations=(boundary,),
        policy=penalty_policy,
    )
    assert feasible.ranking_score < 1.0
    assert infeasible.ranking_score >= 1.0
    assert feasible.ranking_score < infeasible.ranking_score
