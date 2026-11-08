"""Check preliminary-filter equations and the fixed REF-A removal-count gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from steel_frame_optimization.constraints.lrfd import (
    critical_compression_stress_pa,
    nominal_compression_strength_n,
)
from steel_frame_optimization.models.frame import FrameDefinition
from steel_frame_optimization.sections.filtering import (
    PreliminaryFilterPolicy,
    build_preliminary_filter_audit,
)
from steel_frame_optimization.sections.repository import SectionRepository


@pytest.mark.analytical
def test_compression_curve_uses_both_printed_branches() -> None:
    """Check both REF-A Eq. (13) branches independently on either side of 1.5."""

    yield_strength = 345_000_000.0
    assert critical_compression_stress_pa(
        nondimensional_slenderness=1.0, yield_strength_pa=yield_strength
    ) == pytest.approx(0.658 * yield_strength, rel=1e-12)
    assert critical_compression_stress_pa(
        nondimensional_slenderness=2.0, yield_strength_pa=yield_strength
    ) == pytest.approx(0.877 / 4.0 * yield_strength, rel=1e-12)


@pytest.mark.analytical
def test_compression_curve_includes_the_1_5_boundary_in_inelastic_branch() -> None:
    """Freeze the paper's less-than-or-equal branch dispatch at lambda_c = 1.5."""

    yield_strength = 345_000_000.0
    expected = 0.658 ** (1.5**2) * yield_strength
    assert critical_compression_stress_pa(
        nondimensional_slenderness=1.5, yield_strength_pa=yield_strength
    ) == pytest.approx(expected, rel=1e-12)


@pytest.mark.analytical
def test_nominal_compression_strength_hand_fixture() -> None:
    """Compare the composed Pn calculation with a direct hand-style expression."""

    area = 0.01
    length = 4.0
    radius = 0.1
    yield_strength = 345_000_000.0
    elastic_modulus = 200_000_000_000.0
    pi = 3.141592653589793
    slenderness = length / (radius * pi) * (yield_strength / elastic_modulus) ** 0.5
    expected = area * 0.658 ** (slenderness**2) * yield_strength
    assert nominal_compression_strength_n(
        area_m2=area,
        effective_length_factor=1.0,
        length_m=length,
        radius_of_gyration_m=radius,
        yield_strength_pa=yield_strength,
        elastic_modulus_pa=elastic_modulus,
    ) == pytest.approx(expected, rel=1e-12)


def _filter_counts(project_root: Path) -> tuple[dict[str, int], dict[str, int]]:
    """Return approved project counts and immutable published comparison counts."""

    audit = build_preliminary_filter_audit(
        frame=FrameDefinition.from_json(project_root / "configs/frames/three_story_frame.json"),
        repository=SectionRepository.from_json(
            project_root / "data/processed/aisc_v14_1_w_shapes.json"
        ),
        policy=PreliminaryFilterPolicy.from_json(
            project_root / "configs/analysis/phase5_preliminary_filter.json"
        ),
    )
    observed = {group.group_id: group.actual_removed_count for group in audit.groups}
    expected = {group.group_id: group.published_removed_count for group in audit.groups}
    return observed, expected


@pytest.mark.analytical
@pytest.mark.regression
def test_preliminary_filter_removals_match_approved_project_baseline(project_root: Path) -> None:
    """Require the independently audited 273-row project baseline exactly."""

    observed, _ = _filter_counts(project_root)
    assert observed == {"1": 3, "2": 14, "3": 45, "4": 45, "5": 41}


@pytest.mark.analytical
@pytest.mark.regression
def test_published_filter_counts_remain_visible_as_not_reproduced(project_root: Path) -> None:
    """Preserve REF-A's immutable historical values without claiming they were reproduced."""

    observed, published = _filter_counts(project_root)
    assert published == {"1": 16, "2": 32, "3": 42, "4": 42, "5": 38}
    assert observed != published
