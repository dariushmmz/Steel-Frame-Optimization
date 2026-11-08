"""Layer 1/2 tests for Phase 7 FEMA, hierarchy, and interpolation formulas."""

from __future__ import annotations

from typing import Literal

import pytest

from steel_frame_optimization.analysis.hinges import FemaHingeLimits
from steel_frame_optimization.analysis.pushover import CurvePoint, interpolate_base_shear_n
from steel_frame_optimization.constraints.fema import (
    force_controlled_column_utilization,
    plastic_rotation_demand,
    rotation_capacity_for_level,
    story_drift_ratio,
)
from steel_frame_optimization.constraints.hierarchy import strong_column_weak_beam_check

pytestmark = pytest.mark.analytical


def _limits() -> FemaHingeLimits:
    """Return a minimal positive FEMA capacity set for formula tests."""

    return FemaHingeLimits(
        source_row="fixture",
        interpolation_fraction=0.0,
        plastic_rotation_a_rad=0.04,
        plastic_rotation_b_rad=0.06,
        residual_strength_ratio=0.2,
        io_rotation_rad=0.01,
        ls_rotation_rad=0.03,
        cp_rotation_rad=0.04,
    )


def _point(displacement: float, shear: float) -> CurvePoint:
    """Build a valid synthetic curve point for interpolation tests."""

    return CurvePoint(
        roof_displacement_m=displacement,
        base_shear_n=shear,
        lateral_load_factor_n=shear,
        story_displacements_m=(displacement,),
        story_drift_ratios=(displacement / 4.0,),
        maximum_constraint_translation_residual_m=0.0,
        lateral_equilibrium_relative_residual=0.0,
        vertical_equilibrium_relative_residual=0.0,
    )


def test_plastic_rotation_subtracts_corrected_elastic_rotation() -> None:
    """Plastic demand is total spring rotation less recoverable spring rotation."""

    assert plastic_rotation_demand(
        total_spring_rotation_rad=-0.03,
        spring_moment_nm=-200.0,
        initial_stiffness_nm_per_rad=20_000.0,
    ) == pytest.approx(0.02)
    assert (
        plastic_rotation_demand(
            total_spring_rotation_rad=0.005,
            spring_moment_nm=200.0,
            initial_stiffness_nm_per_rad=20_000.0,
        )
        == 0.0
    )


@pytest.mark.parametrize(("level", "expected"), (("IO", 0.01), ("LS", 0.03), ("CP", 0.04)))
def test_rotation_capacity_selects_the_requested_fema_level(
    level: Literal["IO", "LS", "CP"], expected: float
) -> None:
    """IO, LS, and CP select distinct configured FEMA capacities."""

    assert rotation_capacity_for_level(_limits(), level) == expected


def test_force_controlled_column_implements_reference_equation_29() -> None:
    """The force-controlled utilization is P/Pcl plus M/Mn."""

    assert force_controlled_column_utilization(
        compression_force_n=400.0,
        pcl_n=1_000.0,
        moment_nm=150.0,
        nominal_moment_nm=500.0,
    ) == pytest.approx(0.7)


def test_story_drift_uses_relative_not_absolute_floor_displacement() -> None:
    """Rigid translation common to adjacent floors does not create interstory drift."""

    assert story_drift_ratio(
        upper_displacement_m=0.15,
        lower_displacement_m=0.10,
        height_m=4.0,
    ) == pytest.approx(0.0125)


def test_hierarchy_boundary_is_strictly_greater_than_one() -> None:
    """Equality fails REF-A Eq. (31), while any positive excess passes."""

    equal = strong_column_weak_beam_check(
        joint_id="joint",
        column_capacities_nm={"c1": 100.0, "c2": 100.0},
        beam_capacities_nm={"b1": 200.0},
    )
    greater = strong_column_weak_beam_check(
        joint_id="joint",
        column_capacities_nm={"c1": 100.0, "c2": 100.000001},
        beam_capacities_nm={"b1": 200.0},
    )
    assert equal.ratio == 1.0
    assert not equal.passed
    assert greater.passed


def test_pushover_interpolation_is_exact_for_linear_segment() -> None:
    """Target extraction linearly interpolates between adjacent committed points."""

    curve = (_point(0.0, 0.0), _point(0.1, 1_000.0), _point(0.2, 1_500.0))
    assert interpolate_base_shear_n(curve, 0.04) == pytest.approx(400.0)
    assert interpolate_base_shear_n(curve, 0.2) == 1_500.0


def test_pushover_interpolation_rejects_extrapolation() -> None:
    """No response is silently extrapolated beyond the converged curve."""

    curve = (_point(0.0, 0.0), _point(0.1, 1_000.0))
    with pytest.raises(ValueError, match="outside"):
        interpolate_base_shear_n(curve, 0.2)
