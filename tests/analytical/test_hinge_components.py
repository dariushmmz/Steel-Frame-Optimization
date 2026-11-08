"""Analytical checks for Phase 6 hinge strength, stiffness, and FEMA boundaries."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from steel_frame_optimization.analysis.hinges import (
    ColumnHingeMode,
    HingeControl,
    adjusted_member_inertia,
    build_rotational_hinge_parameters,
    classify_hinge_control,
    compression_from_tension_positive_force,
    corrected_spring_hardening_ratio,
    corrected_spring_stiffness,
    expected_column_plastic_moment_nm,
    fema_356_hinge_limits,
    lower_bound_column_compression_strength_n,
    member_double_curvature_stiffness,
)
from steel_frame_optimization.sections.repository import SectionRepository

pytestmark = pytest.mark.analytical

ELASTIC_MODULUS_PA = 200_000_000_000.0
BEAM_FY_PA = 248_000_000.0
BEAM_FYE_PA = 339_000_000.0
COLUMN_FY_PA = 345_000_000.0
COLUMN_FYE_PA = 397_000_000.0


def _repository(project_root: Path) -> SectionRepository:
    """Load the approved project section baseline."""

    return SectionRepository.from_json(project_root / "data/processed/aisc_v14_1_w_shapes.json")


def test_series_correction_recovers_initial_and_postyield_member_stiffness() -> None:
    """Check REF-A Eqs. (3)-(10) independently using springs-in-series mechanics."""

    member_stiffness = member_double_curvature_stiffness(
        elastic_modulus_pa=ELASTIC_MODULUS_PA, inertia_m4=0.0004, length_m=4.0
    )
    adjusted_inertia_value = adjusted_member_inertia(inertia_m4=0.0004, stiffness_multiplier_n=10.0)
    elastic_element_stiffness = member_stiffness * adjusted_inertia_value / 0.0004
    spring_stiffness = corrected_spring_stiffness(
        member_stiffness_nm_per_rad=member_stiffness, stiffness_multiplier_n=10.0
    )
    combined_initial = 1.0 / (1.0 / elastic_element_stiffness + 1.0 / spring_stiffness)
    assert combined_initial == pytest.approx(member_stiffness, rel=1e-12)

    spring_hardening = corrected_spring_hardening_ratio(
        member_hardening_ratio=0.03, stiffness_multiplier_n=10.0
    )
    combined_postyield = 1.0 / (
        1.0 / elastic_element_stiffness + 1.0 / (spring_hardening * spring_stiffness)
    )
    assert combined_postyield == pytest.approx(0.03 * member_stiffness, rel=1e-12)


def test_column_moment_reduction_uses_paper_cap_and_zero_boundary() -> None:
    """Check REF-A Eq. (1) at its cap transition and axial-yield boundary."""

    plastic_modulus = 0.002
    expected_yield = 397_000_000.0
    axial_yield = 4_000_000.0
    unreduced = plastic_modulus * expected_yield
    cap_transition = axial_yield * (1.0 - 1.0 / 1.18)
    assert expected_column_plastic_moment_nm(
        plastic_modulus_m3=plastic_modulus,
        expected_yield_strength_pa=expected_yield,
        compression_force_n=0.0,
        expected_axial_yield_force_n=axial_yield,
    ) == pytest.approx(unreduced)
    assert expected_column_plastic_moment_nm(
        plastic_modulus_m3=plastic_modulus,
        expected_yield_strength_pa=expected_yield,
        compression_force_n=cap_transition,
        expected_axial_yield_force_n=axial_yield,
    ) == pytest.approx(unreduced)
    assert (
        expected_column_plastic_moment_nm(
            plastic_modulus_m3=plastic_modulus,
            expected_yield_strength_pa=expected_yield,
            compression_force_n=axial_yield,
            expected_axial_yield_force_n=axial_yield,
        )
        == 0.0
    )


def test_axial_ratio_control_boundaries_are_explicit() -> None:
    """Keep 0.2 deformation controlled and make 0.5 force controlled."""

    assert classify_hinge_control(member_type="column", axial_ratio=0.2) is HingeControl.DEFORMATION
    assert (
        classify_hinge_control(member_type="column", axial_ratio=math.nextafter(0.5, 0.0))
        is HingeControl.DEFORMATION
    )
    assert classify_hinge_control(member_type="column", axial_ratio=0.5) is HingeControl.FORCE
    assert classify_hinge_control(member_type="beam", axial_ratio=99.0) is HingeControl.DEFORMATION


def test_tension_positive_force_conversion_never_reduces_capacity_in_tension() -> None:
    """Convert compression to a positive magnitude while mapping tension to zero."""

    assert compression_from_tension_positive_force(-123.0) == 123.0
    assert compression_from_tension_positive_force(0.0) == 0.0
    assert compression_from_tension_positive_force(456.0) == 0.0


def test_fema_beam_table_rows_and_interpolation(project_root: Path) -> None:
    """Check compact, slender, and halfway Table 5-6 beam multipliers."""

    section = _repository(project_root).get("W24X55")
    theta_y = 0.01
    strength_ksi = BEAM_FYE_PA / 6_894_757.293168
    root = math.sqrt(strength_ksi)
    compact = section.model_copy(
        update={"flange_slenderness": 52.0 / root, "web_slenderness": 418.0 / root}
    )
    slender = section.model_copy(
        update={"flange_slenderness": 65.0 / root, "web_slenderness": 640.0 / root}
    )
    halfway = section.model_copy(
        update={
            "flange_slenderness": 0.5 * (52.0 + 65.0) / root,
            "web_slenderness": 0.5 * (418.0 + 640.0) / root,
        }
    )
    compact_limits = fema_356_hinge_limits(
        compact,
        member_type="beam",
        expected_yield_strength_pa=BEAM_FYE_PA,
        fema_yield_rotation_rad=theta_y,
    )
    slender_limits = fema_356_hinge_limits(
        slender,
        member_type="beam",
        expected_yield_strength_pa=BEAM_FYE_PA,
        fema_yield_rotation_rad=theta_y,
    )
    halfway_limits = fema_356_hinge_limits(
        halfway,
        member_type="beam",
        expected_yield_strength_pa=BEAM_FYE_PA,
        fema_yield_rotation_rad=theta_y,
    )
    assert compact_limits.cp_rotation_rad == pytest.approx(8.0 * theta_y)
    assert compact_limits.plastic_rotation_a_rad == pytest.approx(9.0 * theta_y)
    assert slender_limits.cp_rotation_rad == pytest.approx(3.0 * theta_y)
    assert slender_limits.residual_strength_ratio == pytest.approx(0.2)
    assert halfway_limits.interpolation_fraction == pytest.approx(0.5)
    assert halfway_limits.cp_rotation_rad == pytest.approx(5.5 * theta_y)


def test_fema_column_axial_branches_and_force_control(project_root: Path) -> None:
    """Check Table 5-6's low-, intermediate-, and force-controlled column regimes."""

    section = _repository(project_root).get("W16X57")
    theta_y = 0.01
    low = fema_356_hinge_limits(
        section,
        member_type="column",
        expected_yield_strength_pa=COLUMN_FYE_PA,
        fema_yield_rotation_rad=theta_y,
        axial_ratio=math.nextafter(0.2, 0.0),
    )
    intermediate = fema_356_hinge_limits(
        section,
        member_type="column",
        expected_yield_strength_pa=COLUMN_FYE_PA,
        fema_yield_rotation_rad=theta_y,
        axial_ratio=0.2,
    )
    force = fema_356_hinge_limits(
        section,
        member_type="column",
        expected_yield_strength_pa=COLUMN_FYE_PA,
        fema_yield_rotation_rad=theta_y,
        axial_ratio=0.5,
    )
    assert "P/Pcl < 0.2" in low.source_row
    assert "0.2 <= P/Pcl < 0.5" in intermediate.source_row
    assert force.cp_rotation_rad is None
    assert force.plastic_rotation_a_rad is None


def test_preliminary_and_final_column_parameters_are_distinct(project_root: Path) -> None:
    """Require preliminary hinges to ignore P while final hinges use P/Pcl and Eq. (1)."""

    section = _repository(project_root).get("W16X57")
    length = 3.9624
    pcl = lower_bound_column_compression_strength_n(
        section,
        effective_length_factor=1.0,
        length_m=length,
        nominal_yield_strength_pa=COLUMN_FY_PA,
        elastic_modulus_pa=ELASTIC_MODULUS_PA,
    )
    compression = 0.35 * pcl
    preliminary = build_rotational_hinge_parameters(
        section,
        member_type="column",
        member_length_m=length,
        elastic_modulus_pa=ELASTIC_MODULUS_PA,
        nominal_yield_strength_pa=COLUMN_FY_PA,
        expected_yield_strength_pa=COLUMN_FYE_PA,
        column_mode=ColumnHingeMode.PRELIMINARY_UNREDUCED,
        compression_force_n=compression,
    )
    final = build_rotational_hinge_parameters(
        section,
        member_type="column",
        member_length_m=length,
        elastic_modulus_pa=ELASTIC_MODULUS_PA,
        nominal_yield_strength_pa=COLUMN_FY_PA,
        expected_yield_strength_pa=COLUMN_FYE_PA,
        column_mode=ColumnHingeMode.FINAL_AXIAL_REDUCED,
        compression_force_n=compression,
    )
    assert preliminary.compression_force_n == 0.0
    assert preliminary.axial_ratio_p_over_pcl == 0.0
    assert preliminary.expected_plastic_moment_nm == pytest.approx(section.zx_m3 * COLUMN_FYE_PA)
    assert final.axial_ratio_p_over_pcl == pytest.approx(0.35)
    assert final.expected_plastic_moment_nm < preliminary.expected_plastic_moment_nm
    assert final.expected_axial_yield_force_n == pytest.approx(section.area_m2 * COLUMN_FY_PA)
    assert final.fema_yield_rotation_rad < preliminary.fema_yield_rotation_rad


def test_force_controlled_final_column_creates_no_rotation_limits(project_root: Path) -> None:
    """Ensure a final column at P/Pcl=0.5 cannot be accepted by plastic rotation."""

    section = _repository(project_root).get("W16X57")
    length = 3.9624
    pcl = lower_bound_column_compression_strength_n(
        section,
        effective_length_factor=1.0,
        length_m=length,
        nominal_yield_strength_pa=COLUMN_FY_PA,
        elastic_modulus_pa=ELASTIC_MODULUS_PA,
    )
    parameters = build_rotational_hinge_parameters(
        section,
        member_type="column",
        member_length_m=length,
        elastic_modulus_pa=ELASTIC_MODULUS_PA,
        nominal_yield_strength_pa=COLUMN_FY_PA,
        expected_yield_strength_pa=COLUMN_FYE_PA,
        column_mode=ColumnHingeMode.FINAL_AXIAL_REDUCED,
        compression_force_n=0.5 * pcl,
    )
    assert parameters.control is HingeControl.FORCE
    assert parameters.fema_limits.io_rotation_rad is None
    assert parameters.fema_limits.cp_rotation_rad is None
