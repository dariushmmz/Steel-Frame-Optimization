"""Define source-traceable concentrated rotational hinges for planar steel members.

The formulas implement REF-A Eqs. (1)--(10) and the FEMA 356 fully restrained
moment-frame component model. They remain independent of any particular frame so the
same components can serve the three- and nine-story models.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal

import openseespy.opensees as ops
from pydantic import Field

from steel_frame_optimization.config.schemas import StrictConfigModel
from steel_frame_optimization.constraints.lrfd import (
    nominal_compression_strength_with_local_buckling,
)
from steel_frame_optimization.sections.records import SectionRecord

KSI_TO_PA = 6_894_757.293168


class ColumnHingeMode(StrEnum):
    """Identify the paper's preliminary or axial-force-corrected column model."""

    PRELIMINARY_UNREDUCED = "preliminary_unreduced"
    FINAL_AXIAL_REDUCED = "final_axial_reduced"


class HingeControl(StrEnum):
    """Classify whether hinge rotation or member force governs acceptance."""

    DEFORMATION = "deformation_controlled"
    FORCE = "force_controlled"


class FemaHingeLimits(StrictConfigModel):
    """Store FEMA 356 Table 5-6 plastic-rotation parameters and acceptance limits."""

    source_row: str = Field(min_length=1)
    interpolation_fraction: float = Field(ge=0.0, le=1.0)
    plastic_rotation_a_rad: float | None = Field(default=None, ge=0.0)
    plastic_rotation_b_rad: float | None = Field(default=None, ge=0.0)
    residual_strength_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    io_rotation_rad: float | None = Field(default=None, ge=0.0)
    ls_rotation_rad: float | None = Field(default=None, ge=0.0)
    cp_rotation_rad: float | None = Field(default=None, ge=0.0)


class RotationalHingeParameters(StrictConfigModel):
    """Persist every value needed to create and audit one rotational spring."""

    member_type: Literal["beam", "column"]
    column_mode: ColumnHingeMode | None
    control: HingeControl
    elastic_modulus_pa: float = Field(gt=0.0, allow_inf_nan=False)
    member_inertia_m4: float = Field(gt=0.0, allow_inf_nan=False)
    member_length_m: float = Field(gt=0.0, allow_inf_nan=False)
    stiffness_multiplier_n: float = Field(gt=1.0, allow_inf_nan=False)
    member_stiffness_nm_per_rad: float = Field(gt=0.0, allow_inf_nan=False)
    adjusted_member_inertia_m4: float = Field(gt=0.0, allow_inf_nan=False)
    spring_initial_stiffness_nm_per_rad: float = Field(gt=0.0, allow_inf_nan=False)
    member_hardening_ratio: float = Field(ge=0.0, lt=1.0, allow_inf_nan=False)
    spring_hardening_ratio: float = Field(ge=0.0, lt=1.0, allow_inf_nan=False)
    expected_plastic_moment_nm: float = Field(gt=0.0, allow_inf_nan=False)
    spring_yield_rotation_rad: float = Field(gt=0.0, allow_inf_nan=False)
    fema_yield_rotation_rad: float = Field(gt=0.0, allow_inf_nan=False)
    compression_force_n: float = Field(ge=0.0, allow_inf_nan=False)
    expected_axial_yield_force_n: float = Field(gt=0.0, allow_inf_nan=False)
    lower_bound_compression_strength_n: float | None = Field(default=None, gt=0.0)
    axial_ratio_p_over_pcl: float | None = Field(default=None, ge=0.0)
    fema_limits: FemaHingeLimits


class HingedMemberTags(StrictConfigModel):
    """Return deterministic OpenSees tags created for one two-ended hinged member."""

    member_element_tag: int = Field(gt=0)
    i_spring_element_tag: int = Field(gt=0)
    j_spring_element_tag: int = Field(gt=0)
    i_internal_node_tag: int = Field(gt=0)
    j_internal_node_tag: int = Field(gt=0)


def compression_from_tension_positive_force(axial_force_n: float) -> float:
    """Convert the project tension-positive axial sign to a nonnegative compression demand."""

    if not math.isfinite(axial_force_n):
        raise ValueError("stored axial force must be finite")
    return max(-axial_force_n, 0.0)


def member_double_curvature_stiffness(
    *, elastic_modulus_pa: float, inertia_m4: float, length_m: float
) -> float:
    """Return ``6EI/L`` from REF-A Eq. (3), in moment per radian."""

    values = (elastic_modulus_pa, inertia_m4, length_m)
    if any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("elastic modulus, inertia, and length must be finite and positive")
    return 6.0 * elastic_modulus_pa * inertia_m4 / length_m


def adjusted_member_inertia(*, inertia_m4: float, stiffness_multiplier_n: float) -> float:
    """Return the elastic-element inertia multiplier ``(n+1)/n`` from REF-A Eq. (6)."""

    if inertia_m4 <= 0.0 or not math.isfinite(inertia_m4):
        raise ValueError("member inertia must be finite and positive")
    if stiffness_multiplier_n <= 1.0 or not math.isfinite(stiffness_multiplier_n):
        raise ValueError("stiffness multiplier n must be finite and greater than one")
    return inertia_m4 * (stiffness_multiplier_n + 1.0) / stiffness_multiplier_n


def corrected_spring_stiffness(
    *, member_stiffness_nm_per_rad: float, stiffness_multiplier_n: float
) -> float:
    """Return ``(n+1) Kmember`` from REF-A Eq. (7)."""

    if member_stiffness_nm_per_rad <= 0.0 or not math.isfinite(member_stiffness_nm_per_rad):
        raise ValueError("member stiffness must be finite and positive")
    if stiffness_multiplier_n <= 1.0 or not math.isfinite(stiffness_multiplier_n):
        raise ValueError("stiffness multiplier n must be finite and greater than one")
    return (stiffness_multiplier_n + 1.0) * member_stiffness_nm_per_rad


def corrected_spring_hardening_ratio(
    *, member_hardening_ratio: float, stiffness_multiplier_n: float
) -> float:
    """Return the spring hardening ratio from REF-A Eq. (10)."""

    if not 0.0 <= member_hardening_ratio < 1.0 or not math.isfinite(member_hardening_ratio):
        raise ValueError("member hardening ratio must be finite in [0, 1)")
    if stiffness_multiplier_n <= 1.0 or not math.isfinite(stiffness_multiplier_n):
        raise ValueError("stiffness multiplier n must be finite and greater than one")
    return member_hardening_ratio / (stiffness_multiplier_n * (1.0 - member_hardening_ratio) + 1.0)


def expected_column_plastic_moment_nm(
    *,
    plastic_modulus_m3: float,
    expected_yield_strength_pa: float,
    compression_force_n: float,
    expected_axial_yield_force_n: float,
) -> float:
    """Return the compression-reduced column moment from REF-A Eq. (1)."""

    values = (plastic_modulus_m3, expected_yield_strength_pa, expected_axial_yield_force_n)
    if any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("section, material, and axial-yield inputs must be finite and positive")
    if compression_force_n < 0.0 or not math.isfinite(compression_force_n):
        raise ValueError("compression force must be finite and nonnegative")
    unreduced = plastic_modulus_m3 * expected_yield_strength_pa
    reduced = 1.18 * unreduced * (1.0 - compression_force_n / expected_axial_yield_force_n)
    return max(0.0, min(reduced, unreduced))


def classify_hinge_control(
    *, member_type: Literal["beam", "column"], axial_ratio: float
) -> HingeControl:
    """Classify FEMA deformation/force control using the ``P/Pcl = 0.5`` boundary."""

    if axial_ratio < 0.0 or not math.isfinite(axial_ratio):
        raise ValueError("axial ratio must be finite and nonnegative")
    if member_type == "beam":
        return HingeControl.DEFORMATION
    return HingeControl.DEFORMATION if axial_ratio < 0.5 else HingeControl.FORCE


def lower_bound_column_compression_strength_n(
    section: SectionRecord,
    *,
    effective_length_factor: float,
    length_m: float,
    nominal_yield_strength_pa: float,
    elastic_modulus_pa: float,
) -> float:
    """Return FEMA ``Pcl`` using AISC E3/E7 with ``phi=1`` and project major-axis policy."""

    result = nominal_compression_strength_with_local_buckling(
        section,
        effective_length_factor=effective_length_factor,
        length_m=length_m,
        radius_of_gyration_m=section.rx_m,
        yield_strength_pa=nominal_yield_strength_pa,
        elastic_modulus_pa=elastic_modulus_pa,
    )
    return result.nominal_strength_n


def _interpolation_fraction(value: float, compact_limit: float, slender_limit: float) -> float:
    """Return the bounded FEMA interpolation coordinate between table rows a and b."""

    if value <= compact_limit:
        return 0.0
    if value >= slender_limit:
        return 1.0
    return (value - compact_limit) / (slender_limit - compact_limit)


def _interpolate(compact: float, slender: float, fraction: float) -> float:
    """Linearly interpolate one FEMA table parameter."""

    return compact + fraction * (slender - compact)


def fema_356_hinge_limits(
    section: SectionRecord,
    *,
    member_type: Literal["beam", "column"],
    expected_yield_strength_pa: float,
    fema_yield_rotation_rad: float,
    axial_ratio: float = 0.0,
) -> FemaHingeLimits:
    """Return FEMA 356 Table 5-6 primary-action limits with required interpolation."""

    if expected_yield_strength_pa <= 0.0 or not math.isfinite(expected_yield_strength_pa):
        raise ValueError("expected yield strength must be finite and positive")
    if fema_yield_rotation_rad <= 0.0 or not math.isfinite(fema_yield_rotation_rad):
        raise ValueError("FEMA yield rotation must be finite and positive")
    control = classify_hinge_control(member_type=member_type, axial_ratio=axial_ratio)
    if control is HingeControl.FORCE:
        return FemaHingeLimits(
            source_row="FEMA 356 Table 5-6 force-controlled column P/Pcl >= 0.5",
            interpolation_fraction=0.0,
        )

    strength_ksi = expected_yield_strength_pa / KSI_TO_PA
    root = math.sqrt(strength_ksi)
    if member_type == "beam":
        compact_web, slender_web = 418.0 / root, 640.0 / root
        compact = (9.0, 11.0, 0.6, 1.0, 6.0, 8.0)
        slender = (4.0, 6.0, 0.2, 0.25, 2.0, 3.0)
        source_row = "FEMA 356 Table 5-6 beams-flexure"
    elif axial_ratio < 0.2:
        compact_web, slender_web = 300.0 / root, 460.0 / root
        compact = (9.0, 11.0, 0.6, 1.0, 6.0, 8.0)
        slender = (4.0, 6.0, 0.2, 0.25, 2.0, 3.0)
        source_row = "FEMA 356 Table 5-6 columns-flexure P/Pcl < 0.2"
    else:
        compact_web, slender_web = 260.0 / root, 400.0 / root
        reduction = 1.0 - 1.7 * axial_ratio
        compact = (
            11.0 * reduction,
            17.0 * reduction,
            0.2,
            0.25,
            8.0 * reduction,
            11.0 * reduction,
        )
        slender = (1.0, 1.5, 0.2, 0.25, 0.5, 0.8)
        source_row = "FEMA 356 Table 5-6 columns-flexure 0.2 <= P/Pcl < 0.5"

    flange_fraction = _interpolation_fraction(section.flange_slenderness, 52.0 / root, 65.0 / root)
    web_fraction = _interpolation_fraction(section.web_slenderness, compact_web, slender_web)
    fraction = max(flange_fraction, web_fraction)
    values = tuple(
        _interpolate(compact_value, slender_value, fraction)
        for compact_value, slender_value in zip(compact, slender, strict=True)
    )
    a, b, c, io, ls, cp = values
    return FemaHingeLimits(
        source_row=source_row,
        interpolation_fraction=fraction,
        plastic_rotation_a_rad=a * fema_yield_rotation_rad,
        plastic_rotation_b_rad=b * fema_yield_rotation_rad,
        residual_strength_ratio=c,
        io_rotation_rad=io * fema_yield_rotation_rad,
        ls_rotation_rad=ls * fema_yield_rotation_rad,
        cp_rotation_rad=cp * fema_yield_rotation_rad,
    )


def build_rotational_hinge_parameters(
    section: SectionRecord,
    *,
    member_type: Literal["beam", "column"],
    member_length_m: float,
    elastic_modulus_pa: float,
    nominal_yield_strength_pa: float,
    expected_yield_strength_pa: float,
    stiffness_multiplier_n: float = 10.0,
    member_hardening_ratio: float = 0.03,
    column_mode: ColumnHingeMode | None = None,
    compression_force_n: float = 0.0,
    effective_length_factor: float = 1.0,
) -> RotationalHingeParameters:
    """Build auditable beam or preliminary/final column hinge parameters."""

    if member_type == "beam" and column_mode is not None:
        raise ValueError("beam hinges cannot declare a column mode")
    if member_type == "column" and column_mode is None:
        raise ValueError("column hinges require an explicit preliminary or final mode")
    if compression_force_n < 0.0 or not math.isfinite(compression_force_n):
        raise ValueError("compression force must be finite and nonnegative")

    member_stiffness = member_double_curvature_stiffness(
        elastic_modulus_pa=elastic_modulus_pa,
        inertia_m4=section.ix_m4,
        length_m=member_length_m,
    )
    spring_stiffness = corrected_spring_stiffness(
        member_stiffness_nm_per_rad=member_stiffness,
        stiffness_multiplier_n=stiffness_multiplier_n,
    )
    expected_axial_yield = section.area_m2 * nominal_yield_strength_pa
    lower_bound: float | None = None
    axial_ratio: float | None = None
    effective_compression = 0.0
    if member_type == "column":
        lower_bound = lower_bound_column_compression_strength_n(
            section,
            effective_length_factor=effective_length_factor,
            length_m=member_length_m,
            nominal_yield_strength_pa=nominal_yield_strength_pa,
            elastic_modulus_pa=elastic_modulus_pa,
        )
        if column_mode is ColumnHingeMode.FINAL_AXIAL_REDUCED:
            effective_compression = compression_force_n
        axial_ratio = effective_compression / lower_bound

    unreduced_moment = section.zx_m3 * expected_yield_strength_pa
    expected_moment = (
        expected_column_plastic_moment_nm(
            plastic_modulus_m3=section.zx_m3,
            expected_yield_strength_pa=expected_yield_strength_pa,
            compression_force_n=effective_compression,
            expected_axial_yield_force_n=expected_axial_yield,
        )
        if member_type == "column"
        else unreduced_moment
    )
    if expected_moment <= 0.0:
        raise ValueError("compression eliminates positive column hinge moment capacity")

    yield_reduction = 1.0 - effective_compression / expected_axial_yield
    fema_yield_rotation = unreduced_moment / member_stiffness * max(yield_reduction, 0.0)
    if fema_yield_rotation <= 0.0:
        raise ValueError("compression eliminates positive FEMA column yield rotation")
    ratio_for_limits = axial_ratio if axial_ratio is not None else 0.0
    limits = fema_356_hinge_limits(
        section,
        member_type=member_type,
        expected_yield_strength_pa=expected_yield_strength_pa,
        fema_yield_rotation_rad=fema_yield_rotation,
        axial_ratio=ratio_for_limits,
    )
    return RotationalHingeParameters(
        member_type=member_type,
        column_mode=column_mode,
        control=classify_hinge_control(member_type=member_type, axial_ratio=ratio_for_limits),
        elastic_modulus_pa=elastic_modulus_pa,
        member_inertia_m4=section.ix_m4,
        member_length_m=member_length_m,
        stiffness_multiplier_n=stiffness_multiplier_n,
        member_stiffness_nm_per_rad=member_stiffness,
        adjusted_member_inertia_m4=adjusted_member_inertia(
            inertia_m4=section.ix_m4, stiffness_multiplier_n=stiffness_multiplier_n
        ),
        spring_initial_stiffness_nm_per_rad=spring_stiffness,
        member_hardening_ratio=member_hardening_ratio,
        spring_hardening_ratio=corrected_spring_hardening_ratio(
            member_hardening_ratio=member_hardening_ratio,
            stiffness_multiplier_n=stiffness_multiplier_n,
        ),
        expected_plastic_moment_nm=expected_moment,
        spring_yield_rotation_rad=expected_moment / spring_stiffness,
        fema_yield_rotation_rad=fema_yield_rotation,
        compression_force_n=effective_compression,
        expected_axial_yield_force_n=expected_axial_yield,
        lower_bound_compression_strength_n=lower_bound,
        axial_ratio_p_over_pcl=axial_ratio,
        fema_limits=limits,
    )


def create_rotational_hinge_material(
    material_tag: int, parameters: RotationalHingeParameters
) -> None:
    """Create a Steel01 deformation hinge or elastic force-controlled spring in OpenSees."""

    if material_tag <= 0:
        raise ValueError("OpenSees material tag must be positive")
    if parameters.control is HingeControl.FORCE:
        ops.uniaxialMaterial(
            "Elastic", material_tag, parameters.spring_initial_stiffness_nm_per_rad
        )
    else:
        ops.uniaxialMaterial(
            "Steel01",
            material_tag,
            parameters.expected_plastic_moment_nm,
            parameters.spring_initial_stiffness_nm_per_rad,
            parameters.spring_hardening_ratio,
        )


def add_hinged_member_2d(
    *,
    outer_i_node_tag: int,
    outer_j_node_tag: int,
    i_internal_node_tag: int,
    j_internal_node_tag: int,
    i_spring_element_tag: int,
    j_spring_element_tag: int,
    member_element_tag: int,
    i_material_tag: int,
    j_material_tag: int,
    area_m2: float,
    elastic_modulus_pa: float,
    adjusted_inertia_m4: float,
    transformation_tag: int,
    i_horizontal_retained_node_tag: int | None = None,
    j_horizontal_retained_node_tag: int | None = None,
) -> HingedMemberTags:
    """Add duplicate end nodes, rotation springs, and one elastic 2D member.

    Optional horizontal retained nodes let a frame diaphragm constrain every
    duplicate node directly to its floor master. This avoids unsupported chains
    in which an ``equalDOF`` constrained node is reused as a retained node.
    """

    node_tags = (
        outer_i_node_tag,
        outer_j_node_tag,
        i_internal_node_tag,
        j_internal_node_tag,
    )
    element_tags = (
        i_spring_element_tag,
        j_spring_element_tag,
        member_element_tag,
    )
    other_tags = (i_material_tag, j_material_tag, transformation_tag)
    if any(tag <= 0 for tag in (*node_tags, *element_tags, *other_tags)):
        raise ValueError("OpenSees tags must be positive")
    if len(set(node_tags)) != len(node_tags) or len(set(element_tags)) != len(element_tags):
        raise ValueError("OpenSees node tags and element tags must be unique within each namespace")
    values = (area_m2, elastic_modulus_pa, adjusted_inertia_m4)
    if any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("member area, modulus, and inertia must be finite and positive")
    i_coordinates = ops.nodeCoord(outer_i_node_tag)
    j_coordinates = ops.nodeCoord(outer_j_node_tag)
    if len(i_coordinates) != 2 or len(j_coordinates) != 2:
        raise ValueError("hinged member requires an existing two-dimensional OpenSees model")
    ops.node(i_internal_node_tag, *i_coordinates)
    ops.node(j_internal_node_tag, *j_coordinates)
    i_horizontal = i_horizontal_retained_node_tag or outer_i_node_tag
    j_horizontal = j_horizontal_retained_node_tag or outer_j_node_tag
    if i_horizontal == outer_i_node_tag:
        ops.equalDOF(outer_i_node_tag, i_internal_node_tag, 1, 2)
    else:
        ops.equalDOF(i_horizontal, i_internal_node_tag, 1)
        ops.equalDOF(outer_i_node_tag, i_internal_node_tag, 2)
    if j_horizontal == outer_j_node_tag:
        ops.equalDOF(outer_j_node_tag, j_internal_node_tag, 1, 2)
    else:
        ops.equalDOF(j_horizontal, j_internal_node_tag, 1)
        ops.equalDOF(outer_j_node_tag, j_internal_node_tag, 2)
    ops.element(
        "zeroLength",
        i_spring_element_tag,
        outer_i_node_tag,
        i_internal_node_tag,
        "-mat",
        i_material_tag,
        "-dir",
        3,
    )
    ops.element(
        "zeroLength",
        j_spring_element_tag,
        outer_j_node_tag,
        j_internal_node_tag,
        "-mat",
        j_material_tag,
        "-dir",
        3,
    )
    ops.element(
        "elasticBeamColumn",
        member_element_tag,
        i_internal_node_tag,
        j_internal_node_tag,
        area_m2,
        elastic_modulus_pa,
        adjusted_inertia_m4,
        transformation_tag,
    )
    return HingedMemberTags(
        member_element_tag=member_element_tag,
        i_spring_element_tag=i_spring_element_tag,
        j_spring_element_tag=j_spring_element_tag,
        i_internal_node_tag=i_internal_node_tag,
        j_internal_node_tag=j_internal_node_tag,
    )
