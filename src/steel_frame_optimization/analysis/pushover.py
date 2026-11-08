"""Assemble and solve the three-story concentrated-hinge pushover model."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal, cast

import openseespy.opensees as ops
from pydantic import Field, model_validator

from steel_frame_optimization.analysis.convergence import (
    DisplacementControlResult,
    NonlinearSolveResult,
    NonlinearSolverPolicy,
    run_displacement_control_path,
    run_static_load_path,
)
from steel_frame_optimization.analysis.hinges import (
    ColumnHingeMode,
    HingeControl,
    RotationalHingeParameters,
    add_hinged_member_2d,
    build_rotational_hinge_parameters,
    create_rotational_hinge_material,
)
from steel_frame_optimization.config.schemas import MemberSpec, StrictConfigModel
from steel_frame_optimization.constraints.fema import (
    PerformanceConstraint,
    PerformanceLevel,
    force_controlled_column_utilization,
    plastic_rotation_demand,
    story_drift_ratio,
)
from steel_frame_optimization.constraints.hierarchy import (
    CapacityHierarchyResult,
    strong_column_weak_beam_check,
)
from steel_frame_optimization.constraints.lrfd import (
    moment_gradient_factor,
    nominal_major_axis_flexural_strength,
)
from steel_frame_optimization.models.failures import FailureRecord
from steel_frame_optimization.models.frame import GeneratedFrame, MaterialAssignment
from steel_frame_optimization.sections.records import SectionRecord
from steel_frame_optimization.validation.phase5 import calculate_effective_lengths


class PerformanceTarget(StrictConfigModel):
    """Define one exact roof target and its drift/rotation acceptance policy."""

    target_id: str = Field(min_length=1)
    performance_level: PerformanceLevel
    roof_displacement_ratio: float = Field(gt=0.0, allow_inf_nan=False)
    drift_limit_ratio: float = Field(gt=0.0, allow_inf_nan=False)
    rotation_check: bool
    primary_acceptance: bool


class PushoverPolicy(StrictConfigModel):
    """Freeze gravity, targets, hierarchy scope, and numerical Phase 7 gates."""

    schema_version: Literal["1.0"] = "1.0"
    gravity_dead_factor: float = Field(gt=0.0, allow_inf_nan=False)
    gravity_live_factor: float = Field(gt=0.0, allow_inf_nan=False)
    constraint_handler: Literal["Penalty"]
    constraint_penalty_factor: float = Field(gt=0.0, allow_inf_nan=False)
    lateral_distribution_exponent: float = Field(gt=0.0, allow_inf_nan=False)
    targets: tuple[PerformanceTarget, ...] = Field(min_length=4)
    baseline_roof_increment_ratio: float = Field(gt=0.0, allow_inf_nan=False)
    refinement_roof_increment_ratios: tuple[float, ...] = Field(min_length=2)
    target_landing_absolute_tolerance_m: float = Field(gt=0.0, allow_inf_nan=False)
    constraint_translation_absolute_tolerance_m: float = Field(gt=0.0, allow_inf_nan=False)
    equilibrium_relative_tolerance: float = Field(gt=0.0, allow_inf_nan=False)
    hinge_force_relative_tolerance: float = Field(gt=0.0, allow_inf_nan=False)
    refinement_relative_tolerance: float = Field(gt=0.0, allow_inf_nan=False)
    repeat_relative_tolerance: float = Field(gt=0.0, allow_inf_nan=False)
    hierarchy_required_ratio: float = Field(gt=0.0, allow_inf_nan=False)
    hierarchy_joint_scope: Literal["levels_with_column_above_and_below_excluding_roof"]
    column_axial_force_source: Literal["preliminary_CP_member_local_end_1_compression"]
    paper_comparison_ls_target_id: str = Field(min_length=1)
    assumption_ids: tuple[str, ...] = Field(min_length=1)

    @classmethod
    def from_json(cls, path: Path) -> PushoverPolicy:
        """Load and strictly validate a Phase 7 policy file."""

        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))

    @model_validator(mode="after")
    def validate_targets_and_refinement(self) -> PushoverPolicy:
        """Require increasing unique targets and increasingly fine declared increments."""

        target_ids = [target.target_id for target in self.targets]
        ratios = [target.roof_displacement_ratio for target in self.targets]
        if len(set(target_ids)) != len(target_ids) or ratios != sorted(set(ratios)):
            raise ValueError("performance targets must have unique IDs and increasing ratios")
        if self.paper_comparison_ls_target_id not in target_ids:
            raise ValueError("paper comparison LS target must exist")
        refinements = self.refinement_roof_increment_ratios
        if any(first <= second for first, second in pairwise(refinements)):
            raise ValueError("refinement roof increments must be strictly decreasing")
        if refinements[-1] != self.baseline_roof_increment_ratio:
            raise ValueError("finest refinement must equal the baseline increment")
        return self


class CurvePoint(StrictConfigModel):
    """Persist one committed pushover equilibrium point."""

    roof_displacement_m: float = Field(ge=0.0, allow_inf_nan=False)
    base_shear_n: float = Field(ge=0.0, allow_inf_nan=False)
    lateral_load_factor_n: float = Field(ge=0.0, allow_inf_nan=False)
    story_displacements_m: tuple[float, ...]
    story_drift_ratios: tuple[float, ...]
    maximum_constraint_translation_residual_m: float = Field(ge=0.0, allow_inf_nan=False)
    lateral_equilibrium_relative_residual: float = Field(ge=0.0, allow_inf_nan=False)
    vertical_equilibrium_relative_residual: float = Field(ge=0.0, allow_inf_nan=False)


class HingeDemand(StrictConfigModel):
    """Persist one end-spring deformation, moment, plastic rotation, and FEMA state."""

    hinge_id: str = Field(min_length=1)
    member_id: str = Field(min_length=1)
    member_type: Literal["beam", "column"]
    end: Literal["i", "j"]
    control: HingeControl
    total_rotation_rad: float = Field(allow_inf_nan=False)
    moment_nm: float = Field(allow_inf_nan=False)
    plastic_rotation_rad: float = Field(ge=0.0, allow_inf_nan=False)
    state: Literal[
        "elastic", "within_io", "within_ls", "within_cp", "beyond_cp", "force_controlled"
    ]
    io_capacity_rad: float | None = Field(default=None, gt=0.0)
    ls_capacity_rad: float | None = Field(default=None, gt=0.0)
    cp_capacity_rad: float | None = Field(default=None, gt=0.0)


class MemberDemand(StrictConfigModel):
    """Persist one member's local axial force and governing end moments at a target."""

    member_id: str = Field(min_length=1)
    member_type: Literal["beam", "column"]
    compression_force_n: float = Field(ge=0.0, allow_inf_nan=False)
    i_end_moment_nm: float = Field(allow_inf_nan=False)
    j_end_moment_nm: float = Field(allow_inf_nan=False)
    maximum_abs_moment_nm: float = Field(ge=0.0, allow_inf_nan=False)


class MemberCapacity(StrictConfigModel):
    """Persist the hinge capacity/control basis assigned while building one member."""

    member_id: str = Field(min_length=1)
    member_type: Literal["beam", "column"]
    section_name: str = Field(min_length=1)
    control: HingeControl
    column_mode: ColumnHingeMode | None
    expected_plastic_moment_nm: float = Field(gt=0.0, allow_inf_nan=False)
    lower_bound_compression_strength_n: float | None = Field(default=None, gt=0.0)
    axial_ratio_p_over_pcl: float | None = Field(default=None, ge=0.0)


class TargetState(StrictConfigModel):
    """Persist complete frame, member, and hinge response at one landed roof target."""

    target: PerformanceTarget
    target_roof_displacement_m: float = Field(gt=0.0, allow_inf_nan=False)
    achieved_roof_displacement_m: float = Field(gt=0.0, allow_inf_nan=False)
    landing_error_m: float = Field(ge=0.0, allow_inf_nan=False)
    base_shear_n: float = Field(ge=0.0, allow_inf_nan=False)
    story_displacements_m: tuple[float, ...]
    story_drift_ratios: tuple[float, ...]
    member_demands: tuple[MemberDemand, ...]
    hinge_demands: tuple[HingeDemand, ...]
    maximum_hinge_force_mismatch_relative: float = Field(ge=0.0, allow_inf_nan=False)
    lateral_equilibrium_relative_residual: float = Field(ge=0.0, allow_inf_nan=False)
    vertical_equilibrium_relative_residual: float = Field(ge=0.0, allow_inf_nan=False)


class PushoverPassResult(StrictConfigModel):
    """Persist one preliminary or final gravity-plus-pushover pass."""

    pass_kind: Literal["preliminary", "final"]
    converged: bool
    gravity_result: NonlinearSolveResult
    gravity_equilibrium_relative_residual: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False
    )
    curve: tuple[CurvePoint, ...]
    target_states: tuple[TargetState, ...]
    member_capacities: tuple[MemberCapacity, ...]
    column_compressions_at_cp_n: dict[str, float]
    displacement_segments: tuple[DisplacementControlResult, ...]
    failure: FailureRecord | None = None

    @model_validator(mode="after")
    def validate_failure_state(self) -> PushoverPassResult:
        """Require explicit failure details only for incomplete passes."""

        if self.converged is (self.failure is not None):
            raise ValueError("converged passes omit failure; failed passes require it")
        return self


class DesignPushoverResult(StrictConfigModel):
    """Persist the complete two-pass evaluation and all performance constraints for one design."""

    design_id: str = Field(min_length=1)
    section_names_by_group: dict[str, str]
    roof_increment_ratio: float = Field(gt=0.0, allow_inf_nan=False)
    preliminary: PushoverPassResult
    final: PushoverPassResult
    constraints: tuple[PerformanceConstraint, ...]
    hierarchy_results: tuple[CapacityHierarchyResult, ...]
    primary_feasible: bool
    failed_primary_constraint_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _NonlinearIndex:
    """Map semantic frame members to transient OpenSees tags and hinge parameters."""

    node_tags: dict[str, int]
    support_node_tags: tuple[int, ...]
    member_element_tags: dict[str, int]
    member_internal_node_tags: dict[tuple[str, str], int]
    spring_element_tags: dict[tuple[str, str], int]
    hinge_parameters: dict[tuple[str, str], RotationalHingeParameters]
    member_capacities: tuple[MemberCapacity, ...]


def _member_length(member: MemberSpec, generated: GeneratedFrame) -> float:
    """Return one generated member centerline length."""

    nodes = {node.node_id: node for node in generated.topology.nodes}
    i_node = nodes[member.i_node]
    j_node = nodes[member.j_node]
    return math.hypot(j_node.x_m - i_node.x_m, j_node.y_m - i_node.y_m)


def _material(generated: GeneratedFrame, member_type: str) -> MaterialAssignment:
    """Return the configured beam or column material assignment."""

    return next(item for item in generated.definition.materials if item.member_type == member_type)


def _horizontal_retained_node_tag(generated: GeneratedFrame, node_id: str) -> int:
    """Return the floor diaphragm master, or the physical node at the foundation."""

    level = int(node_id.split("-")[1][1:])
    if level == 0:
        return generated.node_tags[node_id]
    return generated.node_tags[f"N-L{level}-C0"]


def _build_nonlinear_model(
    generated: GeneratedFrame,
    sections_by_group: dict[str, SectionRecord],
    *,
    pass_kind: Literal["preliminary", "final"],
    column_compressions_n: dict[str, float] | None,
) -> _NonlinearIndex:
    """Build a fresh two-dimensional frame with corrected elastic members and end springs."""

    if set(sections_by_group) != set(generated.definition.design_groups):
        raise ValueError("one section is required for every design group")
    if pass_kind == "final" and column_compressions_n is None:
        raise ValueError("final hinge model requires preliminary column compressions")
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    for node in generated.topology.nodes:
        ops.node(generated.node_tags[node.node_id], node.x_m, node.y_m)
    for support in generated.topology.supports:
        ops.fix(generated.node_tags[support.node_id], 1, 1, 1)
    columns_per_level = len(generated.definition.bay_widths_m) + 1
    for floor in generated.definition.floors:
        master = generated.node_tags[f"N-L{floor.level}-C0"]
        for column in range(1, columns_per_level):
            ops.equalDOF(master, generated.node_tags[f"N-L{floor.level}-C{column}"], 1)
    ops.geomTransf("PDelta", 1)

    effective_lengths = {
        item.member_id: item for item in calculate_effective_lengths(generated, sections_by_group)
    }
    next_node_tag = max(generated.node_tags.values()) + 1
    next_element_tag = 1
    next_material_tag = 1
    member_element_tags: dict[str, int] = {}
    member_internal_node_tags: dict[tuple[str, str], int] = {}
    spring_element_tags: dict[tuple[str, str], int] = {}
    hinge_parameters: dict[tuple[str, str], RotationalHingeParameters] = {}
    capacities: list[MemberCapacity] = []
    for member in generated.topology.members:
        section = sections_by_group[member.group_id]
        material = _material(generated, member.member_type)
        length = _member_length(member, generated)
        if member.member_type == "column":
            mode = (
                ColumnHingeMode.PRELIMINARY_UNREDUCED
                if pass_kind == "preliminary"
                else ColumnHingeMode.FINAL_AXIAL_REDUCED
            )
            compression = (
                0.0
                if pass_kind == "preliminary"
                else (column_compressions_n or {}).get(member.member_id, 0.0)
            )
            effective_length_factor = effective_lengths[member.member_id].effective_length_factor
        else:
            mode = None
            compression = 0.0
            effective_length_factor = 1.0
        parameters = build_rotational_hinge_parameters(
            section,
            member_type=member.member_type,
            member_length_m=length,
            elastic_modulus_pa=material.elastic_modulus_pa,
            nominal_yield_strength_pa=material.nominal_yield_strength_pa,
            expected_yield_strength_pa=material.expected_yield_strength_pa,
            column_mode=mode,
            compression_force_n=compression,
            effective_length_factor=effective_length_factor,
        )
        i_material = next_material_tag
        j_material = next_material_tag + 1
        create_rotational_hinge_material(i_material, parameters)
        create_rotational_hinge_material(j_material, parameters)
        i_spring = next_element_tag
        j_spring = next_element_tag + 1
        member_element = next_element_tag + 2
        add_hinged_member_2d(
            outer_i_node_tag=generated.node_tags[member.i_node],
            outer_j_node_tag=generated.node_tags[member.j_node],
            i_internal_node_tag=next_node_tag,
            j_internal_node_tag=next_node_tag + 1,
            i_spring_element_tag=i_spring,
            j_spring_element_tag=j_spring,
            member_element_tag=member_element,
            i_material_tag=i_material,
            j_material_tag=j_material,
            area_m2=section.area_m2,
            elastic_modulus_pa=material.elastic_modulus_pa,
            adjusted_inertia_m4=parameters.adjusted_member_inertia_m4,
            transformation_tag=1,
            i_horizontal_retained_node_tag=_horizontal_retained_node_tag(generated, member.i_node),
            j_horizontal_retained_node_tag=_horizontal_retained_node_tag(generated, member.j_node),
        )
        member_element_tags[member.member_id] = member_element
        member_internal_node_tags[(member.member_id, "i")] = next_node_tag
        member_internal_node_tags[(member.member_id, "j")] = next_node_tag + 1
        spring_element_tags[(member.member_id, "i")] = i_spring
        spring_element_tags[(member.member_id, "j")] = j_spring
        hinge_parameters[(member.member_id, "i")] = parameters
        hinge_parameters[(member.member_id, "j")] = parameters
        capacities.append(
            MemberCapacity(
                member_id=member.member_id,
                member_type=member.member_type,
                section_name=section.name,
                control=parameters.control,
                column_mode=parameters.column_mode,
                expected_plastic_moment_nm=parameters.expected_plastic_moment_nm,
                lower_bound_compression_strength_n=parameters.lower_bound_compression_strength_n,
                axial_ratio_p_over_pcl=parameters.axial_ratio_p_over_pcl,
            )
        )
        next_node_tag += 2
        next_element_tag += 3
        next_material_tag += 2
    return _NonlinearIndex(
        node_tags=dict(generated.node_tags),
        support_node_tags=tuple(
            generated.node_tags[support.node_id] for support in generated.topology.supports
        ),
        member_element_tags=member_element_tags,
        member_internal_node_tags=member_internal_node_tags,
        spring_element_tags=spring_element_tags,
        hinge_parameters=hinge_parameters,
        member_capacities=tuple(capacities),
    )


def _foundation_resultants(
    generated: GeneratedFrame, index: _NonlinearIndex
) -> tuple[float, float]:
    """Return global horizontal and vertical resultants at all column bases."""

    horizontal = 0.0
    vertical = 0.0
    for member in generated.topology.members:
        if member.member_type != "column" or not member.i_node.startswith("N-L0-"):
            continue
        force = tuple(
            float(value) for value in ops.eleForce(index.member_element_tags[member.member_id])
        )
        if len(force) != 6:
            raise RuntimeError(f"base column {member.member_id} returned an invalid force vector")
        horizontal += force[0]
        vertical += force[1]
    return horizontal, vertical


def _maximum_constraint_translation_residual_m(
    generated: GeneratedFrame, index: _NonlinearIndex
) -> float:
    """Measure the largest actual translation mismatch across declared MP constraints."""

    residuals: list[float] = []
    columns_per_level = len(generated.definition.bay_widths_m) + 1
    for floor in generated.definition.floors:
        master = index.node_tags[f"N-L{floor.level}-C0"]
        master_x = float(ops.nodeDisp(master, 1))
        for column in range(1, columns_per_level):
            slave = index.node_tags[f"N-L{floor.level}-C{column}"]
            residuals.append(abs(float(ops.nodeDisp(slave, 1)) - master_x))
    for member in generated.topology.members:
        for end, outer_node_id in (("i", member.i_node), ("j", member.j_node)):
            internal = index.member_internal_node_tags[(member.member_id, end)]
            horizontal = _horizontal_retained_node_tag(generated, outer_node_id)
            outer = index.node_tags[outer_node_id]
            residuals.append(
                abs(float(ops.nodeDisp(internal, 1)) - float(ops.nodeDisp(horizontal, 1)))
            )
            residuals.append(abs(float(ops.nodeDisp(internal, 2)) - float(ops.nodeDisp(outer, 2))))
    return max(residuals, default=0.0)


def _apply_gravity(
    generated: GeneratedFrame,
    index: _NonlinearIndex,
    policy: PushoverPolicy,
    solver_policy: NonlinearSolverPolicy,
) -> tuple[NonlinearSolveResult, float, float]:
    """Converge configured `1.1(D+L)` gravity and hold it before lateral loading."""

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    total_applied_y = 0.0
    line_loads = {load.member_id: load for load in generated.member_line_loads}
    for member in generated.topology.members:
        if member.member_type != "beam":
            continue
        load = line_loads[member.member_id]
        intensity = (
            policy.gravity_dead_factor * load.dead_n_per_m
            + policy.gravity_live_factor * load.live_n_per_m
        )
        ops.eleLoad(
            "-ele",
            index.member_element_tags[member.member_id],
            "-type",
            "-beamUniform",
            -intensity,
        )
        total_applied_y -= intensity * _member_length(member, generated)
    result = run_static_load_path(
        target_load_increment=1.0,
        nominal_steps=10,
        policy=solver_policy,
    )
    if not result.converged:
        return result, total_applied_y, math.inf
    _, reaction_y = _foundation_resultants(generated, index)
    residual = abs(total_applied_y + reaction_y) / max(abs(total_applied_y), 1.0)
    ops.loadConst("-time", 0.0)
    return result, total_applied_y, residual


def _apply_lateral_pattern(
    generated: GeneratedFrame, index: _NonlinearIndex, policy: PushoverPolicy
) -> None:
    """Apply a unit-total parabolic lateral pattern using story seismic weights."""

    elevations: list[float] = []
    elevation = 0.0
    for height in generated.definition.story_heights_m:
        elevation += height
        elevations.append(elevation)
    raw = tuple(
        floor.seismic_weight_n * elevations[floor.level - 1] ** policy.lateral_distribution_exponent
        for floor in generated.definition.floors
    )
    total = sum(raw)
    fractions_by_level = {
        floor.level: value / total
        for floor, value in zip(generated.definition.floors, raw, strict=True)
    }
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    for floor in generated.definition.floors:
        master_node = index.node_tags[f"N-L{floor.level}-C0"]
        ops.load(master_node, fractions_by_level[floor.level], 0.0, 0.0)


def _curve_point(
    generated: GeneratedFrame,
    index: _NonlinearIndex,
    *,
    total_applied_gravity_y_n: float,
) -> CurvePoint:
    """Extract one equilibrium point from the current committed OpenSees state."""

    reaction_x, reaction_y = _foundation_resultants(generated, index)
    load_factor = float(ops.getTime())
    base_shear = abs(reaction_x)
    story_displacements = tuple(
        float(ops.nodeDisp(index.node_tags[f"N-L{floor.level}-C0"], 1))
        for floor in generated.definition.floors
    )
    drift_ratios = tuple(
        story_drift_ratio(
            upper_displacement_m=value,
            lower_displacement_m=story_displacements[position - 1] if position else 0.0,
            height_m=generated.definition.story_heights_m[position],
        )
        for position, value in enumerate(story_displacements)
    )
    lateral_residual = abs(base_shear - load_factor) / max(abs(load_factor), 1.0)
    vertical_residual = abs(total_applied_gravity_y_n + reaction_y) / max(
        abs(total_applied_gravity_y_n), 1.0
    )
    return CurvePoint(
        roof_displacement_m=max(story_displacements[-1], 0.0),
        base_shear_n=base_shear,
        lateral_load_factor_n=max(load_factor, 0.0),
        story_displacements_m=story_displacements,
        story_drift_ratios=drift_ratios,
        maximum_constraint_translation_residual_m=(
            _maximum_constraint_translation_residual_m(generated, index)
        ),
        lateral_equilibrium_relative_residual=lateral_residual,
        vertical_equilibrium_relative_residual=vertical_residual,
    )


def _hinge_state(
    parameters: RotationalHingeParameters, plastic_rotation: float
) -> Literal["elastic", "within_io", "within_ls", "within_cp", "beyond_cp", "force_controlled"]:
    """Classify one hinge by its FEMA IO/LS/CP plastic-rotation thresholds."""

    if parameters.control is HingeControl.FORCE:
        return "force_controlled"
    if plastic_rotation <= 1e-14:
        return "elastic"
    limits = parameters.fema_limits
    if limits.io_rotation_rad is not None and plastic_rotation <= limits.io_rotation_rad:
        return "within_io"
    if limits.ls_rotation_rad is not None and plastic_rotation <= limits.ls_rotation_rad:
        return "within_ls"
    if limits.cp_rotation_rad is not None and plastic_rotation <= limits.cp_rotation_rad:
        return "within_cp"
    return "beyond_cp"


def _extract_target_state(
    generated: GeneratedFrame,
    index: _NonlinearIndex,
    target: PerformanceTarget,
    *,
    total_height_m: float,
    total_applied_gravity_y_n: float,
) -> TargetState:
    """Extract complete member/hinge response at an exactly landed performance target."""

    point = _curve_point(
        generated,
        index,
        total_applied_gravity_y_n=total_applied_gravity_y_n,
    )
    member_demands: list[MemberDemand] = []
    hinges: list[HingeDemand] = []
    maximum_mismatch = 0.0
    members = {member.member_id: member for member in generated.topology.members}
    for member_id, member_element_tag in index.member_element_tags.items():
        member = members[member_id]
        local = tuple(float(value) for value in ops.eleResponse(member_element_tag, "localForce"))
        if len(local) != 6:
            raise RuntimeError(f"member {member_id} returned {len(local)} local-force values")
        compression = max(local[0], 0.0) if member.member_type == "column" else 0.0
        member_demands.append(
            MemberDemand(
                member_id=member_id,
                member_type=member.member_type,
                compression_force_n=compression,
                i_end_moment_nm=local[2],
                j_end_moment_nm=-local[5],
                maximum_abs_moment_nm=max(abs(local[2]), abs(local[5])),
            )
        )
        for raw_end, member_moment in (("i", local[2]), ("j", local[5])):
            end = cast(Literal["i", "j"], raw_end)
            parameters = index.hinge_parameters[(member_id, end)]
            spring_tag = index.spring_element_tags[(member_id, end)]
            deformation = tuple(
                float(value) for value in ops.eleResponse(spring_tag, "deformation")
            )
            force = tuple(float(value) for value in ops.eleForce(spring_tag))
            if len(deformation) != 1 or len(force) != 6:
                raise RuntimeError(f"hinge {member_id}-{end} returned an invalid response vector")
            spring_moment = force[5]
            mismatch = abs(abs(spring_moment) - abs(member_moment)) / max(abs(member_moment), 1.0)
            maximum_mismatch = max(maximum_mismatch, mismatch)
            plastic = plastic_rotation_demand(
                total_spring_rotation_rad=deformation[0],
                spring_moment_nm=spring_moment,
                initial_stiffness_nm_per_rad=parameters.spring_initial_stiffness_nm_per_rad,
            )
            hinges.append(
                HingeDemand(
                    hinge_id=f"{member_id}-{end}",
                    member_id=member_id,
                    member_type=member.member_type,
                    end=end,
                    control=parameters.control,
                    total_rotation_rad=deformation[0],
                    moment_nm=spring_moment,
                    plastic_rotation_rad=plastic,
                    state=_hinge_state(parameters, plastic),
                    io_capacity_rad=parameters.fema_limits.io_rotation_rad,
                    ls_capacity_rad=parameters.fema_limits.ls_rotation_rad,
                    cp_capacity_rad=parameters.fema_limits.cp_rotation_rad,
                )
            )
    target_displacement = target.roof_displacement_ratio * total_height_m
    return TargetState(
        target=target,
        target_roof_displacement_m=target_displacement,
        achieved_roof_displacement_m=point.roof_displacement_m,
        landing_error_m=abs(point.roof_displacement_m - target_displacement),
        base_shear_n=point.base_shear_n,
        story_displacements_m=point.story_displacements_m,
        story_drift_ratios=point.story_drift_ratios,
        member_demands=tuple(member_demands),
        hinge_demands=tuple(hinges),
        maximum_hinge_force_mismatch_relative=maximum_mismatch,
        lateral_equilibrium_relative_residual=point.lateral_equilibrium_relative_residual,
        vertical_equilibrium_relative_residual=point.vertical_equilibrium_relative_residual,
    )


def run_pushover_pass(
    generated: GeneratedFrame,
    sections_by_group: dict[str, SectionRecord],
    *,
    pass_kind: Literal["preliminary", "final"],
    policy: PushoverPolicy,
    solver_policy: NonlinearSolverPolicy,
    roof_increment_ratio: float,
    column_compressions_n: dict[str, float] | None = None,
) -> PushoverPassResult:
    """Run gravity then exact-target displacement control for one fresh hinge model."""

    total_height = sum(generated.definition.story_heights_m)
    index = _build_nonlinear_model(
        generated,
        sections_by_group,
        pass_kind=pass_kind,
        column_compressions_n=column_compressions_n,
    )
    pass_solver_policy = solver_policy.model_copy(
        update={
            "constraint_handler": policy.constraint_handler,
            "constraint_penalty_factor": policy.constraint_penalty_factor,
        }
    )
    try:
        gravity, total_gravity_y, gravity_residual = _apply_gravity(
            generated, index, policy, pass_solver_policy
        )
        if not gravity.converged:
            return PushoverPassResult(
                pass_kind=pass_kind,
                converged=False,
                gravity_result=gravity,
                gravity_equilibrium_relative_residual=None,
                curve=(),
                target_states=(),
                member_capacities=index.member_capacities,
                column_compressions_at_cp_n={},
                displacement_segments=(),
                failure=gravity.failure,
            )
        _apply_lateral_pattern(generated, index, policy)
        curve = [
            _curve_point(
                generated,
                index,
                total_applied_gravity_y_n=total_gravity_y,
            )
        ]
        states: list[TargetState] = []
        segments: list[DisplacementControlResult] = []
        previous_target = 0.0
        nominal_increment = roof_increment_ratio * total_height
        roof_node = index.node_tags[f"N-L{len(generated.definition.floors)}-C0"]
        for target in policy.targets:
            target_displacement = target.roof_displacement_ratio * total_height
            segment = run_displacement_control_path(
                control_node=roof_node,
                control_dof=1,
                target_displacement_increment=target_displacement - previous_target,
                nominal_displacement_increment=nominal_increment,
                policy=pass_solver_policy,
                on_commit=lambda _step: curve.append(
                    _curve_point(
                        generated,
                        index,
                        total_applied_gravity_y_n=total_gravity_y,
                    )
                ),
            )
            segments.append(segment)
            if not segment.converged:
                return PushoverPassResult(
                    pass_kind=pass_kind,
                    converged=False,
                    gravity_result=gravity,
                    gravity_equilibrium_relative_residual=gravity_residual,
                    curve=tuple(curve),
                    target_states=tuple(states),
                    member_capacities=index.member_capacities,
                    column_compressions_at_cp_n={},
                    displacement_segments=tuple(segments),
                    failure=segment.failure,
                )
            states.append(
                _extract_target_state(
                    generated,
                    index,
                    target,
                    total_height_m=total_height,
                    total_applied_gravity_y_n=total_gravity_y,
                )
            )
            previous_target = target_displacement
        cp_state = next(state for state in states if state.target.performance_level == "CP")
        column_compressions = {
            demand.member_id: demand.compression_force_n
            for demand in cp_state.member_demands
            if demand.member_type == "column"
        }
        return PushoverPassResult(
            pass_kind=pass_kind,
            converged=True,
            gravity_result=gravity,
            gravity_equilibrium_relative_residual=gravity_residual,
            curve=tuple(curve),
            target_states=tuple(states),
            member_capacities=index.member_capacities,
            column_compressions_at_cp_n=column_compressions,
            displacement_segments=tuple(segments),
        )
    finally:
        ops.wipe()


def _column_cb(demand: MemberDemand) -> float:
    """Return AISC Cb for a linear no-transverse-load column moment diagram."""

    start = demand.i_end_moment_nm
    end = demand.j_end_moment_nm
    quarter = 0.75 * start + 0.25 * end
    middle = 0.5 * (start + end)
    three_quarter = 0.25 * start + 0.75 * end
    return moment_gradient_factor(
        moment_a_nm=quarter,
        moment_b_nm=middle,
        moment_c_nm=three_quarter,
        maximum_moment_nm=max(abs(start), abs(end)),
    )


def _performance_constraints(
    generated: GeneratedFrame,
    sections_by_group: dict[str, SectionRecord],
    final: PushoverPassResult,
) -> tuple[PerformanceConstraint, ...]:
    """Evaluate drift, deformation-hinge, and force-controlled column constraints."""

    members = {member.member_id: member for member in generated.topology.members}
    capacities = {item.member_id: item for item in final.member_capacities}
    results: list[PerformanceConstraint] = []
    for state in final.target_states:
        target = state.target
        for story, drift in enumerate(state.story_drift_ratios, start=1):
            utilization = drift / target.drift_limit_ratio
            results.append(
                PerformanceConstraint(
                    check_id=f"{target.target_id}:story:{story}:drift",
                    check_type="story_drift",
                    target_id=target.target_id,
                    performance_level=target.performance_level,
                    subject_id=f"story-{story}",
                    demand=drift,
                    capacity=target.drift_limit_ratio,
                    utilization=utilization,
                    passed=utilization <= 1.0,
                    primary_acceptance=target.primary_acceptance,
                    source="REF-A Eq. (30) and Table 5",
                )
            )
        if target.rotation_check:
            level = target.performance_level
            if level == "OP":
                raise RuntimeError("OP target cannot request a hinge rotation check")
            for hinge in state.hinge_demands:
                if hinge.control is not HingeControl.DEFORMATION:
                    continue
                parameters_capacity = {
                    "IO": hinge.io_capacity_rad,
                    "LS": hinge.ls_capacity_rad,
                    "CP": hinge.cp_capacity_rad,
                }[level]
                if parameters_capacity is None:
                    raise RuntimeError(f"hinge {hinge.hinge_id} has no {level} capacity")
                utilization = hinge.plastic_rotation_rad / parameters_capacity
                results.append(
                    PerformanceConstraint(
                        check_id=f"{target.target_id}:{hinge.hinge_id}:rotation",
                        check_type="hinge_rotation",
                        target_id=target.target_id,
                        performance_level=level,
                        subject_id=hinge.hinge_id,
                        demand=hinge.plastic_rotation_rad,
                        capacity=parameters_capacity,
                        utilization=utilization,
                        passed=utilization <= 1.0,
                        primary_acceptance=target.primary_acceptance,
                        source="REF-A Eqs. (27)-(28); FEMA 356 Table 5-6",
                    )
                )
            demand_by_member = {item.member_id: item for item in state.member_demands}
            for member_id, capacity in capacities.items():
                if capacity.member_type != "column" or capacity.control is not HingeControl.FORCE:
                    continue
                member = members[member_id]
                demand = demand_by_member[member_id]
                section = sections_by_group[member.group_id]
                material = _material(generated, "column")
                moment_strength = nominal_major_axis_flexural_strength(
                    section,
                    elastic_modulus_pa=material.elastic_modulus_pa,
                    yield_strength_pa=material.nominal_yield_strength_pa,
                    resistance_factor=1.0,
                    unbraced_length_m=_member_length(member, generated),
                    cb=_column_cb(demand),
                    fully_braced=False,
                ).nominal_strength_nm
                if capacity.lower_bound_compression_strength_n is None:
                    raise RuntimeError(f"force-controlled column {member_id} has no Pcl")
                utilization = force_controlled_column_utilization(
                    compression_force_n=demand.compression_force_n,
                    pcl_n=capacity.lower_bound_compression_strength_n,
                    moment_nm=demand.maximum_abs_moment_nm,
                    nominal_moment_nm=moment_strength,
                )
                results.append(
                    PerformanceConstraint(
                        check_id=f"{target.target_id}:{member_id}:force",
                        check_type="force_controlled_column",
                        target_id=target.target_id,
                        performance_level=level,
                        subject_id=member_id,
                        demand=utilization,
                        capacity=1.0,
                        utilization=utilization,
                        passed=utilization < 1.0,
                        primary_acceptance=target.primary_acceptance,
                        source="REF-A Eq. (29)",
                    )
                )
    return tuple(results)


def _hierarchy_results(
    generated: GeneratedFrame, final: PushoverPassResult, *, required_ratio: float
) -> tuple[CapacityHierarchyResult, ...]:
    """Evaluate Eq. (31) at every non-roof beam-column joint."""

    capacities = {item.member_id: item for item in final.member_capacities}
    incident: dict[str, list[MemberSpec]] = defaultdict(list)
    for member in generated.topology.members:
        incident[member.i_node].append(member)
        incident[member.j_node].append(member)
    results: list[CapacityHierarchyResult] = []
    roof_level = len(generated.definition.floors)
    for node in generated.topology.nodes:
        if node.y_m <= 0.0 or node.node_id.startswith(f"N-L{roof_level}-"):
            continue
        columns = {
            member.member_id: capacities[member.member_id].expected_plastic_moment_nm
            for member in incident[node.node_id]
            if member.member_type == "column"
        }
        beams = {
            member.member_id: capacities[member.member_id].expected_plastic_moment_nm
            for member in incident[node.node_id]
            if member.member_type == "beam"
        }
        if columns and beams:
            results.append(
                strong_column_weak_beam_check(
                    joint_id=node.node_id,
                    column_capacities_nm=columns,
                    beam_capacities_nm=beams,
                    required_ratio=required_ratio,
                )
            )
    return tuple(results)


def evaluate_design_pushover(
    generated: GeneratedFrame,
    sections_by_group: dict[str, SectionRecord],
    *,
    design_id: str,
    policy: PushoverPolicy,
    solver_policy: NonlinearSolverPolicy,
    roof_increment_ratio: float,
) -> DesignPushoverResult:
    """Run the paper's preliminary/final hinge sequence and evaluate all Phase 7 constraints."""

    preliminary = run_pushover_pass(
        generated,
        sections_by_group,
        pass_kind="preliminary",
        policy=policy,
        solver_policy=solver_policy,
        roof_increment_ratio=roof_increment_ratio,
    )
    if not preliminary.converged:
        raise RuntimeError(f"preliminary pushover failed: {preliminary.failure}")
    final = run_pushover_pass(
        generated,
        sections_by_group,
        pass_kind="final",
        policy=policy,
        solver_policy=solver_policy,
        roof_increment_ratio=roof_increment_ratio,
        column_compressions_n=preliminary.column_compressions_at_cp_n,
    )
    if not final.converged:
        raise RuntimeError(f"final pushover failed: {final.failure}")
    constraints = list(_performance_constraints(generated, sections_by_group, final))
    hierarchy = _hierarchy_results(generated, final, required_ratio=policy.hierarchy_required_ratio)
    cp_target = next(target for target in policy.targets if target.performance_level == "CP")
    for result in hierarchy:
        constraints.append(
            PerformanceConstraint(
                check_id=f"CP:{result.joint_id}:hierarchy",
                check_type="hierarchy",
                target_id=cp_target.target_id,
                performance_level="CP",
                subject_id=result.joint_id,
                demand=result.beam_capacity_sum_nm,
                capacity=result.column_capacity_sum_nm,
                utilization=1.0 / result.ratio,
                passed=result.passed,
                primary_acceptance=True,
                source="REF-A Eq. (31)",
            )
        )
    failed = tuple(
        item.check_id for item in constraints if item.primary_acceptance and not item.passed
    )
    return DesignPushoverResult(
        design_id=design_id,
        section_names_by_group={
            group: section.name for group, section in sections_by_group.items()
        },
        roof_increment_ratio=roof_increment_ratio,
        preliminary=preliminary,
        final=final,
        constraints=tuple(constraints),
        hierarchy_results=hierarchy,
        primary_feasible=not failed,
        failed_primary_constraint_ids=failed,
    )


def interpolate_base_shear_n(curve: tuple[CurvePoint, ...], roof_displacement_m: float) -> float:
    """Linearly interpolate base shear on a monotonic-displacement pushover curve."""

    if not curve:
        raise ValueError("pushover curve must not be empty")
    if (
        roof_displacement_m < curve[0].roof_displacement_m
        or roof_displacement_m > curve[-1].roof_displacement_m
    ):
        raise ValueError("requested displacement lies outside the pushover curve")
    for lower, upper in pairwise(curve):
        if math.isclose(roof_displacement_m, lower.roof_displacement_m, abs_tol=1e-14):
            return lower.base_shear_n
        if lower.roof_displacement_m <= roof_displacement_m <= upper.roof_displacement_m:
            span = upper.roof_displacement_m - lower.roof_displacement_m
            if span <= 0.0:
                raise ValueError("pushover curve displacement must be strictly increasing")
            fraction = (roof_displacement_m - lower.roof_displacement_m) / span
            return lower.base_shear_n + fraction * (upper.base_shear_n - lower.base_shear_n)
    return curve[-1].base_shear_n
