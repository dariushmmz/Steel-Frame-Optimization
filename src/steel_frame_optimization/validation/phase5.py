"""Evaluate complete three-story Stage 1 LRFD constraints and research comparisons."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from steel_frame_optimization.analysis.elastic import (
    LoadFactors,
    StaticResult,
    run_modal_analysis,
    run_static_analysis,
)
from steel_frame_optimization.analysis.modal import (
    distribute_story_forces_n,
    elastic_base_shear_n,
    spectral_acceleration_g,
)
from steel_frame_optimization.config.schemas import MemberSpec, StrictConfigModel
from steel_frame_optimization.constraints.lrfd import (
    FlexuralStrength,
    beam_column_interaction_utilization,
    effective_length_factor_unbraced,
    moment_gradient_factor,
    nominal_compression_strength_with_local_buckling,
    nominal_major_axis_flexural_strength,
)
from steel_frame_optimization.constraints.serviceability import (
    chord_relative_midspan_deflection_m,
    deflection_utilization,
    drift_utilization,
)
from steel_frame_optimization.models.frame import (
    FrameDefinition,
    GeneratedFrame,
    generate_frame,
)
from steel_frame_optimization.sections.records import SectionRecord
from steel_frame_optimization.sections.repository import SectionRepository
from steel_frame_optimization.serialization import write_canonical_json
from steel_frame_optimization.validation.phase4 import AnalysisPolicyInput, SpectrumInput


class Stage1Case(StrictConfigModel):
    """Validate a reusable three- or nine-story Stage 1 assignment and spectrum."""

    schema_version: Literal["1.0"] = "1.0"
    case_id: str = Field(min_length=1)
    frame_id: Literal["3-story", "9-story"]
    domain_label: Literal["273-section domain, pending client confirmation"]
    source: str = Field(min_length=1)
    sections_by_group: dict[str, str]
    spectrum: SpectrumInput
    load_combination: LoadFactors
    analysis_policy: AnalysisPolicyInput

    @classmethod
    def from_json(cls, path: Path) -> Stage1Case:
        """Load and validate one Stage 1 case configuration."""

        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))

    @model_validator(mode="after")
    def require_all_groups(self) -> Stage1Case:
        """Require exactly the source-defined group assignments for the selected frame."""

        final_group = 5 if self.frame_id == "3-story" else 13
        expected = {str(group) for group in range(1, final_group + 1)}
        if set(self.sections_by_group) != expected:
            raise ValueError(f"Stage 1 case must assign exactly groups 1 through {final_group}")
        return self


class Stage1Policy(StrictConfigModel):
    """Validate the owner-authorized full Stage 1 analysis policy."""

    schema_version: Literal["1.0"] = "1.0"
    policy_id: Literal["stage1-lrfd-research-baseline-v1"]
    domain_approval: Literal["owner-approved 273-section AISC v14.1 implementation baseline"]
    strength_gravity: LoadFactors
    strength_seismic: LoadFactors
    service_dead: LoadFactors
    service_live: LoadFactors
    service_dead_plus_live: LoadFactors
    compression_resistance_factor: float = Field(gt=0.0, le=1.0)
    flexural_resistance_factor: float = Field(gt=0.0, le=1.0)
    beam_fully_braced: Literal[True]
    column_unbraced_length: Literal["story_centerline_length"]
    column_buckling_axis: Literal["major_rx"]
    deflection_denominators: dict[str, float]
    elastic_story_drift_limit: float = Field(gt=0.0)
    elastic_roof_drift_limit: float = Field(gt=0.0)
    equilibrium_relative_tolerance: float = Field(gt=0.0)
    assumption_ids: tuple[str, ...]
    status: Literal[
        "owner-authorized research implementation, pending independent engineering review"
    ]

    @classmethod
    def from_json(cls, path: Path) -> Stage1Policy:
        """Load and validate the full Stage 1 policy from JSON."""

        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))

    @model_validator(mode="after")
    def require_fixed_cases_and_limits(self) -> Stage1Policy:
        """Reject incomplete or silently renamed Stage 1 combinations and service limits."""

        expected_factors = (
            (self.strength_gravity.dead, self.strength_gravity.live, self.strength_gravity.seismic),
            (self.strength_seismic.dead, self.strength_seismic.live, self.strength_seismic.seismic),
            (self.service_dead.dead, self.service_dead.live, self.service_dead.seismic),
            (self.service_live.dead, self.service_live.live, self.service_live.seismic),
            (
                self.service_dead_plus_live.dead,
                self.service_dead_plus_live.live,
                self.service_dead_plus_live.seismic,
            ),
        )
        if expected_factors != (
            (1.2, 1.6, 0.0),
            (1.2, 1.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        ):
            raise ValueError("Stage 1 load combinations differ from the approved policy")
        if self.deflection_denominators != {
            "service_D": 120.0,
            "service_L": 360.0,
            "service_D_plus_L": 240.0,
        }:
            raise ValueError("service deflection limits must be L/120, L/360, and L/240")
        return self


class PublishedStage1Reference(StrictConfigModel):
    """Store the exact REF-A Table 6-7 Case 2 values as diagnostics."""

    schema_version: Literal["1.0"] = "1.0"
    source: Literal["REF-A Tables 6 and 7, Case 2", "REF-A Tables 10 and 11, Case 2 GWO"]
    metrics: dict[str, float]
    comparison_policy: Literal["diagnostic only; discrepancies do not imply implementation failure"]

    @classmethod
    def from_json(cls, path: Path) -> PublishedStage1Reference:
        """Load and validate the published comparison values."""

        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


class EffectiveLengthResult(StrictConfigModel):
    """Record candidate-specific end stiffness ratios and effective length for one column."""

    member_id: str
    group_id: str
    g_i: float = Field(ge=0.0, allow_inf_nan=False)
    g_j: float = Field(ge=0.0, allow_inf_nan=False)
    effective_length_factor: float = Field(gt=0.0, allow_inf_nan=False)
    member_length_m: float = Field(gt=0.0, allow_inf_nan=False)


class ConstraintResult(StrictConfigModel):
    """Record one member/service/drift constraint with explicit demand and capacity."""

    check_id: str
    member_id: str
    group_id: str
    story: int = Field(gt=0)
    member_type: Literal["beam", "column", "frame"]
    load_case: str
    check_type: Literal[
        "beam_flexure", "column_interaction", "beam_deflection", "story_drift", "roof_drift"
    ]
    demand: float = Field(ge=0.0, allow_inf_nan=False)
    capacity: float = Field(gt=0.0, allow_inf_nan=False)
    utilization: float = Field(ge=0.0, allow_inf_nan=False)
    passed: bool
    source: str
    details: dict[str, float | str | bool]


class ResearchComparison(StrictConfigModel):
    """Keep one published diagnostic visible without treating agreement as a software gate."""

    metric_id: str
    actual: float = Field(ge=0.0, allow_inf_nan=False)
    published: float = Field(ge=0.0, allow_inf_nan=False)
    signed_difference: float = Field(allow_inf_nan=False)
    relative_difference: float = Field(allow_inf_nan=False)
    disposition: Literal["diagnostic_not_an_acceptance_gate"]


class Stage1Artifact(StrictConfigModel):
    """Persist all Stage 1 inputs, direct analyses, constraints, and research diagnostics."""

    schema_version: Literal["1.0"] = "1.0"
    review_status: Literal["not independently engineering-validated"]
    domain_approval: Literal["owner-approved 273-section AISC v14.1 implementation baseline"]
    case: Stage1Case
    policy: Stage1Policy
    effective_lengths: tuple[EffectiveLengthResult, ...]
    analyses: dict[str, StaticResult]
    constraints: tuple[ConstraintResult, ...]
    feasible: bool
    failed_constraint_ids: tuple[str, ...]
    maximum_utilization: float = Field(ge=0.0, allow_inf_nan=False)
    research_comparisons: tuple[ResearchComparison, ...]
    research_comparison_status: Literal["recorded_with_no_agreement_claim"]


def _member_length(member: MemberSpec, generated: GeneratedFrame) -> float:
    """Return one generated member's centerline length."""

    nodes = {node.node_id: node for node in generated.topology.nodes}
    i_node = nodes[member.i_node]
    j_node = nodes[member.j_node]
    return math.hypot(j_node.x_m - i_node.x_m, j_node.y_m - i_node.y_m)


def _story(member: MemberSpec) -> int:
    """Return a generated beam level or column story from its semantic ID."""

    token = member.member_id.split("-")[1]
    return int(token[1:])


def _material_properties(
    generated: GeneratedFrame, member_type: Literal["beam", "column"]
) -> tuple[float, float]:
    """Return nominal yield strength and elastic modulus for one member type."""

    material = next(
        item for item in generated.definition.materials if item.member_type == member_type
    )
    return material.nominal_yield_strength_pa, material.elastic_modulus_pa


def calculate_effective_lengths(
    generated: GeneratedFrame, sections_by_group: dict[str, SectionRecord]
) -> tuple[EffectiveLengthResult, ...]:
    """Calculate REF-A Eqs. (25)-(26) from the current candidate's joint stiffnesses."""

    nodes = {node.node_id: node for node in generated.topology.nodes}
    incident: dict[str, list[MemberSpec]] = defaultdict(list)
    for member in generated.topology.members:
        incident[member.i_node].append(member)
        incident[member.j_node].append(member)
    _, beam_e = _material_properties(generated, "beam")
    _, column_e = _material_properties(generated, "column")

    def joint_g(node_id: str) -> float:
        if math.isclose(nodes[node_id].y_m, 0.0, abs_tol=1e-12):
            return 1.0
        column_stiffness = sum(
            column_e * sections_by_group[member.group_id].ix_m4 / _member_length(member, generated)
            for member in incident[node_id]
            if member.member_type == "column"
        )
        beam_stiffness = sum(
            beam_e * sections_by_group[member.group_id].ix_m4 / _member_length(member, generated)
            for member in incident[node_id]
            if member.member_type == "beam"
        )
        if beam_stiffness <= 0.0:
            raise ValueError(f"joint {node_id} has no positive beam stiffness for G calculation")
        return column_stiffness / beam_stiffness

    results = []
    for member in generated.topology.members:
        if member.member_type != "column":
            continue
        g_i = joint_g(member.i_node)
        g_j = joint_g(member.j_node)
        results.append(
            EffectiveLengthResult(
                member_id=member.member_id,
                group_id=member.group_id,
                g_i=g_i,
                g_j=g_j,
                effective_length_factor=effective_length_factor_unbraced(g_a=g_i, g_b=g_j),
                member_length_m=_member_length(member, generated),
            )
        )
    return tuple(results)


def _forces_by_member(result: StaticResult) -> dict[str, list[tuple[float, ...]]]:
    """Group local analysis-element forces by the semantic physical member."""

    grouped: dict[str, list[tuple[float, ...]]] = defaultdict(list)
    for force in result.element_forces:
        grouped[force.semantic_member_id].append(force.local_end_forces_n_nm)
    return grouped


def _column_cb(forces: tuple[float, ...]) -> float:
    """Calculate Cb from the linear physical moment diagram of one column element."""

    start = forces[2]
    end = -forces[5]
    quarter = 0.75 * start + 0.25 * end
    middle = 0.5 * start + 0.5 * end
    three_quarter = 0.25 * start + 0.75 * end
    maximum = max(abs(start), abs(end))
    return moment_gradient_factor(
        moment_a_nm=quarter,
        moment_b_nm=middle,
        moment_c_nm=three_quarter,
        maximum_moment_nm=maximum,
    )


def _strength_constraints(
    *,
    load_case: str,
    result: StaticResult,
    generated: GeneratedFrame,
    sections_by_group: dict[str, SectionRecord],
    policy: Stage1Policy,
    effective_lengths: dict[str, EffectiveLengthResult],
) -> list[ConstraintResult]:
    """Build beam flexure and column interaction checks for one direct combination."""

    grouped = _forces_by_member(result)
    beam_fy, beam_e = _material_properties(generated, "beam")
    column_fy, column_e = _material_properties(generated, "column")
    checks: list[ConstraintResult] = []
    for member in generated.topology.members:
        section = sections_by_group[member.group_id]
        member_forces = grouped[member.member_id]
        moment_demand = max(
            abs(value) for forces in member_forces for value in (forces[2], forces[5])
        )
        if member.member_type == "beam":
            strength = nominal_major_axis_flexural_strength(
                section,
                elastic_modulus_pa=beam_e,
                yield_strength_pa=beam_fy,
                resistance_factor=policy.flexural_resistance_factor,
                unbraced_length_m=0.0,
                cb=1.0,
                fully_braced=policy.beam_fully_braced,
            )
            utilization = moment_demand / strength.design_strength_nm
            checks.append(
                ConstraintResult(
                    check_id=f"{load_case}:{member.member_id}:beam_flexure",
                    member_id=member.member_id,
                    group_id=member.group_id,
                    story=_story(member),
                    member_type="beam",
                    load_case=load_case,
                    check_type="beam_flexure",
                    demand=moment_demand,
                    capacity=strength.design_strength_nm,
                    utilization=utilization,
                    passed=utilization <= 1.0,
                    source="AISC 360-16 F1/F2/F3; ASM-006/008/009",
                    details=_flexural_details(strength),
                )
            )
            continue

        effective = effective_lengths[member.member_id]
        compression_demand = max(
            0.0,
            *(max(forces[0], -forces[3]) for forces in member_forces),
        )
        compression = nominal_compression_strength_with_local_buckling(
            section,
            effective_length_factor=effective.effective_length_factor,
            length_m=effective.member_length_m,
            radius_of_gyration_m=section.rx_m,
            yield_strength_pa=column_fy,
            elastic_modulus_pa=column_e,
        )
        cb = _column_cb(member_forces[0])
        flexure = nominal_major_axis_flexural_strength(
            section,
            elastic_modulus_pa=column_e,
            yield_strength_pa=column_fy,
            resistance_factor=policy.flexural_resistance_factor,
            unbraced_length_m=effective.member_length_m,
            cb=cb,
            fully_braced=False,
        )
        utilization = beam_column_interaction_utilization(
            compression_demand_n=compression_demand,
            moment_demand_nm=moment_demand,
            nominal_compression_strength_n_value=compression.nominal_strength_n,
            nominal_flexural_strength_nm=flexure.nominal_strength_nm,
            compression_resistance_factor=policy.compression_resistance_factor,
            flexural_resistance_factor=policy.flexural_resistance_factor,
        )
        checks.append(
            ConstraintResult(
                check_id=f"{load_case}:{member.member_id}:column_interaction",
                member_id=member.member_id,
                group_id=member.group_id,
                story=_story(member),
                member_type="column",
                load_case=load_case,
                check_type="column_interaction",
                demand=utilization,
                capacity=1.0,
                utilization=utilization,
                passed=utilization <= 1.0,
                source="REF-A Eqs. (11), (13)-(15), (25)-(26); AISC 360-16 E3/F2/F3/H1",
                details={
                    "compression_demand_n": compression_demand,
                    "moment_demand_nm": moment_demand,
                    "nominal_compression_strength_n": compression.nominal_strength_n,
                    "critical_compression_stress_pa": compression.critical_stress_pa,
                    "gross_area_m2": compression.gross_area_m2,
                    "effective_area_m2": compression.effective_area_m2,
                    "axial_flange_class": compression.classification.flange_class,
                    "axial_web_class": compression.classification.web_class,
                    "compression_limit_state": compression.controlling_limit_state,
                    "nominal_flexural_strength_nm": flexure.nominal_strength_nm,
                    "effective_length_factor": effective.effective_length_factor,
                    **_flexural_details(flexure),
                },
            )
        )
    return checks


def _flexural_details(strength: FlexuralStrength) -> dict[str, float | str | bool]:
    """Flatten one flexural-strength record into auditable constraint details."""

    return {
        "flange_class": strength.classification.flange_class,
        "web_class": strength.classification.web_class,
        "nominal_flexural_strength_nm": strength.nominal_strength_nm,
        "design_flexural_strength_nm": strength.design_strength_nm,
        "controlling_limit_state": strength.controlling_limit_state,
        "unbraced_length_m": strength.unbraced_length_m,
        "lp_m": strength.lp_m,
        "lr_m": strength.lr_m,
        "cb": strength.cb,
    }


def _service_constraints(
    *,
    load_case: str,
    result: StaticResult,
    generated: GeneratedFrame,
    denominator: float,
) -> list[ConstraintResult]:
    """Build chord-relative beam deflection checks for one service combination."""

    nodes = {node.node_id: node for node in generated.topology.nodes}
    checks = []
    for member in generated.topology.members:
        if member.member_type != "beam":
            continue
        deflection = chord_relative_midspan_deflection_m(
            midpoint_vertical_m=result.beam_midspan_vertical_displacements_m[member.member_id],
            i_end_vertical_m=result.node_displacements_m_rad[member.i_node][1],
            j_end_vertical_m=result.node_displacements_m_rad[member.j_node][1],
        )
        span = math.hypot(
            nodes[member.j_node].x_m - nodes[member.i_node].x_m,
            nodes[member.j_node].y_m - nodes[member.i_node].y_m,
        )
        capacity = span / denominator
        utilization = deflection_utilization(
            deflection_m=deflection, span_m=span, denominator=denominator
        )
        checks.append(
            ConstraintResult(
                check_id=f"{load_case}:{member.member_id}:beam_deflection",
                member_id=member.member_id,
                group_id=member.group_id,
                story=_story(member),
                member_type="beam",
                load_case=load_case,
                check_type="beam_deflection",
                demand=deflection,
                capacity=capacity,
                utilization=utilization,
                passed=utilization <= 1.0,
                source="REF-A Table 1",
                details={"limit_denominator": denominator, "span_m": span},
            )
        )
    return checks


def _drift_constraints(
    *, load_case: str, result: StaticResult, generated: GeneratedFrame, policy: Stage1Policy
) -> list[ConstraintResult]:
    """Build story and roof drift checks for one direct seismic direction."""

    checks = []
    for story, drift in enumerate(result.story_drifts, start=1):
        utilization = drift_utilization(
            drift_ratio=drift, allowable_ratio=policy.elastic_story_drift_limit
        )
        checks.append(
            ConstraintResult(
                check_id=f"{load_case}:story-{story}:story_drift",
                member_id=f"story-{story}",
                group_id="frame",
                story=story,
                member_type="frame",
                load_case=load_case,
                check_type="story_drift",
                demand=abs(drift),
                capacity=policy.elastic_story_drift_limit,
                utilization=utilization,
                passed=utilization <= 1.0,
                source="REF-A Table 2",
                details={"signed_drift_ratio": drift},
            )
        )
    total_height = sum(generated.definition.story_heights_m)
    roof_ratio = result.roof_displacement_m / total_height
    utilization = drift_utilization(
        drift_ratio=roof_ratio, allowable_ratio=policy.elastic_roof_drift_limit
    )
    checks.append(
        ConstraintResult(
            check_id=f"{load_case}:roof:roof_drift",
            member_id="roof",
            group_id="frame",
            story=len(generated.definition.story_heights_m),
            member_type="frame",
            load_case=load_case,
            check_type="roof_drift",
            demand=abs(roof_ratio),
            capacity=policy.elastic_roof_drift_limit,
            utilization=utilization,
            passed=utilization <= 1.0,
            source="REF-A Table 2",
            details={"signed_roof_drift_ratio": roof_ratio, "total_height_m": total_height},
        )
    )
    return checks


def _research_actuals(
    constraints: tuple[ConstraintResult, ...], generated: GeneratedFrame
) -> dict[str, float]:
    """Aggregate member constraints into the applicable REF-A table positions."""

    actuals: dict[str, float] = {}

    def position(result: ConstraintResult) -> Literal["corner", "middle"]:
        index = (
            int(result.member_id.rsplit("C", maxsplit=1)[1])
            if result.member_type == "column"
            else int(result.member_id.rsplit("B", maxsplit=1)[1])
        )
        last_index = (
            len(generated.definition.bay_widths_m)
            if result.member_type == "column"
            else len(generated.definition.bay_widths_m) - 1
        )
        outer = {0, last_index}
        return "corner" if index in outer else "middle"

    for story in range(1, len(generated.definition.floors) + 1):
        for member_type, check_type, label in (
            ("column", "column_interaction", "columns"),
            ("beam", "beam_flexure", "beams"),
        ):
            for location in ("corner", "middle"):
                gravity_values = [
                    item.utilization
                    for item in constraints
                    if item.load_case == "strength_gravity"
                    and item.story == story
                    and item.member_type == member_type
                    and item.check_type == check_type
                    and position(item) == location
                ]
                lateral_values = [
                    item.utilization
                    for item in constraints
                    if item.load_case in {"strength_seismic_positive", "strength_seismic_negative"}
                    and item.story == story
                    and item.member_type == member_type
                    and item.check_type == check_type
                    and position(item) == location
                ]
                actuals[f"gravity.story{story}.{label}.{location}"] = max(gravity_values)
                actuals[f"lateral.story{story}.{label}.{location}"] = max(lateral_values)
        for location in ("corner", "middle"):
            deflection_values = [
                item.utilization
                for item in constraints
                if item.story == story
                and item.check_type == "beam_deflection"
                and position(item) == location
            ]
            actuals[f"gravity.story{story}.deflection.{location}"] = max(deflection_values)
        drift_values = [
            item.utilization
            for item in constraints
            if item.story == story
            and item.check_type == "story_drift"
            and item.load_case in {"strength_seismic_positive", "strength_seismic_negative"}
        ]
        actuals[f"lateral.story{story}.drift"] = max(drift_values)
    return actuals


def evaluate_stage1_design(
    *,
    generated: GeneratedFrame,
    sections_by_group: dict[str, SectionRecord],
    case: Stage1Case,
    policy: Stage1Policy,
    reference: PublishedStage1Reference,
) -> Stage1Artifact:
    """Evaluate one explicit section assignment with the verified Stage 1 implementation."""

    if case.frame_id != generated.definition.frame_id:
        raise ValueError("Stage 1 case and generated frame identities differ")
    if set(sections_by_group) != set(generated.definition.design_groups):
        raise ValueError("Stage 1 evaluation requires one section for every design group")
    if {
        group_id: section.name for group_id, section in sections_by_group.items()
    } != case.sections_by_group:
        raise ValueError("Stage 1 case names and provided section records differ")
    modal = run_modal_analysis(generated, sections_by_group)
    acceleration = spectral_acceleration_g(
        modal.periods_s[0],
        short_period_g=case.spectrum.short_period_g,
        one_second_g=case.spectrum.one_second_g,
        fa=case.spectrum.fa,
        fv=case.spectrum.fv,
    )
    weights = tuple(floor.seismic_weight_n for floor in generated.definition.floors)
    base_shear = elastic_base_shear_n(sum(weights), acceleration, case.spectrum.r_factor)
    elevations: list[float] = []
    elevation = 0.0
    for height in generated.definition.story_heights_m:
        elevation += height
        elevations.append(elevation)
    story_forces = distribute_story_forces_n(
        base_shear, weights, elevations, exponent=case.spectrum.distribution_exponent
    )
    zeros = tuple(0.0 for _ in generated.definition.floors)
    analyses = {
        "strength_gravity": run_static_analysis(
            generated, sections_by_group, policy.strength_gravity, zeros, p_delta=True
        ),
        "strength_seismic_positive": run_static_analysis(
            generated, sections_by_group, policy.strength_seismic, story_forces, p_delta=True
        ),
        "strength_seismic_negative": run_static_analysis(
            generated,
            sections_by_group,
            policy.strength_seismic,
            tuple(-value for value in story_forces),
            p_delta=True,
        ),
        "service_D": run_static_analysis(
            generated, sections_by_group, policy.service_dead, zeros, p_delta=False
        ),
        "service_L": run_static_analysis(
            generated, sections_by_group, policy.service_live, zeros, p_delta=False
        ),
        "service_D_plus_L": run_static_analysis(
            generated, sections_by_group, policy.service_dead_plus_live, zeros, p_delta=False
        ),
    }
    for name, result in analyses.items():
        if result.equilibrium_relative_residual > policy.equilibrium_relative_tolerance:
            raise RuntimeError(
                f"{name} equilibrium residual {result.equilibrium_relative_residual} exceeds "
                f"{policy.equilibrium_relative_tolerance}"
            )
        if not result.combination_evaluated_directly:
            raise RuntimeError(f"{name} was not evaluated as a direct combination")

    effective = calculate_effective_lengths(generated, sections_by_group)
    effective_by_member = {item.member_id: item for item in effective}
    checks: list[ConstraintResult] = []
    for load_case in (
        "strength_gravity",
        "strength_seismic_positive",
        "strength_seismic_negative",
    ):
        checks.extend(
            _strength_constraints(
                load_case=load_case,
                result=analyses[load_case],
                generated=generated,
                sections_by_group=sections_by_group,
                policy=policy,
                effective_lengths=effective_by_member,
            )
        )
    for load_case in ("service_D", "service_L", "service_D_plus_L"):
        checks.extend(
            _service_constraints(
                load_case=load_case,
                result=analyses[load_case],
                generated=generated,
                denominator=policy.deflection_denominators[load_case],
            )
        )
    for load_case in ("strength_seismic_positive", "strength_seismic_negative"):
        checks.extend(
            _drift_constraints(
                load_case=load_case,
                result=analyses[load_case],
                generated=generated,
                policy=policy,
            )
        )
    constraints = tuple(checks)
    actuals = _research_actuals(constraints, generated)
    if set(actuals) != set(reference.metrics):
        raise RuntimeError("research comparison metric identities are incomplete")
    comparisons = tuple(
        ResearchComparison(
            metric_id=metric_id,
            actual=actuals[metric_id],
            published=published,
            signed_difference=actuals[metric_id] - published,
            relative_difference=(actuals[metric_id] - published) / published,
            disposition="diagnostic_not_an_acceptance_gate",
        )
        for metric_id, published in sorted(reference.metrics.items())
    )
    failed = tuple(item.check_id for item in constraints if not item.passed)
    return Stage1Artifact(
        review_status="not independently engineering-validated",
        domain_approval=policy.domain_approval,
        case=case,
        policy=policy,
        effective_lengths=effective,
        analyses=analyses,
        constraints=constraints,
        feasible=not failed,
        failed_constraint_ids=failed,
        maximum_utilization=max(item.utilization for item in constraints),
        research_comparisons=comparisons,
        research_comparison_status="recorded_with_no_agreement_claim",
    )


def build_stage1_artifact(
    *,
    frame_path: Path,
    sections_path: Path,
    case_path: Path,
    policy_path: Path,
    reference_path: Path,
) -> Stage1Artifact:
    """Load explicit inputs and run the reusable verified Stage 1 evaluator."""

    generated = generate_frame(FrameDefinition.from_json(frame_path))
    repository = SectionRepository.from_json(sections_path)
    case = Stage1Case.from_json(case_path)
    sections = {
        group_id: repository.get(section_name)
        for group_id, section_name in case.sections_by_group.items()
    }
    return evaluate_stage1_design(
        generated=generated,
        sections_by_group=sections,
        case=case,
        policy=Stage1Policy.from_json(policy_path),
        reference=PublishedStage1Reference.from_json(reference_path),
    )


def _parser() -> argparse.ArgumentParser:
    """Create the Phase 5 evidence command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--sections", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the Stage 1 artifact without converting research mismatch into failure."""

    arguments = _parser().parse_args(argv)
    artifact = build_stage1_artifact(
        frame_path=arguments.frame,
        sections_path=arguments.sections,
        case_path=arguments.case,
        policy_path=arguments.policy,
        reference_path=arguments.reference,
    )
    write_canonical_json(arguments.output, artifact)
    print(
        f"Stage 1 constraints={len(artifact.constraints)}, feasible={artifact.feasible}, "
        f"max_utilization={artifact.maximum_utilization:.6f}, "
        f"research_metrics={len(artifact.research_comparisons)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
