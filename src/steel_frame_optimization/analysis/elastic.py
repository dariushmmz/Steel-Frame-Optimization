"""Build and solve deterministic two-dimensional elastic OpenSees frame models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import openseespy.opensees as ops
from pydantic import Field

from steel_frame_optimization.analysis.modal import periods_from_eigenvalues
from steel_frame_optimization.config.schemas import StrictConfigModel
from steel_frame_optimization.models.frame import GeneratedFrame
from steel_frame_optimization.sections.records import SectionRecord


class LoadFactors(StrictConfigModel):
    """Declare one linear load case as factors on configured D, L, and E records."""

    dead: float = Field(allow_inf_nan=False)
    live: float = Field(allow_inf_nan=False)
    seismic: float = Field(allow_inf_nan=False)


class ModalResult(StrictConfigModel):
    """Persist eigenvalues, periods, and roof-positive normalized story mode components."""

    eigenvalues_per_s2: tuple[float, ...]
    periods_s: tuple[float, ...]
    normalized_story_modes: tuple[tuple[float, ...], ...]


class ElementForceResult(StrictConfigModel):
    """Persist one analysis element's global and local end-force vectors."""

    analysis_element_tag: int
    semantic_member_id: str
    global_end_forces_n_nm: tuple[float, ...]
    local_end_forces_n_nm: tuple[float, ...]


class StaticResult(StrictConfigModel):
    """Persist deterministic linear-static response and equilibrium diagnostics."""

    converged: Literal[True]
    p_delta: bool
    node_displacements_m_rad: dict[str, tuple[float, float, float]]
    beam_midspan_vertical_displacements_m: dict[str, float]
    support_reactions_n_nm: dict[str, tuple[float, float, float]]
    element_forces: tuple[ElementForceResult, ...]
    applied_force_x_n: float
    applied_force_y_n: float
    reaction_force_x_n: float
    reaction_force_y_n: float
    equilibrium_relative_residual: float
    roof_displacement_m: float
    story_drifts: tuple[float, ...]
    model_node_count: int
    model_element_count: int
    load_stages: tuple[Literal["gravity_converged_and_held"], Literal["lateral_converged"]]
    combination_evaluated_directly: Literal[True]


@dataclass(frozen=True, slots=True)
class _AnalysisIndex:
    """Map semantic records to deterministic OpenSees tags for one transient model build."""

    node_tags: dict[str, int]
    beam_midpoint_tags: dict[str, int]
    element_members: dict[int, str]
    beam_element_tags: dict[str, tuple[int, int]]


def _three_values(values: list[float]) -> tuple[float, float, float]:
    """Convert one OpenSees planar node vector to a statically typed triple."""

    if len(values) != 3:
        raise RuntimeError(f"expected three planar DOF values, received {len(values)}")
    return (float(values[0]), float(values[1]), float(values[2]))


def _build_model(
    generated: GeneratedFrame,
    sections_by_group: dict[str, SectionRecord],
    *,
    p_delta: bool,
) -> _AnalysisIndex:
    """Create a fresh OpenSees model with beam mid-nodes and explicit diaphragm constraints."""

    if set(sections_by_group) != set(generated.definition.design_groups):
        raise ValueError("one section is required for every design group")
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    node_tags = dict(generated.node_tags)
    nodes = {node.node_id: node for node in generated.topology.nodes}
    for node in generated.topology.nodes:
        ops.node(node_tags[node.node_id], node.x_m, node.y_m)
    for support in generated.topology.supports:
        ops.fix(node_tags[support.node_id], 1, 1, 1)
    for allocation in generated.node_masses:
        ops.mass(node_tags[allocation.node_id], allocation.mass_kg, 0.0, 0.0)

    next_node_tag = max(node_tags.values()) + 1
    beam_midpoint_tags: dict[str, int] = {}
    for member in generated.topology.members:
        if member.member_type != "beam":
            continue
        i_node = nodes[member.i_node]
        j_node = nodes[member.j_node]
        ops.node(
            next_node_tag,
            (i_node.x_m + j_node.x_m) / 2.0,
            (i_node.y_m + j_node.y_m) / 2.0,
        )
        beam_midpoint_tags[member.member_id] = next_node_tag
        next_node_tag += 1

    columns_per_level = len(generated.definition.bay_widths_m) + 1
    for floor in generated.definition.floors:
        master = node_tags[f"N-L{floor.level}-C0"]
        for column in range(1, columns_per_level):
            ops.equalDOF(master, node_tags[f"N-L{floor.level}-C{column}"], 1)
        for member in generated.topology.members:
            if member.member_type == "beam" and member.member_id.startswith(f"B-L{floor.level}-"):
                ops.equalDOF(master, beam_midpoint_tags[member.member_id], 1)

    ops.geomTransf("PDelta" if p_delta else "Linear", 1)
    element_members: dict[int, str] = {}
    beam_element_tags: dict[str, tuple[int, int]] = {}
    next_element_tag = 1
    elastic_modulus_pa = generated.definition.materials[0].elastic_modulus_pa
    for member in generated.topology.members:
        section = sections_by_group[member.group_id]
        i_tag = node_tags[member.i_node]
        j_tag = node_tags[member.j_node]
        if member.member_type == "column":
            ops.element(
                "elasticBeamColumn",
                next_element_tag,
                i_tag,
                j_tag,
                section.area_m2,
                elastic_modulus_pa,
                section.ix_m4,
                1,
            )
            element_members[next_element_tag] = member.member_id
            next_element_tag += 1
        else:
            midpoint_tag = beam_midpoint_tags[member.member_id]
            pair = (next_element_tag, next_element_tag + 1)
            for element_tag, end_tags in zip(
                pair, ((i_tag, midpoint_tag), (midpoint_tag, j_tag)), strict=True
            ):
                ops.element(
                    "elasticBeamColumn",
                    element_tag,
                    *end_tags,
                    section.area_m2,
                    elastic_modulus_pa,
                    section.ix_m4,
                    1,
                )
                element_members[element_tag] = member.member_id
            beam_element_tags[member.member_id] = pair
            next_element_tag += 2
    return _AnalysisIndex(node_tags, beam_midpoint_tags, element_members, beam_element_tags)


def run_modal_analysis(
    generated: GeneratedFrame,
    sections_by_group: dict[str, SectionRecord],
    *,
    mode_count: int = 3,
) -> ModalResult:
    """Solve the frame eigenproblem and normalize every mode to a positive unit roof component."""

    if mode_count <= 0:
        raise ValueError("mode count must be positive")
    index = _build_model(generated, sections_by_group, p_delta=False)
    try:
        raw = ops.eigen("-fullGenLapack", mode_count)
        eigenvalues = (
            (float(raw),) if isinstance(raw, float) else tuple(float(value) for value in raw)
        )
        periods = periods_from_eigenvalues(eigenvalues)
        modes: list[tuple[float, ...]] = []
        roof_tag = index.node_tags[f"N-L{len(generated.definition.floors)}-C0"]
        for mode in range(1, mode_count + 1):
            roof_value = float(ops.nodeEigenvector(roof_tag, mode, 1))
            if abs(roof_value) <= 1e-15:
                raise RuntimeError(f"mode {mode} has zero roof component")
            modes.append(
                tuple(
                    float(ops.nodeEigenvector(index.node_tags[f"N-L{level}-C0"], mode, 1))
                    / roof_value
                    for level in range(1, len(generated.definition.floors) + 1)
                )
            )
        return ModalResult(
            eigenvalues_per_s2=eigenvalues,
            periods_s=periods,
            normalized_story_modes=tuple(modes),
        )
    finally:
        ops.wipe()


def run_static_analysis(
    generated: GeneratedFrame,
    sections_by_group: dict[str, SectionRecord],
    factors: LoadFactors,
    story_forces_n: tuple[float, ...],
    *,
    p_delta: bool,
) -> StaticResult:
    """Solve one direct D/L/E combination by holding gravity before the lateral stage."""

    if len(story_forces_n) != len(generated.definition.floors):
        raise ValueError("one lateral force is required per story")
    index = _build_model(generated, sections_by_group, p_delta=p_delta)
    try:
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        line_loads = {load.member_id: load for load in generated.member_line_loads}
        members = {member.member_id: member for member in generated.topology.members}
        nodes = {node.node_id: node for node in generated.topology.nodes}
        applied_y = 0.0
        for member_id, element_tags in index.beam_element_tags.items():
            load = line_loads[member_id]
            intensity = factors.dead * load.dead_n_per_m + factors.live * load.live_n_per_m
            for element_tag in element_tags:
                ops.eleLoad("-ele", element_tag, "-type", "-beamUniform", -intensity)
            member = members[member_id]
            i_node = nodes[member.i_node]
            j_node = nodes[member.j_node]
            applied_y -= intensity * math.hypot(j_node.x_m - i_node.x_m, j_node.y_m - i_node.y_m)

        _configure_static_analysis()
        if ops.analyze(1) != 0:
            raise RuntimeError("OpenSees gravity stage did not converge")
        ops.loadConst("-time", 0.0)

        fractions = [mass.tributary_fraction for mass in generated.node_masses if mass.level == 1]
        ops.timeSeries("Linear", 2)
        ops.pattern("Plain", 2, 2)
        applied_x = 0.0
        for floor, story_force in zip(generated.definition.floors, story_forces_n, strict=True):
            force = factors.seismic * story_force
            applied_x += force
            for column, fraction in enumerate(fractions):
                ops.load(
                    index.node_tags[f"N-L{floor.level}-C{column}"],
                    force * fraction,
                    0.0,
                    0.0,
                )

        ops.wipeAnalysis()
        _configure_static_analysis()
        if ops.analyze(1) != 0:
            raise RuntimeError("OpenSees lateral stage did not converge")
        ops.reactions()

        displacements = {
            node_id: _three_values(ops.nodeDisp(tag)) for node_id, tag in index.node_tags.items()
        }
        midpoint_displacements = {
            member_id: float(ops.nodeDisp(tag, 2))
            for member_id, tag in index.beam_midpoint_tags.items()
        }
        reactions = {
            support.node_id: _three_values(ops.nodeReaction(index.node_tags[support.node_id]))
            for support in generated.topology.supports
        }
        element_forces = tuple(
            ElementForceResult(
                analysis_element_tag=tag,
                semantic_member_id=member_id,
                global_end_forces_n_nm=tuple(float(value) for value in ops.eleForce(tag)),
                local_end_forces_n_nm=tuple(
                    float(value) for value in ops.eleResponse(tag, "localForce")
                ),
            )
            for tag, member_id in index.element_members.items()
        )
        reaction_x = sum(value[0] for value in reactions.values())
        reaction_y = sum(value[1] for value in reactions.values())
        residual = math.hypot(applied_x + reaction_x, applied_y + reaction_y)
        scale = max(math.hypot(applied_x, applied_y), 1.0)
        master_displacements = [
            displacements[f"N-L{floor.level}-C0"][0] for floor in generated.definition.floors
        ]
        story_drifts = tuple(
            (displacement - (master_displacements[index_value - 1] if index_value else 0.0))
            / generated.definition.story_heights_m[index_value]
            for index_value, displacement in enumerate(master_displacements)
        )
        return StaticResult(
            converged=True,
            p_delta=p_delta,
            node_displacements_m_rad=displacements,
            beam_midspan_vertical_displacements_m=midpoint_displacements,
            support_reactions_n_nm=reactions,
            element_forces=element_forces,
            applied_force_x_n=applied_x,
            applied_force_y_n=applied_y,
            reaction_force_x_n=reaction_x,
            reaction_force_y_n=reaction_y,
            equilibrium_relative_residual=residual / scale,
            roof_displacement_m=master_displacements[-1],
            story_drifts=story_drifts,
            model_node_count=len(index.node_tags) + len(index.beam_midpoint_tags),
            model_element_count=len(index.element_members),
            load_stages=("gravity_converged_and_held", "lateral_converged"),
            combination_evaluated_directly=True,
        )
    finally:
        ops.wipe()


def _configure_static_analysis() -> None:
    """Configure a Newton solve that re-forms the tangent for each explicit load stage."""

    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1e-12, 50)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
