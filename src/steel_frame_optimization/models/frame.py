"""Generate deterministic two-dimensional frame topology and load audits from configuration.

Phase 3 deliberately stops at model definition. This module creates semantic nodes, members,
supports, mass allocations, and load records, but it does not build or solve an OpenSees model.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib
from pydantic import Field, model_validator

from steel_frame_optimization.config.schemas import (
    FrameSpec,
    MemberSpec,
    NodeSpec,
    StrictConfigModel,
    SupportSpec,
)
from steel_frame_optimization.sections.domains import CandidateDomain, maximum_domain_weight_n
from steel_frame_optimization.sections.repository import SectionRepository
from steel_frame_optimization.serialization import write_canonical_json

matplotlib.use("Agg")
from matplotlib import pyplot as plt


class MaterialAssignment(StrictConfigModel):
    """Store one source-driven member material assignment in canonical SI units."""

    member_type: Literal["beam", "column"]
    nominal_yield_strength_pa: float = Field(gt=0.0, allow_inf_nan=False)
    expected_yield_strength_pa: float = Field(gt=0.0, allow_inf_nan=False)
    elastic_modulus_pa: float = Field(gt=0.0, allow_inf_nan=False)
    grade_label: str = Field(min_length=1)
    grade_label_status: Literal["inferred_pending_layer4"]
    assumption_id: Literal["ASM-004"] = "ASM-004"


class FloorInput(StrictConfigModel):
    """Declare one elevated floor's published line loads and fixed seismic weight."""

    level: int = Field(gt=0)
    dead_line_load_n_per_m: float = Field(gt=0.0, allow_inf_nan=False)
    live_line_load_n_per_m: float = Field(gt=0.0, allow_inf_nan=False)
    seismic_weight_n: float = Field(gt=0.0, allow_inf_nan=False)


class FramePolicy(StrictConfigModel):
    """Make every Phase 3 boundary, diaphragm, mass, and self-weight choice explicit."""

    joint_model: Literal["rigid_centerline_no_panel_zone_or_offset"]
    base_support: Literal["fixed_ux_uy_rz"]
    floor_diaphragm: Literal["equal_horizontal_translation_per_level"]
    mass_distribution: Literal["tributary_width_to_floor_nodes"]
    rotational_mass: Literal["zero"]
    member_axial_deformation: Literal["retained"]
    member_shear_deformation: Literal["neglected"]
    modal_mass_source: Literal["fixed_published_story_weights"]
    candidate_member_mass_added_to_modal_mass: Literal[False]
    gravity_load_source: Literal["published_line_loads_only"]
    candidate_member_self_weight_added_to_gravity: Literal[False]
    objective_member_weight_accounting: Literal["section_weight_times_member_length"]
    mass_self_weight_assumption_id: Literal["ASM-020"] = "ASM-020"


class FrameDefinition(StrictConfigModel):
    """Define either source-transcribed orthogonal moment frame without enumerating nodes."""

    schema_version: Literal["1.0"] = "1.0"
    frame_id: Literal["3-story", "9-story"]
    domain_label: Literal["273-section domain, pending client confirmation"]
    story_heights_m: tuple[float, ...] = Field(min_length=1)
    bay_widths_m: tuple[float, ...] = Field(min_length=1)
    design_groups: tuple[str, ...] = Field(min_length=1)
    floors: tuple[FloorInput, ...] = Field(min_length=1)
    materials: tuple[MaterialAssignment, ...] = Field(min_length=2, max_length=2)
    policy: FramePolicy
    gravity_acceleration_m_per_s2: float = Field(gt=0.0, allow_inf_nan=False)
    column_group_policy: Literal["exterior_1_interior_2", "ref_a_figure_8_column_story_bands"]
    beam_group_policy: Literal["level_1_to_3_maps_group_3_to_5", "ref_a_figure_8_beam_level_bands"]
    beam_group_decision_id: Literal["DEC-017", "DEC-021"]

    @classmethod
    def from_json(cls, path: Path) -> FrameDefinition:
        """Load and strictly validate one frame-definition JSON file."""

        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))

    @model_validator(mode="after")
    def validate_levels_groups_and_materials(self) -> FrameDefinition:
        """Require complete source-specific groups, floors, geometry, and materials."""

        if any(height <= 0.0 for height in self.story_heights_m):
            raise ValueError("story heights must be positive")
        if any(width <= 0.0 for width in self.bay_widths_m):
            raise ValueError("bay widths must be positive")
        expected_levels = list(range(1, len(self.story_heights_m) + 1))
        if [floor.level for floor in self.floors] != expected_levels:
            raise ValueError("floor levels must be ordered and cover every story")
        expected_groups = (
            tuple(str(group) for group in range(1, 6))
            if self.frame_id == "3-story"
            else tuple(str(group) for group in range(1, 14))
        )
        if self.design_groups != expected_groups:
            raise ValueError(
                f"{self.frame_id} design groups must be exactly 1 through {len(expected_groups)}"
            )
        expected_shape = (3, 4) if self.frame_id == "3-story" else (9, 5)
        if (len(self.story_heights_m), len(self.bay_widths_m)) != expected_shape:
            raise ValueError(
                f"{self.frame_id} geometry must have {expected_shape[0]} stories and "
                f"{expected_shape[1]} bays"
            )
        expected_policies = (
            ("exterior_1_interior_2", "level_1_to_3_maps_group_3_to_5", "DEC-021")
            if self.frame_id == "3-story"
            else (
                "ref_a_figure_8_column_story_bands",
                "ref_a_figure_8_beam_level_bands",
                "DEC-017",
            )
        )
        if (
            self.column_group_policy,
            self.beam_group_policy,
            self.beam_group_decision_id,
        ) != expected_policies:
            raise ValueError(f"{self.frame_id} grouping policies do not match the source decision")
        if {material.member_type for material in self.materials} != {"beam", "column"}:
            raise ValueError("materials must define exactly one beam and one column assignment")
        return self


@dataclass(frozen=True, slots=True)
class NodeMass:
    """Store one translational nodal mass allocation derived from a fixed story weight."""

    node_id: str
    level: int
    tributary_fraction: float
    mass_kg: float


@dataclass(frozen=True, slots=True)
class MemberLineLoad:
    """Store published gravity line loads for one generated beam member."""

    member_id: str
    level: int
    dead_n_per_m: float
    live_n_per_m: float


@dataclass(frozen=True, slots=True)
class GeneratedFrame:
    """Bundle generated topology, deterministic tags, masses, and line-load records."""

    definition: FrameDefinition
    topology: FrameSpec
    node_tags: dict[str, int]
    member_tags: dict[str, int]
    node_masses: tuple[NodeMass, ...]
    member_line_loads: tuple[MemberLineLoad, ...]


class AuditNode(StrictConfigModel):
    """Persist one generated node and its deterministic solver tag."""

    node_id: str
    tag: int = Field(gt=0)
    x_m: float = Field(allow_inf_nan=False)
    y_m: float = Field(allow_inf_nan=False)
    restrained_dofs: tuple[str, ...]


class AuditMember(StrictConfigModel):
    """Persist one generated member's topology, orientation, group, length, and tag."""

    member_id: str
    tag: int = Field(gt=0)
    member_type: Literal["beam", "column"]
    group_id: str
    i_node: str
    j_node: str
    length_m: float = Field(gt=0.0, allow_inf_nan=False)
    orientation: Literal["left_to_right", "bottom_to_top"]
    dead_line_load_n_per_m: float = Field(ge=0.0, allow_inf_nan=False)
    live_line_load_n_per_m: float = Field(ge=0.0, allow_inf_nan=False)


class AuditFloor(StrictConfigModel):
    """Persist one floor's elevation, loads, seismic mass, and tributary allocation."""

    level: int = Field(gt=0)
    elevation_m: float = Field(gt=0.0, allow_inf_nan=False)
    dead_line_load_n_per_m: float = Field(gt=0.0, allow_inf_nan=False)
    live_line_load_n_per_m: float = Field(gt=0.0, allow_inf_nan=False)
    seismic_weight_n: float = Field(gt=0.0, allow_inf_nan=False)
    seismic_mass_kg: float = Field(gt=0.0, allow_inf_nan=False)
    node_mass_allocations_kg: dict[str, float]


class GroupLength(StrictConfigModel):
    """Persist the independently summed total member length for one design group."""

    group_id: str
    member_count: int = Field(gt=0)
    total_length_m: float = Field(gt=0.0, allow_inf_nan=False)


class WmaxCrossCheck(StrictConfigModel):
    """Record the geometry/domain Wmax diagnostic without tuning it to the paper tables."""

    domain_label: Literal["273-section domain, pending client confirmation"]
    domain_basis: Literal[
        "unfiltered_all_w_for_every_group", "unfiltered_w14_columns_and_all_w_beams"
    ]
    heaviest_section: str
    heaviest_weight_n_per_m: float = Field(gt=0.0, allow_inf_nan=False)
    computed_wmax_n: float = Field(gt=0.0, allow_inf_nan=False)
    back_calculated_reference_n: float = Field(gt=0.0, allow_inf_nan=False)
    signed_relative_difference: float = Field(allow_inf_nan=False)
    disposition: Literal["mismatch_inspect_domain_upper_bound_and_grouping"]


class TopologyAudit(StrictConfigModel):
    """Persist the complete Phase 3 topology/load audit and all aggregate invariants."""

    schema_version: Literal["1.0"] = "1.0"
    validation_status: Literal["not engineering-validated"]
    frame_id: Literal["3-story", "9-story"]
    domain_label: Literal["273-section domain, pending client confirmation"]
    nodes: tuple[AuditNode, ...]
    members: tuple[AuditMember, ...]
    floors: tuple[AuditFloor, ...]
    group_lengths: tuple[GroupLength, ...]
    node_count: int = Field(gt=0)
    member_count: int = Field(gt=0)
    column_count: int = Field(gt=0)
    beam_count: int = Field(gt=0)
    fixed_support_count: int = Field(gt=0)
    total_dead_load_n: float = Field(gt=0.0, allow_inf_nan=False)
    total_live_load_n: float = Field(gt=0.0, allow_inf_nan=False)
    total_seismic_weight_n: float = Field(gt=0.0, allow_inf_nan=False)
    total_seismic_mass_kg: float = Field(gt=0.0, allow_inf_nan=False)
    policy: FramePolicy
    wmax_cross_check: WmaxCrossCheck


def _node_id(level: int, column: int) -> str:
    """Return a stable semantic node ID independent of numerical solver tags."""

    return f"N-L{level}-C{column}"


def _beam_level(member_id: str) -> int:
    """Extract the configured floor level from a generated semantic beam ID."""

    return int(member_id.split("-")[1][1:])


def _column_group_id(definition: FrameDefinition, story: int, column: int) -> str:
    """Return the configured column group transcribed from the applicable source figure."""

    exterior = column in {0, len(definition.bay_widths_m)}
    if definition.frame_id == "3-story":
        return "1" if exterior else "2"
    if story <= 3:
        return "1" if exterior else ("2" if story == 1 else "3")
    if story <= 5:
        return "4" if exterior else "5"
    if story <= 7:
        return "6" if exterior else "7"
    return "8" if exterior else "6"


def _beam_group_id(definition: FrameDefinition, level: int) -> str:
    """Return the configured beam group transcribed from the applicable source figure."""

    if definition.frame_id == "3-story":
        return str(level + 2)
    if level <= 2:
        return "9"
    if level <= 6:
        return "10"
    return {7: "11", 8: "12", 9: "13"}[level]


def generate_frame(definition: FrameDefinition) -> GeneratedFrame:
    """Generate deterministic full-frame topology, loads, and masses from one definition."""

    x_coordinates = [0.0]
    for width in definition.bay_widths_m:
        x_coordinates.append(x_coordinates[-1] + width)
    y_coordinates = [0.0]
    for height in definition.story_heights_m:
        y_coordinates.append(y_coordinates[-1] + height)

    nodes = tuple(
        NodeSpec(node_id=_node_id(level, column), x_m=x, y_m=y_coordinates[level])
        for level in range(len(y_coordinates))
        for column, x in enumerate(x_coordinates)
    )
    supports = tuple(
        SupportSpec(node_id=_node_id(0, column), restrained_dofs=("ux", "uy", "rz"))
        for column in range(len(x_coordinates))
    )

    members: list[MemberSpec] = []
    for story in range(1, len(y_coordinates)):
        for column in range(len(x_coordinates)):
            group_id = _column_group_id(definition, story, column)
            members.append(
                MemberSpec(
                    member_id=f"C-S{story}-C{column}",
                    i_node=_node_id(story - 1, column),
                    j_node=_node_id(story, column),
                    group_id=group_id,
                    member_type="column",
                )
            )
    for level in range(1, len(y_coordinates)):
        for bay in range(len(definition.bay_widths_m)):
            members.append(
                MemberSpec(
                    member_id=f"B-L{level}-B{bay}",
                    i_node=_node_id(level, bay),
                    j_node=_node_id(level, bay + 1),
                    group_id=_beam_group_id(definition, level),
                    member_type="beam",
                )
            )

    topology = FrameSpec(
        frame_id=definition.frame_id,
        design_groups=definition.design_groups,
        nodes=nodes,
        members=tuple(members),
        supports=supports,
    )
    node_tags = {node.node_id: index for index, node in enumerate(nodes, start=1)}
    member_tags = {member.member_id: index for index, member in enumerate(members, start=1)}

    total_width = sum(definition.bay_widths_m)
    tributary_widths = (
        definition.bay_widths_m[0] / 2.0,
        *(
            (definition.bay_widths_m[index - 1] + definition.bay_widths_m[index]) / 2.0
            for index in range(1, len(definition.bay_widths_m))
        ),
        definition.bay_widths_m[-1] / 2.0,
    )
    node_masses = tuple(
        NodeMass(
            node_id=_node_id(floor.level, column),
            level=floor.level,
            tributary_fraction=width / total_width,
            mass_kg=(floor.seismic_weight_n / definition.gravity_acceleration_m_per_s2)
            * width
            / total_width,
        )
        for floor in definition.floors
        for column, width in enumerate(tributary_widths)
    )
    floor_by_level = {floor.level: floor for floor in definition.floors}
    member_line_loads = tuple(
        MemberLineLoad(
            member_id=member.member_id,
            level=_beam_level(member.member_id),
            dead_n_per_m=floor_by_level[_beam_level(member.member_id)].dead_line_load_n_per_m,
            live_n_per_m=floor_by_level[_beam_level(member.member_id)].live_line_load_n_per_m,
        )
        for member in topology.members
        if member.member_type == "beam"
    )
    return GeneratedFrame(
        definition=definition,
        topology=topology,
        node_tags=node_tags,
        member_tags=member_tags,
        node_masses=node_masses,
        member_line_loads=member_line_loads,
    )


def _member_length(member: MemberSpec, nodes: dict[str, NodeSpec]) -> float:
    """Return one member's Euclidean centerline length in metres."""

    i_node = nodes[member.i_node]
    j_node = nodes[member.j_node]
    return math.hypot(j_node.x_m - i_node.x_m, j_node.y_m - i_node.y_m)


def build_topology_audit(
    generated: GeneratedFrame, section_repository: SectionRepository
) -> TopologyAudit:
    """Build a complete deterministic audit, including the required Wmax cross-check."""

    nodes = {node.node_id: node for node in generated.topology.nodes}
    supports = {support.node_id: support.restrained_dofs for support in generated.topology.supports}
    loads = {load.member_id: load for load in generated.member_line_loads}
    audit_nodes = tuple(
        AuditNode(
            node_id=node.node_id,
            tag=generated.node_tags[node.node_id],
            x_m=node.x_m,
            y_m=node.y_m,
            restrained_dofs=supports.get(node.node_id, ()),
        )
        for node in generated.topology.nodes
    )
    audit_members = tuple(
        AuditMember(
            member_id=member.member_id,
            tag=generated.member_tags[member.member_id],
            member_type=member.member_type,
            group_id=member.group_id,
            i_node=member.i_node,
            j_node=member.j_node,
            length_m=_member_length(member, nodes),
            orientation="left_to_right" if member.member_type == "beam" else "bottom_to_top",
            dead_line_load_n_per_m=loads[member.member_id].dead_n_per_m
            if member.member_type == "beam"
            else 0.0,
            live_line_load_n_per_m=loads[member.member_id].live_n_per_m
            if member.member_type == "beam"
            else 0.0,
        )
        for member in generated.topology.members
    )
    masses_by_level: dict[int, list[NodeMass]] = defaultdict(list)
    for allocation in generated.node_masses:
        masses_by_level[allocation.level].append(allocation)
    elevations = []
    elevation = 0.0
    for height in generated.definition.story_heights_m:
        elevation += height
        elevations.append(elevation)
    floors = tuple(
        AuditFloor(
            level=floor.level,
            elevation_m=elevations[floor.level - 1],
            dead_line_load_n_per_m=floor.dead_line_load_n_per_m,
            live_line_load_n_per_m=floor.live_line_load_n_per_m,
            seismic_weight_n=floor.seismic_weight_n,
            seismic_mass_kg=floor.seismic_weight_n
            / generated.definition.gravity_acceleration_m_per_s2,
            node_mass_allocations_kg={
                allocation.node_id: allocation.mass_kg
                for allocation in masses_by_level[floor.level]
            },
        )
        for floor in generated.definition.floors
    )
    lengths_by_group: dict[str, list[float]] = defaultdict(list)
    for member in audit_members:
        lengths_by_group[member.group_id].append(member.length_m)
    group_lengths = tuple(
        GroupLength(
            group_id=group_id,
            member_count=len(lengths_by_group[group_id]),
            total_length_m=sum(lengths_by_group[group_id]),
        )
        for group_id in generated.definition.design_groups
    )
    domains = {
        group.group_id: CandidateDomain.from_family(
            group.group_id,
            "w14"
            if generated.definition.frame_id == "9-story" and int(group.group_id) <= 8
            else "all_w",
            section_repository,
        )
        for group in group_lengths
    }
    lengths = {group.group_id: group.total_length_m for group in group_lengths}
    computed_wmax = maximum_domain_weight_n(
        member_length_m_by_group=lengths,
        domains_by_group=domains,
    )
    heaviest = max(section_repository.family("all_w"), key=lambda record: record.weight_n_per_m)
    reference_wmax_n = 1_974_010.0 if generated.definition.frame_id == "3-story" else 7_185_870.0
    total_dead = sum(member.dead_line_load_n_per_m * member.length_m for member in audit_members)
    total_live = sum(member.live_line_load_n_per_m * member.length_m for member in audit_members)
    return TopologyAudit(
        validation_status="not engineering-validated",
        frame_id=generated.definition.frame_id,
        domain_label=generated.definition.domain_label,
        nodes=audit_nodes,
        members=audit_members,
        floors=floors,
        group_lengths=group_lengths,
        node_count=len(audit_nodes),
        member_count=len(audit_members),
        column_count=sum(member.member_type == "column" for member in audit_members),
        beam_count=sum(member.member_type == "beam" for member in audit_members),
        fixed_support_count=len(generated.topology.supports),
        total_dead_load_n=total_dead,
        total_live_load_n=total_live,
        total_seismic_weight_n=sum(floor.seismic_weight_n for floor in floors),
        total_seismic_mass_kg=sum(floor.seismic_mass_kg for floor in floors),
        policy=generated.definition.policy,
        wmax_cross_check=WmaxCrossCheck(
            domain_label=generated.definition.domain_label,
            domain_basis=(
                "unfiltered_all_w_for_every_group"
                if generated.definition.frame_id == "3-story"
                else "unfiltered_w14_columns_and_all_w_beams"
            ),
            heaviest_section=heaviest.name,
            heaviest_weight_n_per_m=heaviest.weight_n_per_m,
            computed_wmax_n=computed_wmax,
            back_calculated_reference_n=reference_wmax_n,
            signed_relative_difference=(computed_wmax - reference_wmax_n) / reference_wmax_n,
            disposition="mismatch_inspect_domain_upper_bound_and_grouping",
        ),
    )


def save_topology_figure(generated: GeneratedFrame, output_path: Path) -> None:
    """Render a deterministic configuration-derived SVG of geometry, groups, and floor loads."""

    nodes = {node.node_id: node for node in generated.topology.nodes}
    palette = tuple(plt.get_cmap("tab20")(index / 19.0) for index in range(20))
    colors = {
        group_id: palette[index % len(palette)]
        for index, group_id in enumerate(generated.definition.design_groups)
    }
    with matplotlib.rc_context({"svg.hashsalt": "steel-frame-phase3"}):
        figure, axis = plt.subplots(figsize=(10, 4.8))
        for member in generated.topology.members:
            i_node = nodes[member.i_node]
            j_node = nodes[member.j_node]
            axis.plot(
                (i_node.x_m, j_node.x_m),
                (i_node.y_m, j_node.y_m),
                color=colors[member.group_id],
                linewidth=2.4,
            )
            axis.text(
                (i_node.x_m + j_node.x_m) / 2.0,
                (i_node.y_m + j_node.y_m) / 2.0,
                member.group_id,
                fontsize=7,
                ha="center",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
            )
        for node in generated.topology.nodes:
            axis.plot(node.x_m, node.y_m, "ko", markersize=2.5)
        support_half_width = 0.05 * min(generated.definition.bay_widths_m)
        hatch_spacing = support_half_width / 2.5
        for support in generated.topology.supports:
            node = nodes[support.node_id]
            stem_bottom = node.y_m - 0.12 * min(generated.definition.story_heights_m)
            axis.plot((node.x_m, node.x_m), (node.y_m, stem_bottom), color="black", linewidth=2.0)
            axis.plot(
                (node.x_m - support_half_width, node.x_m + support_half_width),
                (stem_bottom, stem_bottom),
                color="black",
                linewidth=4.0,
            )
            for offset in (-2, -1, 0, 1, 2):
                x_value = node.x_m + offset * hatch_spacing
                axis.plot(
                    (x_value - hatch_spacing / 2.0, x_value),
                    (stem_bottom - hatch_spacing / 2.0, stem_bottom),
                    color="black",
                    linewidth=0.8,
                )
        for floor in generated.definition.floors:
            y = sum(generated.definition.story_heights_m[: floor.level])
            axis.text(
                sum(generated.definition.bay_widths_m) + 0.5,
                y,
                f"L{floor.level}: D={floor.dead_line_load_n_per_m / 1000:g}, "
                f"L={floor.live_line_load_n_per_m / 1000:g} kN/m; "
                f"W={floor.seismic_weight_n / 1000:g} kN",
                fontsize=8,
                va="center",
            )
        axis.set_aspect("equal")
        axis.set_xlabel("Global X (m)")
        axis.set_ylabel("Global Y (m)")
        axis.set_title(
            f"{generated.definition.frame_id} frame: group IDs, fixed bases, "
            "and published floor loads"
        )
        axis.grid(alpha=0.2)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, format="svg", metadata={"Date": None})
        plt.close(figure)
    # Matplotlib follows the host newline convention; artifacts must hash identically across OSes.
    svg_text = output_path.read_text(encoding="utf-8")
    output_path.write_text(svg_text, encoding="utf-8", newline="\n")


def _parser() -> argparse.ArgumentParser:
    """Create the Phase 3 deterministic artifact-generation command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sections", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the Phase 3 topology audit and SVG without running structural analysis."""

    arguments = _parser().parse_args(argv)
    definition = FrameDefinition.from_json(arguments.config)
    generated = generate_frame(definition)
    repository = SectionRepository.from_json(arguments.sections)
    audit = build_topology_audit(generated, repository)
    write_canonical_json(arguments.audit, audit)
    save_topology_figure(generated, arguments.figure)
    print(
        f"wrote {audit.node_count} nodes, {audit.member_count} members, "
        f"and Wmax={audit.wmax_cross_check.computed_wmax_n:.3f} N"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
