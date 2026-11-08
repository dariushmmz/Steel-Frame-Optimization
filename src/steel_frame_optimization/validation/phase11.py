"""Validate the nine-story topology, preliminary domains, modal model, and Stage 1 checks.

The module keeps the REF-A comparison values diagnostic.  It does not adjust geometry, loads,
sections, filter thresholds, or analysis assumptions to reproduce the publication.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from steel_frame_optimization.analysis.elastic import ModalResult, run_modal_analysis
from steel_frame_optimization.config.schemas import StrictConfigModel
from steel_frame_optimization.constraints.lrfd import (
    compact_beam_design_flexural_strength_nm,
    nominal_compression_strength_n,
)
from steel_frame_optimization.models.frame import (
    FrameDefinition,
    GeneratedFrame,
    TopologyAudit,
    build_topology_audit,
    generate_frame,
    save_topology_figure,
)
from steel_frame_optimization.sections.filtering import (
    BeamFilterPolicy,
    ColumnFilterPolicy,
    GroupFilterAudit,
    RejectedSection,
)
from steel_frame_optimization.sections.records import SectionRecord
from steel_frame_optimization.sections.repository import SectionRepository
from steel_frame_optimization.serialization import write_canonical_json
from steel_frame_optimization.validation.phase5 import (
    PublishedStage1Reference,
    Stage1Artifact,
    Stage1Case,
    Stage1Policy,
    evaluate_stage1_design,
)


class NineStoryFilterPolicy(StrictConfigModel):
    """Freeze the source comparison and project filter arithmetic for all thirteen groups."""

    schema_version: Literal["1.0"] = "1.0"
    frame_id: Literal["9-story"]
    domain_label: Literal["273-section domain, pending client confirmation"]
    source: str = Field(min_length=1)
    beam: BeamFilterPolicy
    column: ColumnFilterPolicy
    published_removed_counts: dict[str, int]
    comparison_policy: Literal[
        "research diagnostic only; approved project calculations are not fitted to published counts"
    ]

    @classmethod
    def from_json(cls, path: Path) -> NineStoryFilterPolicy:
        """Load and strictly validate the nine-story preliminary-filter policy."""

        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))

    @model_validator(mode="after")
    def require_published_counts(self) -> NineStoryFilterPolicy:
        """Keep the thirteen printed removal counts exact and visible."""

        expected = {
            "1": 6,
            "2": 12,
            "3": 10,
            "4": 2,
            "5": 8,
            "6": 1,
            "7": 2,
            "8": 1,
            "9": 42,
            "10": 42,
            "11": 42,
            "12": 42,
            "13": 38,
        }
        if self.published_removed_counts != expected:
            raise ValueError("nine-story published removal counts differ from REF-A")
        return self


class Figure8TranscriptionAudit(StrictConfigModel):
    """Record an independently enumerated REF-A Figure 8 topology comparison."""

    source: Literal["REF-A PDF p. 10 / journal p. 86, Figure 8"]
    bays: Literal[5]
    stories: Literal[9]
    nodes: Literal[60]
    members: Literal[99]
    columns: Literal[54]
    beams: Literal[45]
    fixed_supports: Literal[6]
    width_m: float = Field(gt=0.0)
    height_m: float = Field(gt=0.0)
    group_member_counts: dict[str, int]
    expected_group_member_counts: dict[str, int]
    exact_member_group_map_match: Literal[True]
    independent_audit_passed: Literal[True]


class NineStoryFilterArtifact(StrictConfigModel):
    """Persist project filter domains and the nonblocking published-count diagnostic."""

    schema_version: Literal["1.0"] = "1.0"
    validation_status: Literal["not independently engineering-validated"]
    domain_label: Literal["273-section domain, pending client confirmation"]
    column_base_domain_size: Literal[36]
    beam_base_domain_size: Literal[273]
    groups: tuple[GroupFilterAudit, ...] = Field(min_length=13, max_length=13)
    critical_member_by_group: dict[str, str]
    actual_removed_counts: dict[str, int]
    published_removed_counts: dict[str, int]
    published_counts_match: bool
    disposition: Literal["project_baseline_recorded_publication_diagnostic"]


class Phase11Artifact(StrictConfigModel):
    """Bundle every deterministic Phase 11 result without claiming engineering certification."""

    schema_version: Literal["1.0"] = "1.0"
    validation_status: Literal["verified research software; not engineering-certified"]
    transcription: Figure8TranscriptionAudit
    topology: TopologyAudit
    modal: ModalResult
    preliminary_filter: NineStoryFilterArtifact
    published_case2_stage1: Stage1Artifact
    three_story_model_changed: Literal[False]


@dataclass(frozen=True, slots=True)
class _ScreeningContext:
    """Hold one member-specific demand and candidate capacities during group screening."""

    member_id: str
    demand: float
    capacities: Mapping[str, float]


def _expected_figure8_groups(generated: GeneratedFrame) -> dict[str, str]:
    """Independently enumerate the exact story-band mapping visible in REF-A Figure 8."""

    exterior_columns = ("1", "1", "1", "4", "4", "6", "6", "8", "8")
    interior_columns = ("2", "3", "3", "5", "5", "7", "7", "6", "6")
    beam_levels = ("9", "9", "10", "10", "10", "10", "11", "12", "13")
    final_column = len(generated.definition.bay_widths_m)
    expected: dict[str, str] = {}
    for story in range(1, 10):
        for column in range(final_column + 1):
            expected[f"C-S{story}-C{column}"] = (
                exterior_columns[story - 1]
                if column in {0, final_column}
                else interior_columns[story - 1]
            )
        for bay in range(final_column):
            expected[f"B-L{story}-B{bay}"] = beam_levels[story - 1]
    return expected


def audit_figure8_transcription(generated: GeneratedFrame) -> Figure8TranscriptionAudit:
    """Compare generated topology against an independent literal transcription of Figure 8."""

    expected = _expected_figure8_groups(generated)
    actual = {member.member_id: member.group_id for member in generated.topology.members}
    if actual != expected:
        missing = sorted(set(expected) ^ set(actual))
        mismatched = sorted(
            key for key in set(expected) & set(actual) if expected[key] != actual[key]
        )
        raise RuntimeError(
            f"Figure 8 transcription mismatch; missing={missing}, mismatched={mismatched}"
        )
    counts = dict(sorted(Counter(actual.values()).items(), key=lambda item: int(item[0])))
    expected_counts = {
        "1": 6,
        "2": 4,
        "3": 8,
        "4": 4,
        "5": 8,
        "6": 12,
        "7": 8,
        "8": 4,
        "9": 10,
        "10": 20,
        "11": 5,
        "12": 5,
        "13": 5,
    }
    if counts != expected_counts:
        raise RuntimeError(f"Figure 8 group counts differ: {counts}")
    observed_counts = (
        len(generated.topology.nodes),
        len(generated.topology.members),
        sum(member.member_type == "column" for member in generated.topology.members),
        sum(member.member_type == "beam" for member in generated.topology.members),
        len(generated.topology.supports),
    )
    if observed_counts != (60, 99, 54, 45, 6):
        raise RuntimeError(f"Figure 8 aggregate counts differ: {observed_counts}")
    return Figure8TranscriptionAudit(
        source="REF-A PDF p. 10 / journal p. 86, Figure 8",
        bays=5,
        stories=9,
        nodes=60,
        members=99,
        columns=54,
        beams=45,
        fixed_supports=6,
        width_m=sum(generated.definition.bay_widths_m),
        height_m=sum(generated.definition.story_heights_m),
        group_member_counts=counts,
        expected_group_member_counts=expected_counts,
        exact_member_group_map_match=True,
        independent_audit_passed=True,
    )


def _group_screening_audit(
    *,
    group_id: str,
    member_type: Literal["beam", "column"],
    demand_kind: Literal["maximum_moment_nm", "bottom_story_axial_force_n"],
    records: tuple[SectionRecord, ...],
    contexts: Sequence[_ScreeningContext],
    published_removed_count: int,
) -> tuple[GroupFilterAudit, str]:
    """Select the member producing the greatest eliminations, as stated by REF-A."""

    candidates: list[tuple[int, str, _ScreeningContext, SectionRecord, float]] = []
    for context in contexts:
        ordered = sorted(
            records,
            key=lambda record: (context.capacities[record.name], record.source_row_id),
        )
        threshold = next(
            (record for record in ordered if context.capacities[record.name] >= context.demand),
            None,
        )
        if threshold is None:
            raise ValueError(f"no section satisfies {group_id} demand for {context.member_id}")
        removed = sum(context.capacities[record.name] < context.demand for record in records)
        candidates.append(
            (removed, context.member_id, context, threshold, context.capacities[threshold.name])
        )
    removed_count, member_id, critical, threshold, threshold_capacity = max(
        candidates, key=lambda item: (item[0], item[1])
    )
    reason: Literal[
        "design_flexural_strength_below_demand", "design_axial_strength_below_demand"
    ] = (
        "design_flexural_strength_below_demand"
        if member_type == "beam"
        else "design_axial_strength_below_demand"
    )
    rejected = tuple(
        RejectedSection(
            source_row_id=record.source_row_id,
            section_name=record.name,
            reason=reason,
            demand=critical.demand,
            design_capacity=critical.capacities[record.name],
            threshold_section=threshold.name,
            source_dataset_id="aisc-v14.1-w-dev-001",
        )
        for record in sorted(
            records,
            key=lambda record: (critical.capacities[record.name], record.source_row_id),
        )
        if critical.capacities[record.name] < critical.demand
    )
    return (
        GroupFilterAudit(
            group_id=group_id,
            member_type=member_type,
            demand_kind=demand_kind,
            demand=critical.demand,
            temporary_ordering="design_capacity_then_source_row",
            persistent_operator_ordering="weight_then_source_row",
            threshold_section=threshold.name,
            threshold_design_capacity=threshold_capacity,
            actual_removed_count=removed_count,
            published_removed_count=published_removed_count,
            matches_published_count=removed_count == published_removed_count,
            rejected_sections=rejected,
        ),
        member_id,
    )


def build_nine_story_filter(
    *,
    generated: GeneratedFrame,
    repository: SectionRepository,
    policy: NineStoryFilterPolicy,
) -> NineStoryFilterArtifact:
    """Run all thirteen preliminary screens with W14 columns and all-W beams."""

    if generated.definition.frame_id != "9-story":
        raise ValueError("nine-story filtering requires the nine-story frame")
    column_records = repository.family("w14")
    beam_records = repository.family("all_w")
    contexts: dict[str, list[_ScreeningContext]] = defaultdict(list)
    column = policy.column
    beam = policy.beam
    for member in generated.topology.members:
        story = int(member.member_id.split("-")[1][1:])
        if member.member_type == "column":
            column_index = int(member.member_id.rsplit("C", maxsplit=1)[1])
            reaction_multiplier = (
                0.5 if column_index in {0, len(generated.definition.bay_widths_m)} else 1.0
            )
            demand = (
                reaction_multiplier
                * generated.definition.bay_widths_m[0]
                * sum(
                    column.dead_load_factor * floor.dead_line_load_n_per_m
                    + column.live_load_factor * floor.live_line_load_n_per_m
                    for floor in generated.definition.floors[story - 1 :]
                )
            )
            length = generated.definition.story_heights_m[story - 1]
            capacities = {
                record.name: column.resistance_factor
                * nominal_compression_strength_n(
                    area_m2=record.area_m2,
                    effective_length_factor=column.preliminary_effective_length_factor,
                    length_m=length,
                    radius_of_gyration_m=record.rx_m,
                    yield_strength_pa=column.yield_strength_pa,
                    elastic_modulus_pa=column.elastic_modulus_pa,
                )
                for record in column_records
            }
        else:
            floor = generated.definition.floors[story - 1]
            factored_load = (
                beam.dead_load_factor * floor.dead_line_load_n_per_m
                + beam.live_load_factor * floor.live_line_load_n_per_m
            )
            demand = (
                beam.maximum_moment_coefficient
                * factored_load
                * generated.definition.bay_widths_m[0] ** 2
            )
            capacities = {
                record.name: compact_beam_design_flexural_strength_nm(
                    plastic_section_modulus_m3=record.zx_m3,
                    yield_strength_pa=beam.yield_strength_pa,
                    resistance_factor=beam.resistance_factor,
                )
                for record in beam_records
            }
        contexts[member.group_id].append(
            _ScreeningContext(member_id=member.member_id, demand=demand, capacities=capacities)
        )

    audits: list[GroupFilterAudit] = []
    critical_members: dict[str, str] = {}
    for group_id in generated.definition.design_groups:
        is_column = int(group_id) <= 8
        audit, critical_member = _group_screening_audit(
            group_id=group_id,
            member_type="column" if is_column else "beam",
            demand_kind="bottom_story_axial_force_n" if is_column else "maximum_moment_nm",
            records=column_records if is_column else beam_records,
            contexts=contexts[group_id],
            published_removed_count=policy.published_removed_counts[group_id],
        )
        audits.append(audit)
        critical_members[group_id] = critical_member
    actual = {item.group_id: item.actual_removed_count for item in audits}
    if len(column_records) != 36 or len(beam_records) != 273:
        raise RuntimeError(
            f"approved domain sizes differ: W14={len(column_records)}, W={len(beam_records)}"
        )
    return NineStoryFilterArtifact(
        validation_status="not independently engineering-validated",
        domain_label=policy.domain_label,
        column_base_domain_size=36,
        beam_base_domain_size=273,
        groups=tuple(audits),
        critical_member_by_group=critical_members,
        actual_removed_counts=actual,
        published_removed_counts=policy.published_removed_counts,
        published_counts_match=actual == policy.published_removed_counts,
        disposition="project_baseline_recorded_publication_diagnostic",
    )


def build_phase11_artifact(
    *,
    frame_path: Path,
    sections_path: Path,
    case_path: Path,
    stage1_policy_path: Path,
    filter_policy_path: Path,
    reference_path: Path,
) -> Phase11Artifact:
    """Build the complete deterministic nine-story topology and Stage 1 evidence."""

    definition = FrameDefinition.from_json(frame_path)
    if definition.frame_id != "9-story":
        raise ValueError("Phase 11 requires the nine-story definition")
    generated = generate_frame(definition)
    repository = SectionRepository.from_json(sections_path)
    case = Stage1Case.from_json(case_path)
    sections = {
        group_id: repository.get(section_name)
        for group_id, section_name in case.sections_by_group.items()
    }
    stage1 = evaluate_stage1_design(
        generated=generated,
        sections_by_group=sections,
        case=case,
        policy=Stage1Policy.from_json(stage1_policy_path),
        reference=PublishedStage1Reference.from_json(reference_path),
    )
    return Phase11Artifact(
        validation_status="verified research software; not engineering-certified",
        transcription=audit_figure8_transcription(generated),
        topology=build_topology_audit(generated, repository),
        modal=run_modal_analysis(generated, sections, mode_count=9),
        preliminary_filter=build_nine_story_filter(
            generated=generated,
            repository=repository,
            policy=NineStoryFilterPolicy.from_json(filter_policy_path),
        ),
        published_case2_stage1=stage1,
        three_story_model_changed=False,
    )


def _parser() -> argparse.ArgumentParser:
    """Create the Phase 11 evidence command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--sections", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--stage1-policy", type=Path, required=True)
    parser.add_argument("--filter-policy", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write canonical Phase 11 evidence and a source-transcribed topology figure."""

    arguments = _parser().parse_args(argv)
    artifact = build_phase11_artifact(
        frame_path=arguments.frame,
        sections_path=arguments.sections,
        case_path=arguments.case,
        stage1_policy_path=arguments.stage1_policy,
        filter_policy_path=arguments.filter_policy,
        reference_path=arguments.reference,
    )
    write_canonical_json(arguments.output, artifact)
    definition = FrameDefinition.from_json(arguments.frame)
    save_topology_figure(generate_frame(definition), arguments.figure)
    print(
        "Phase 11: "
        f"members={artifact.transcription.members}, "
        f"T1={artifact.modal.periods_s[0]:.6f}s, "
        f"Stage1 feasible={artifact.published_case2_stage1.feasible}, "
        f"Wmax={artifact.topology.wmax_cross_check.computed_wmax_n:.3f}N"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
