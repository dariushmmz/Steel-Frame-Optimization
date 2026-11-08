"""Audit the paper's approximate three-story preliminary section-domain filters.

This module performs domain reduction only. It deliberately does not expose the preliminary
calculations as full Stage 1 member checks.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from steel_frame_optimization.config.schemas import StrictConfigModel
from steel_frame_optimization.constraints.lrfd import (
    compact_beam_design_flexural_strength_nm,
    nominal_compression_strength_n,
)
from steel_frame_optimization.models.frame import FrameDefinition
from steel_frame_optimization.sections.records import SectionRecord
from steel_frame_optimization.sections.repository import SectionRepository
from steel_frame_optimization.serialization import write_canonical_json


class BeamFilterPolicy(StrictConfigModel):
    """Freeze source-derived inputs for the compact-beam preliminary filter."""

    dead_load_factor: float = Field(gt=0.0, allow_inf_nan=False)
    live_load_factor: float = Field(gt=0.0, allow_inf_nan=False)
    zero_moment_fraction: float = Field(gt=0.0, lt=0.5, allow_inf_nan=False)
    maximum_moment_coefficient: float = Field(gt=0.0, allow_inf_nan=False)
    resistance_factor: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    yield_strength_pa: float = Field(gt=0.0, allow_inf_nan=False)
    section_modulus: Literal["plastic_zx_compact_assumption"]

    @model_validator(mode="after")
    def verify_moment_coefficient(self) -> BeamFilterPolicy:
        """Require the coefficient implied by symmetric zero points under uniform load."""

        fraction = self.zero_moment_fraction
        derived = 0.125 - 0.5 * fraction + 0.5 * fraction**2
        if not math.isclose(self.maximum_moment_coefficient, derived, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("maximum moment coefficient is inconsistent with zero-moment points")
        return self


class ColumnFilterPolicy(StrictConfigModel):
    """Freeze the delegated preliminary column-screening inputs."""

    dead_load_factor: float = Field(gt=0.0, allow_inf_nan=False)
    live_load_factor: float = Field(gt=0.0, allow_inf_nan=False)
    resistance_factor: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    yield_strength_pa: float = Field(gt=0.0, allow_inf_nan=False)
    elastic_modulus_pa: float = Field(gt=0.0, allow_inf_nan=False)
    buckling_axis: Literal["major_rx"]
    preliminary_effective_length_factor: float = Field(gt=0.0, allow_inf_nan=False)
    effective_length_basis: Literal["per_story_centerline_length"]


class PreliminaryFilterPolicy(StrictConfigModel):
    """Validate the complete immutable policy and published comparison counts."""

    schema_version: Literal["1.0"] = "1.0"
    domain_label: Literal["273-section domain, pending client confirmation"]
    assumption_id: Literal["ASM-028"]
    beam: BeamFilterPolicy
    column: ColumnFilterPolicy
    published_removed_counts: dict[str, int]

    @classmethod
    def from_json(cls, path: Path) -> PreliminaryFilterPolicy:
        """Load and validate one preliminary-filter policy file."""

        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))

    @model_validator(mode="after")
    def validate_reference_counts(self) -> PreliminaryFilterPolicy:
        """Require all and only the five published three-story group counts."""

        expected = {"1": 16, "2": 32, "3": 42, "4": 42, "5": 38}
        if self.published_removed_counts != expected:
            raise ValueError("published removal counts must preserve REF-A exactly")
        return self


class RejectedSection(StrictConfigModel):
    """Preserve one rejected section and the exact screening comparison that rejected it."""

    source_row_id: int = Field(ge=2)
    section_name: str = Field(min_length=1)
    reason: Literal["design_flexural_strength_below_demand", "design_axial_strength_below_demand"]
    demand: float = Field(gt=0.0, allow_inf_nan=False)
    design_capacity: float = Field(gt=0.0, allow_inf_nan=False)
    threshold_section: str = Field(min_length=1)
    source_dataset_id: Literal["aisc-v14.1-w-dev-001"]


class GroupFilterAudit(StrictConfigModel):
    """Record one design group's demand, inclusive threshold, and rejected source rows."""

    group_id: str = Field(min_length=1)
    member_type: Literal["beam", "column"]
    demand_kind: Literal["maximum_moment_nm", "bottom_story_axial_force_n"]
    demand: float = Field(gt=0.0, allow_inf_nan=False)
    temporary_ordering: Literal["design_capacity_then_source_row"]
    persistent_operator_ordering: Literal["weight_then_source_row"]
    threshold_section: str = Field(min_length=1)
    threshold_design_capacity: float = Field(gt=0.0, allow_inf_nan=False)
    actual_removed_count: int = Field(ge=0)
    published_removed_count: int = Field(ge=0)
    matches_published_count: bool
    rejected_sections: tuple[RejectedSection, ...]


class PreliminaryFilterAudit(StrictConfigModel):
    """Persist the five-group filter comparison without accepting a discrepancy."""

    schema_version: Literal["1.0"] = "1.0"
    validation_status: Literal["not engineering-validated"]
    domain_label: Literal["273-section domain, pending client confirmation"]
    assumption_id: Literal["ASM-028"]
    groups: tuple[GroupFilterAudit, ...] = Field(min_length=5, max_length=5)
    all_counts_match: bool
    disposition: Literal["pass", "mandatory_stop_reference_discrepancy"]


def _threshold_and_rejected(
    *,
    records: tuple[SectionRecord, ...],
    demand: float,
    capacity_by_name: dict[str, float],
    reason: Literal["design_flexural_strength_below_demand", "design_axial_strength_below_demand"],
) -> tuple[SectionRecord, float, tuple[RejectedSection, ...]]:
    """Find the inclusive capacity threshold and retain a complete rejection audit."""

    ordered = sorted(records, key=lambda item: (capacity_by_name[item.name], item.source_row_id))
    threshold = next((item for item in ordered if capacity_by_name[item.name] >= demand), None)
    if threshold is None:
        raise ValueError("no section satisfies the preliminary filter demand")
    rejected_records = tuple(item for item in ordered if capacity_by_name[item.name] < demand)
    rejected = tuple(
        RejectedSection(
            source_row_id=item.source_row_id,
            section_name=item.name,
            reason=reason,
            demand=demand,
            design_capacity=capacity_by_name[item.name],
            threshold_section=threshold.name,
            source_dataset_id="aisc-v14.1-w-dev-001",
        )
        for item in rejected_records
    )
    return threshold, capacity_by_name[threshold.name], rejected


def _beam_group_audit(
    *,
    group_id: str,
    span_m: float,
    dead_load_n_per_m: float,
    live_load_n_per_m: float,
    records: tuple[SectionRecord, ...],
    policy: PreliminaryFilterPolicy,
) -> GroupFilterAudit:
    """Build one compact-beam filter audit from the paper's factored uniform load model."""

    beam = policy.beam
    factored_load = (
        beam.dead_load_factor * dead_load_n_per_m + beam.live_load_factor * live_load_n_per_m
    )
    demand = beam.maximum_moment_coefficient * factored_load * span_m**2
    capacities = {
        record.name: compact_beam_design_flexural_strength_nm(
            plastic_section_modulus_m3=record.zx_m3,
            yield_strength_pa=beam.yield_strength_pa,
            resistance_factor=beam.resistance_factor,
        )
        for record in records
    }
    threshold, threshold_capacity, rejected = _threshold_and_rejected(
        records=records,
        demand=demand,
        capacity_by_name=capacities,
        reason="design_flexural_strength_below_demand",
    )
    published = policy.published_removed_counts[group_id]
    return GroupFilterAudit(
        group_id=group_id,
        member_type="beam",
        demand_kind="maximum_moment_nm",
        demand=demand,
        temporary_ordering="design_capacity_then_source_row",
        persistent_operator_ordering="weight_then_source_row",
        threshold_section=threshold.name,
        threshold_design_capacity=threshold_capacity,
        actual_removed_count=len(rejected),
        published_removed_count=published,
        matches_published_count=len(rejected) == published,
        rejected_sections=rejected,
    )


def _column_group_audit(
    *,
    group_id: str,
    reaction_multiplier: float,
    story_height_m: float,
    bay_width_m: float,
    frame: FrameDefinition,
    records: tuple[SectionRecord, ...],
    policy: PreliminaryFilterPolicy,
) -> GroupFilterAudit:
    """Build one bottom-story axial-only column filter audit with accumulated floor reactions."""

    column = policy.column
    demand = (
        reaction_multiplier
        * bay_width_m
        * sum(
            column.dead_load_factor * floor.dead_line_load_n_per_m
            + column.live_load_factor * floor.live_line_load_n_per_m
            for floor in frame.floors
        )
    )
    capacities = {
        record.name: column.resistance_factor
        * nominal_compression_strength_n(
            area_m2=record.area_m2,
            effective_length_factor=column.preliminary_effective_length_factor,
            length_m=story_height_m,
            radius_of_gyration_m=record.rx_m,
            yield_strength_pa=column.yield_strength_pa,
            elastic_modulus_pa=column.elastic_modulus_pa,
        )
        for record in records
    }
    threshold, threshold_capacity, rejected = _threshold_and_rejected(
        records=records,
        demand=demand,
        capacity_by_name=capacities,
        reason="design_axial_strength_below_demand",
    )
    published = policy.published_removed_counts[group_id]
    return GroupFilterAudit(
        group_id=group_id,
        member_type="column",
        demand_kind="bottom_story_axial_force_n",
        demand=demand,
        temporary_ordering="design_capacity_then_source_row",
        persistent_operator_ordering="weight_then_source_row",
        threshold_section=threshold.name,
        threshold_design_capacity=threshold_capacity,
        actual_removed_count=len(rejected),
        published_removed_count=published,
        matches_published_count=len(rejected) == published,
        rejected_sections=rejected,
    )


def build_preliminary_filter_audit(
    *, frame: FrameDefinition, repository: SectionRepository, policy: PreliminaryFilterPolicy
) -> PreliminaryFilterAudit:
    """Build the fixed five-group research comparison and classify any count discrepancy."""

    if frame.domain_label != policy.domain_label:
        raise ValueError("frame and filter policy domain labels must match")
    records = repository.family("all_w")
    groups: list[GroupFilterAudit] = [
        _column_group_audit(
            group_id="1",
            reaction_multiplier=0.5,
            story_height_m=frame.story_heights_m[0],
            bay_width_m=frame.bay_widths_m[0],
            frame=frame,
            records=records,
            policy=policy,
        ),
        _column_group_audit(
            group_id="2",
            reaction_multiplier=1.0,
            story_height_m=frame.story_heights_m[0],
            bay_width_m=frame.bay_widths_m[0],
            frame=frame,
            records=records,
            policy=policy,
        ),
    ]
    for floor in frame.floors:
        groups.append(
            _beam_group_audit(
                group_id=str(floor.level + 2),
                span_m=frame.bay_widths_m[0],
                dead_load_n_per_m=floor.dead_line_load_n_per_m,
                live_load_n_per_m=floor.live_line_load_n_per_m,
                records=records,
                policy=policy,
            )
        )
    all_match = all(group.matches_published_count for group in groups)
    return PreliminaryFilterAudit(
        validation_status="not engineering-validated",
        domain_label=policy.domain_label,
        assumption_id=policy.assumption_id,
        groups=tuple(groups),
        all_counts_match=all_match,
        disposition="pass" if all_match else "mandatory_stop_reference_discrepancy",
    )


def _parser() -> argparse.ArgumentParser:
    """Create the deterministic Phase 5 preliminary-filter command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--sections", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write the filter audit and return nonzero when published counts disagree."""

    arguments = _parser().parse_args(argv)
    audit = build_preliminary_filter_audit(
        frame=FrameDefinition.from_json(arguments.frame),
        repository=SectionRepository.from_json(arguments.sections),
        policy=PreliminaryFilterPolicy.from_json(arguments.policy),
    )
    write_canonical_json(arguments.output, audit)
    summary = ", ".join(
        f"G{group.group_id}={group.actual_removed_count}/{group.published_removed_count}"
        for group in audit.groups
    )
    print(f"preliminary filter removals: {summary}")
    return 0 if audit.all_counts_match else 2


if __name__ == "__main__":
    raise SystemExit(main())
