"""Define reversible group domains and separate operator order from capacity filtering."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from steel_frame_optimization.config.schemas import StrictConfigModel
from steel_frame_optimization.sections.records import SectionRecord
from steel_frame_optimization.sections.repository import SectionRepository


class GroupDomainSpec(StrictConfigModel):
    """Declare one design group's base family and published lower-bound removal count."""

    group_id: str = Field(min_length=1)
    member_type: Literal["beam", "column"]
    base_family: Literal["all_w", "w14"]
    published_removed_count: int = Field(ge=0)


class FrameDomainSpec(StrictConfigModel):
    """Declare every ordered group domain for one paper frame."""

    frame_id: Literal["3-story", "9-story"]
    groups: tuple[GroupDomainSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_group_ids(self) -> FrameDomainSpec:
        """Require unique group IDs without inferring the disputed paper numbering."""

        group_ids = [group.group_id for group in self.groups]
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("frame group IDs must be unique")
        return self


class DomainCatalog(StrictConfigModel):
    """Persist both frame base domains under the provisional DEV-001 identity."""

    schema_version: Literal["1.0"] = "1.0"
    domain_label: Literal["273-section domain, pending client confirmation"]
    deviation_id: Literal["DEV-001"] = "DEV-001"
    operator_order: Literal["ascending_weight_then_source_row"]
    column_filter_order: Literal["ascending_axial_capacity_then_source_row"]
    frames: tuple[FrameDomainSpec, ...] = Field(min_length=2, max_length=2)

    @classmethod
    def from_json(cls, path: Path) -> DomainCatalog:
        """Load and validate the configured frame/group domain catalog."""

        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class CandidateDomain:
    """Represent one reversible integer-allele domain in the approved operator order."""

    group_id: str
    records: tuple[SectionRecord, ...]

    def __post_init__(self) -> None:
        """Reject empty, duplicate, or incorrectly ordered domains."""

        if not self.records:
            raise ValueError("candidate domain cannot be empty")
        names = [record.name for record in self.records]
        if len(set(names)) != len(names):
            raise ValueError("candidate domain section names must be unique")
        expected = sorted(
            self.records, key=lambda record: (record.weight_n_per_m, record.source_row_id)
        )
        if list(self.records) != expected:
            raise ValueError("candidate domain must use ascending weight/source-row operator order")

    @classmethod
    def from_family(
        cls, group_id: str, family: str, repository: SectionRepository
    ) -> CandidateDomain:
        """Build a group domain from an explicitly configured base family."""

        return cls(group_id=group_id, records=repository.family(family))

    def encode(self, section_name: str) -> int:
        """Map a canonical section name to its reversible zero-based allele."""

        canonical_name = section_name.upper()
        for index, record in enumerate(self.records):
            if record.name == canonical_name:
                return index
        raise ValueError(f"section {section_name} is not legal for group {self.group_id}")

    def decode(self, allele: int) -> SectionRecord:
        """Map one legal zero-based allele back to its full section record."""

        if allele < 0 or allele >= len(self.records):
            raise ValueError(f"allele {allele} is outside group {self.group_id}")
        return self.records[allele]


@dataclass(frozen=True, slots=True)
class LowerBoundSelection:
    """Audit the inclusive capacity threshold selected by the paper's column rule."""

    demand_n: float
    lower_bound_source_row_id: int
    lower_bound_name: str
    lower_bound_capacity_n: float
    removed_count: int


def filter_column_domain_by_capacity(
    domain: CandidateDomain,
    *,
    axial_capacity_n_by_source_row: Mapping[int, float],
    demand_n: float,
) -> tuple[CandidateDomain, LowerBoundSelection]:
    """Find the first adequate capacity, then preserve operator order among survivors."""

    if demand_n < 0.0:
        raise ValueError("axial compression demand must be nonnegative")
    missing = {
        record.source_row_id
        for record in domain.records
        if record.source_row_id not in axial_capacity_n_by_source_row
    }
    if missing:
        raise ValueError(f"missing axial capacities for source rows: {sorted(missing)}")
    capacity_order = sorted(
        domain.records,
        key=lambda record: (
            axial_capacity_n_by_source_row[record.source_row_id],
            record.source_row_id,
        ),
    )
    lower_bound = next(
        (
            record
            for record in capacity_order
            if axial_capacity_n_by_source_row[record.source_row_id] >= demand_n
        ),
        None,
    )
    if lower_bound is None:
        raise ValueError("no candidate section satisfies the axial demand")
    threshold = axial_capacity_n_by_source_row[lower_bound.source_row_id]
    survivors = tuple(
        record
        for record in domain.records
        if axial_capacity_n_by_source_row[record.source_row_id] >= threshold
    )
    filtered = CandidateDomain(group_id=domain.group_id, records=survivors)
    selection = LowerBoundSelection(
        demand_n=demand_n,
        lower_bound_source_row_id=lower_bound.source_row_id,
        lower_bound_name=lower_bound.name,
        lower_bound_capacity_n=threshold,
        removed_count=len(domain.records) - len(filtered.records),
    )
    return filtered, selection


def maximum_domain_weight_n(
    *,
    member_length_m_by_group: Mapping[str, float],
    domains_by_group: Mapping[str, CandidateDomain],
) -> float:
    """Compute Wmax by assigning the heaviest allowed section to every group length."""

    if set(member_length_m_by_group) != set(domains_by_group):
        raise ValueError("member-length and candidate-domain group IDs must match exactly")
    total_weight_n = 0.0
    for group_id, total_length_m in member_length_m_by_group.items():
        if total_length_m <= 0.0:
            raise ValueError(f"group {group_id} total member length must be positive")
        heaviest = max(domains_by_group[group_id].records, key=lambda record: record.weight_n_per_m)
        total_weight_n += total_length_m * heaviest.weight_n_per_m
    return total_weight_n
