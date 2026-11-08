"""Compute deterministic Phase 2 section-data consistency evidence."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import Field

from steel_frame_optimization.config.schemas import StrictConfigModel
from steel_frame_optimization.sections.records import SectionDataset, SectionRecord

STANDARD_GRAVITY_M_PER_S2: Final = 9.80665
STEEL_DENSITY_KG_PER_M3: Final = 7850.0
RELATIVE_CONSISTENCY_TOLERANCE: Final = 0.02


class ResidualMaximum(StrictConfigModel):
    """Record the section and value governing one dimensionless consistency residual."""

    section_name: str = Field(min_length=1)
    relative_residual: float = Field(ge=0.0, allow_inf_nan=False)


class SectionDataQualityReport(StrictConfigModel):
    """Persist exact counts and independently recomputed geometry/weight residuals."""

    schema_version: Literal["1.0"] = "1.0"
    validation_layer: Literal["layer 2 numerical verification"]
    domain_label: Literal["273-section domain, pending client confirmation"]
    section_count: int
    w14_count: int
    unique_name_count: int
    unique_source_row_count: int
    tolerance: float
    area_weight_maximum: ResidualMaximum
    ix_radius_maximum: ResidualMaximum
    iy_radius_maximum: ResidualMaximum
    all_checks_pass: bool


def _relative_residual(actual: float, expected: float) -> float:
    """Return absolute relative residual against a strictly positive expected value."""

    return abs(actual - expected) / expected


def _maximum(records: tuple[SectionRecord, ...], residual: str) -> ResidualMaximum:
    """Return the record with the largest named derived residual."""

    values: list[tuple[SectionRecord, float]] = []
    for record in records:
        if residual == "area_weight":
            expected = record.area_m2 * STEEL_DENSITY_KG_PER_M3 * STANDARD_GRAVITY_M_PER_S2
            value = _relative_residual(record.weight_n_per_m, expected)
        elif residual == "ix_radius":
            value = _relative_residual(record.rx_m**2, record.ix_m4 / record.area_m2)
        elif residual == "iy_radius":
            value = _relative_residual(record.ry_m**2, record.iy_m4 / record.area_m2)
        else:
            raise ValueError(f"unknown residual: {residual}")
        values.append((record, value))
    governing_record, governing_value = max(values, key=lambda item: item[1])
    return ResidualMaximum(
        section_name=governing_record.name,
        relative_residual=governing_value,
    )


def build_data_quality_report(dataset: SectionDataset) -> SectionDataQualityReport:
    """Recompute every Phase 2 count and dimensional relationship from persisted SI values."""

    area_weight = _maximum(dataset.records, "area_weight")
    ix_radius = _maximum(dataset.records, "ix_radius")
    iy_radius = _maximum(dataset.records, "iy_radius")
    residuals = (area_weight, ix_radius, iy_radius)
    return SectionDataQualityReport(
        validation_layer="layer 2 numerical verification",
        domain_label=dataset.domain_label,
        section_count=len(dataset.records),
        w14_count=sum(record.name.startswith("W14X") for record in dataset.records),
        unique_name_count=len({record.name for record in dataset.records}),
        unique_source_row_count=len({record.source_row_id for record in dataset.records}),
        tolerance=RELATIVE_CONSISTENCY_TOLERANCE,
        area_weight_maximum=area_weight,
        ix_radius_maximum=ix_radius,
        iy_radius_maximum=iy_radius,
        all_checks_pass=all(
            maximum.relative_residual <= RELATIVE_CONSISTENCY_TOLERANCE for maximum in residuals
        ),
    )
