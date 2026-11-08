"""Define immutable, unit-explicit section-data records and provenance models.

The persisted values are converted once from the AISC workbook's U.S. customary columns to the
project's canonical SI boundary. Field names include their units so downstream calculations cannot
silently mix source and canonical quantities.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from steel_frame_optimization.config.schemas import StrictConfigModel


class SectionSource(StrictConfigModel):
    """Record the exact local source identity without redistributing the raw workbook."""

    filename: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sheet: Literal["Database v14.1"] = "Database v14.1"
    retrieval_url: str = Field(min_length=1)
    retrieved_at_utc: str = Field(min_length=1)
    authority_limitation: str = Field(min_length=1)


class SectionRecord(StrictConfigModel):
    """Store one W shape in canonical SI units with its immutable workbook row identity."""

    source_row_id: int = Field(ge=2)
    name: str = Field(pattern=r"^W\d+(?:\.\d+)?X\d+(?:\.\d+)?$")
    source_weight_lb_per_ft: float = Field(gt=0.0, allow_inf_nan=False)
    mass_kg_per_m: float = Field(gt=0.0, allow_inf_nan=False)
    weight_n_per_m: float = Field(gt=0.0, allow_inf_nan=False)
    area_m2: float = Field(gt=0.0, allow_inf_nan=False)
    depth_m: float = Field(gt=0.0, allow_inf_nan=False)
    flange_width_m: float = Field(gt=0.0, allow_inf_nan=False)
    web_thickness_m: float = Field(gt=0.0, allow_inf_nan=False)
    flange_thickness_m: float = Field(gt=0.0, allow_inf_nan=False)
    ix_m4: float = Field(gt=0.0, allow_inf_nan=False)
    zx_m3: float = Field(gt=0.0, allow_inf_nan=False)
    sx_m3: float = Field(gt=0.0, allow_inf_nan=False)
    rx_m: float = Field(gt=0.0, allow_inf_nan=False)
    iy_m4: float = Field(gt=0.0, allow_inf_nan=False)
    zy_m3: float = Field(gt=0.0, allow_inf_nan=False)
    sy_m3: float = Field(gt=0.0, allow_inf_nan=False)
    ry_m: float = Field(gt=0.0, allow_inf_nan=False)
    torsional_j_m4: float = Field(gt=0.0, allow_inf_nan=False)
    warping_cw_m6: float = Field(gt=0.0, allow_inf_nan=False)
    rts_m: float = Field(gt=0.0, allow_inf_nan=False)
    flange_centroid_distance_ho_m: float = Field(gt=0.0, allow_inf_nan=False)
    flange_slenderness: float = Field(gt=0.0, allow_inf_nan=False)
    web_slenderness: float = Field(gt=0.0, allow_inf_nan=False)


class SectionDataset(StrictConfigModel):
    """Persist the provisional W-shape domain and its explicit published-count deviation."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: Literal["aisc-v14.1-w-dev-001"] = "aisc-v14.1-w-dev-001"
    domain_label: Literal["273-section domain, pending client confirmation"]
    deviation_id: Literal["DEV-001"] = "DEV-001"
    published_section_count: Literal[274] = 274
    actual_section_count: Literal[273] = 273
    w14_section_count: Literal[36] = 36
    source: SectionSource
    records: tuple[SectionRecord, ...] = Field(min_length=273, max_length=273)

    @model_validator(mode="after")
    def validate_dataset_identity(self) -> SectionDataset:
        """Require exact counts and unique source/name identities for the provisional domain."""

        names = [record.name for record in self.records]
        source_rows = [record.source_row_id for record in self.records]
        if len(set(names)) != self.actual_section_count:
            raise ValueError("section names must be unique")
        if len(set(source_rows)) != self.actual_section_count:
            raise ValueError("source row IDs must be unique")
        if sum(name.startswith("W14X") for name in names) != self.w14_section_count:
            raise ValueError("W14 subset must contain exactly 36 sections")
        if source_rows != sorted(source_rows):
            raise ValueError("dataset records must preserve ascending source-row identity")
        return self
