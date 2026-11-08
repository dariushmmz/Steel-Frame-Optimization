"""Expose validated W-section records, repositories, and candidate-domain helpers."""

from steel_frame_optimization.sections.domains import (
    CandidateDomain,
    DomainCatalog,
    LowerBoundSelection,
    filter_column_domain_by_capacity,
    maximum_domain_weight_n,
)
from steel_frame_optimization.sections.records import SectionDataset, SectionRecord
from steel_frame_optimization.sections.repository import SectionRepository

__all__ = [
    "CandidateDomain",
    "DomainCatalog",
    "LowerBoundSelection",
    "SectionDataset",
    "SectionRecord",
    "SectionRepository",
    "filter_column_domain_by_capacity",
    "maximum_domain_weight_n",
]
"""Provide section ingestion, validation, querying, domains, and preliminary filtering."""
