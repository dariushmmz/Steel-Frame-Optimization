"""Layer 1 and layer 2 tests for the provisional Phase 2 W-section domain."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from steel_frame_optimization.config.schemas import CampaignAggregateResult
from steel_frame_optimization.sections.domains import (
    CandidateDomain,
    DomainCatalog,
    filter_column_domain_by_capacity,
    maximum_domain_weight_n,
)
from steel_frame_optimization.sections.quality import build_data_quality_report
from steel_frame_optimization.sections.records import SectionDataset
from steel_frame_optimization.sections.repository import SectionRepository
from steel_frame_optimization.serialization import canonical_json_dumps

pytestmark = pytest.mark.unit


@pytest.fixture
def section_repository(project_root: Path) -> SectionRepository:
    """Load the committed canonical DEV-001 dataset for tests."""

    path = project_root / "data" / "processed" / "aisc_v14_1_w_shapes.json"
    return SectionRepository.from_json(path)


def test_provisional_dataset_identity_and_counts(section_repository: SectionRepository) -> None:
    """Require 273 real W rows, 36 W14 rows, source identity, and explicit DEV-001 labeling."""

    dataset = section_repository.dataset
    assert dataset.actual_section_count == len(dataset.records) == 273
    assert dataset.published_section_count == 274
    assert dataset.w14_section_count == 36
    assert dataset.deviation_id == "DEV-001"
    assert dataset.domain_label == "273-section domain, pending client confirmation"
    assert dataset.source.sha256 == (
        "492cb5dfc59d609bd02da84f069e794244ba8b0e8e80bd481e73eff327658584"
    )


def test_dataset_round_trip_and_invalid_record_rejection(
    section_repository: SectionRepository,
) -> None:
    """Round-trip every field and reject missing or physically invalid required properties."""

    content = section_repository.dataset.model_dump(mode="json")
    assert SectionDataset.model_validate(content) == section_repository.dataset
    invalid = dict(content)
    invalid_records = [dict(record) for record in content["records"]]
    invalid_records[0]["area_m2"] = -1.0
    invalid["records"] = invalid_records
    with pytest.raises(ValidationError, match="area_m2"):
        SectionDataset.model_validate(invalid)
    del invalid_records[0]["area_m2"]
    with pytest.raises(ValidationError, match="area_m2"):
        SectionDataset.model_validate(invalid)


@pytest.mark.analytical
def test_source_weight_conversion_is_exactly_reproducible(
    section_repository: SectionRepository,
) -> None:
    """Verify every stored mass and weight from its preserved lb/ft source value."""

    pound_mass_to_kg = 0.45359237
    pound_force_to_n = 4.4482216152605
    foot_to_m = 0.3048
    for record in section_repository.dataset.records:
        assert record.mass_kg_per_m == pytest.approx(
            record.source_weight_lb_per_ft * pound_mass_to_kg / foot_to_m,
            rel=1e-15,
        )
        assert record.weight_n_per_m == pytest.approx(
            record.source_weight_lb_per_ft * pound_force_to_n / foot_to_m,
            rel=1e-15,
        )


def test_source_and_operator_orders_are_distinct_and_stable(
    section_repository: SectionRepository,
) -> None:
    """Keep workbook provenance order separate from weight-based genetic-operator order."""

    source = section_repository.source_order()
    operator = section_repository.operator_order()
    assert [record.source_row_id for record in source] == sorted(
        record.source_row_id for record in source
    )
    assert [(record.weight_n_per_m, record.source_row_id) for record in operator] == sorted(
        (record.weight_n_per_m, record.source_row_id) for record in source
    )
    assert [record.name for record in source] != [record.name for record in operator]


@pytest.mark.parametrize("family, expected_count", [("all_w", 273), ("w14", 36)])
def test_integer_domain_round_trip(
    section_repository: SectionRepository, family: str, expected_count: int
) -> None:
    """Require every legal integer allele to decode and encode without loss."""

    domain = CandidateDomain.from_family("test-group", family, section_repository)
    assert len(domain.records) == expected_count
    for allele, record in enumerate(domain.records):
        assert domain.decode(allele) == record
        assert domain.encode(record.name.lower()) == allele


def test_integer_domain_rejects_invalid_values(section_repository: SectionRepository) -> None:
    """Reject illegal alleles and section names instead of silently clipping them."""

    domain = CandidateDomain.from_family("test-group", "w14", section_repository)
    with pytest.raises(ValueError, match="outside"):
        domain.decode(-1)
    with pytest.raises(ValueError, match="outside"):
        domain.decode(len(domain.records))
    with pytest.raises(ValueError, match="not legal"):
        domain.encode("W0X0")


def test_column_lower_bound_uses_capacity_then_restores_operator_order(
    section_repository: SectionRepository,
) -> None:
    """Apply the client-image capacity threshold without redefining GA neighborhood order."""

    base_records = section_repository.family("w14")[:4]
    domain = CandidateDomain(group_id="column", records=base_records)
    capacities = {
        base_records[0].source_row_id: 400.0,
        base_records[1].source_row_id: 100.0,
        base_records[2].source_row_id: 300.0,
        base_records[3].source_row_id: 200.0,
    }
    filtered, selection = filter_column_domain_by_capacity(
        domain, axial_capacity_n_by_source_row=capacities, demand_n=250.0
    )
    assert selection.lower_bound_capacity_n == 300.0
    assert selection.lower_bound_name == base_records[2].name
    assert selection.removed_count == 2
    assert filtered.records == (base_records[0], base_records[2])


def test_maximum_domain_weight_uses_heaviest_legal_section(
    section_repository: SectionRepository,
) -> None:
    """Compute Wmax from group lengths and domain maxima without a fitted paper constant."""

    all_w = CandidateDomain.from_family("all", "all_w", section_repository)
    w14 = CandidateDomain.from_family("w14", "w14", section_repository)
    actual = maximum_domain_weight_n(
        member_length_m_by_group={"all": 2.0, "w14": 3.0},
        domains_by_group={"all": all_w, "w14": w14},
    )
    expected = 2.0 * all_w.records[-1].weight_n_per_m + 3.0 * w14.records[-1].weight_n_per_m
    assert actual == expected


@pytest.mark.analytical
def test_all_section_properties_pass_independent_dimensional_checks(
    section_repository: SectionRepository,
) -> None:
    """Check all rows against weight/area and radius-of-gyration identities at layer 2."""

    report = build_data_quality_report(section_repository.dataset)
    assert report.section_count == 273
    assert report.w14_count == 36
    assert report.unique_name_count == 273
    assert report.unique_source_row_count == 273
    assert report.all_checks_pass


@pytest.mark.analytical
def test_aisc_360_16_user_note_compactness_spot_checks(
    section_repository: SectionRepository,
) -> None:
    """Reproduce three AISC 360-16 F2 user-note classifications at Fy=345 MPa."""

    elastic_modulus_mpa = 200_000.0
    yield_strength_mpa = 345.0
    compact_flange_limit = 0.38 * (elastic_modulus_mpa / yield_strength_mpa) ** 0.5
    assert section_repository.get("W14X99").flange_slenderness > compact_flange_limit
    assert section_repository.get("W14X90").flange_slenderness > compact_flange_limit
    assert section_repository.get("W14X82").flange_slenderness <= compact_flange_limit


def test_domain_catalog_matches_paper_grouping_contract(project_root: Path) -> None:
    """Validate five 3-story and thirteen 9-story configured design groups."""

    catalog = DomainCatalog.from_json(
        project_root / "configs" / "frames" / "phase2_candidate_domains.json"
    )
    frames = {frame.frame_id: frame for frame in catalog.frames}
    assert len(frames["3-story"].groups) == 5
    assert len(frames["9-story"].groups) == 13
    assert all(group.base_family == "w14" for group in frames["9-story"].groups[:8])
    assert all(group.base_family == "all_w" for group in frames["9-story"].groups[8:])


def test_campaign_aggregate_contract_requires_feasibility_breakdown() -> None:
    """Require the owner-directed all-run and feasible-only 30-run output fields."""

    values = {
        "frame_id": "3-story",
        "domain_label": "273-section domain, pending client confirmation",
        "best_normalized_objective": 0.1,
        "champion_weight_kn": 100.0,
        "champion_unweighted_uniformity": 0.01,
        "champion_weight_contribution": 0.08,
        "champion_uniformity_contribution": 0.02,
        "all_runs_mean_final_weight_kn": 110.0,
        "all_runs_worst_final_weight_kn": 120.0,
        "feasible_runs_mean_final_weight_kn": 105.0,
        "feasible_runs_worst_final_weight_kn": 115.0,
        "feasible_final_design_count": 25,
        "infeasible_final_design_count": 5,
    }
    result = CampaignAggregateResult.model_validate(values)
    assert result.feasible_final_design_count == 25
    values["infeasible_final_design_count"] = 4
    with pytest.raises(ValueError, match="sum to 30"):
        CampaignAggregateResult.model_validate(values)


def test_dataset_is_valid_json_and_contains_no_header_row(project_root: Path) -> None:
    """Confirm the canonical artifact has 273 data objects and never counts a header as a shape."""

    path = project_root / "data" / "processed" / "aisc_v14_1_w_shapes.json"
    content = json.loads(path.read_text(encoding="utf-8"))
    names = [record["name"] for record in content["records"]]
    assert len(names) == 273
    assert "EDI_STD_NOMENCLATURE" not in names
    assert path.read_text(encoding="utf-8") == canonical_json_dumps(content)
