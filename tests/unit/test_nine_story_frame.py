"""Unit and source-transcription checks for the nine-story frame definition."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from steel_frame_optimization.models.frame import (
    FrameDefinition,
    build_topology_audit,
    generate_frame,
)
from steel_frame_optimization.sections.repository import SectionRepository
from steel_frame_optimization.validation.phase11 import audit_figure8_transcription

pytestmark = pytest.mark.unit


@pytest.fixture
def definition(project_root: Path) -> FrameDefinition:
    """Load the source-transcribed nine-story frame."""

    return FrameDefinition.from_json(project_root / "configs/frames/nine_story_frame.json")


@pytest.fixture
def repository(project_root: Path) -> SectionRepository:
    """Load the approved project W-section repository."""

    return SectionRepository.from_json(project_root / "data/processed/aisc_v14_1_w_shapes.json")


def test_figure8_topology_grouping_geometry_and_supports(definition: FrameDefinition) -> None:
    """Require every aggregate and member-group invariant independently read from Figure 8."""

    generated = generate_frame(definition)
    audit = audit_figure8_transcription(generated)
    assert (audit.nodes, audit.members, audit.columns, audit.beams, audit.fixed_supports) == (
        60,
        99,
        54,
        45,
        6,
    )
    assert audit.width_m == pytest.approx(45.72)
    assert audit.height_m == pytest.approx(37.1856)
    assert audit.group_member_counts == audit.expected_group_member_counts
    assert audit.group_member_counts == {
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
    assert all(
        support.restrained_dofs == ("ux", "uy", "rz") for support in generated.topology.supports
    )


def test_mixed_figure8_group_six_is_transcribed_exactly(definition: FrameDefinition) -> None:
    """Protect the unusual group 6 use in middle exterior and upper interior columns."""

    generated = generate_frame(definition)
    group_six = {
        member.member_id for member in generated.topology.members if member.group_id == "6"
    }
    expected = {
        *(f"C-S{story}-C{column}" for story in (6, 7) for column in (0, 5)),
        *(f"C-S{story}-C{column}" for story in (8, 9) for column in range(1, 5)),
    }
    assert group_six == expected


@pytest.mark.analytical
def test_nine_story_load_mass_lengths_and_wmax_are_reproducible(
    definition: FrameDefinition, repository: SectionRepository
) -> None:
    """Recover configured totals and the project-specific Wmax without paper fitting."""

    topology = build_topology_audit(generate_frame(definition), repository)
    assert topology.total_dead_load_n == pytest.approx(5_577_840.0)
    assert topology.total_live_load_n == pytest.approx(4_000_500.0)
    assert topology.total_seismic_weight_n == pytest.approx(44_822_000.0)
    assert topology.total_seismic_mass_kg == pytest.approx(44_822_000.0 / 9.80665)
    lengths = {item.group_id: item.total_length_m for item in topology.group_lengths}
    assert lengths == pytest.approx(
        {
            "1": 26.8224,
            "2": 21.9456,
            "3": 31.6992,
            "4": 15.8496,
            "5": 31.6992,
            "6": 47.5488,
            "7": 31.6992,
            "8": 15.8496,
            "9": 91.44,
            "10": 182.88,
            "11": 45.72,
            "12": 45.72,
            "13": 45.72,
        }
    )
    wmax = topology.wmax_cross_check
    assert wmax.domain_basis == "unfiltered_w14_columns_and_all_w_beams"
    assert wmax.computed_wmax_n == pytest.approx(6_760_674.104169824)
    assert wmax.signed_relative_difference == pytest.approx(-0.05917110883305374)


def test_nine_story_definition_rejects_wrong_group_or_geometry(project_root: Path) -> None:
    """Reject a missing group or a geometry inconsistent with the source frame."""

    path = project_root / "configs/frames/nine_story_frame.json"
    content = json.loads(path.read_text(encoding="utf-8"))
    content["design_groups"] = content["design_groups"][:-1]
    with pytest.raises(ValidationError, match="exactly 1 through 13"):
        FrameDefinition.model_validate(content)
    content = json.loads(path.read_text(encoding="utf-8"))
    content["bay_widths_m"] = content["bay_widths_m"][:-1]
    with pytest.raises(ValidationError, match="9 stories and 5 bays"):
        FrameDefinition.model_validate(content)


def test_nine_story_generation_is_deterministic(definition: FrameDefinition) -> None:
    """Require stable topology, group counts, and solver tags across fresh generation."""

    first = generate_frame(definition)
    second = generate_frame(definition)
    assert first == second
    assert Counter(member.member_type for member in first.topology.members) == {
        "column": 54,
        "beam": 45,
    }
    assert sorted(first.node_tags.values()) == list(range(1, 61))
    assert sorted(first.member_tags.values()) == list(range(1, 100))
