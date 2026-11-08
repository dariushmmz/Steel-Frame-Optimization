"""Layer 1 and layer 2 checks for the configured three-story frame definition."""

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
    save_topology_figure,
)
from steel_frame_optimization.sections.repository import SectionRepository

pytestmark = pytest.mark.unit


@pytest.fixture
def definition(project_root: Path) -> FrameDefinition:
    """Load the committed three-story frame definition."""

    return FrameDefinition.from_json(project_root / "configs" / "frames" / "three_story_frame.json")


@pytest.fixture
def repository(project_root: Path) -> SectionRepository:
    """Load the committed provisional W-section domain."""

    return SectionRepository.from_json(
        project_root / "data" / "processed" / "aisc_v14_1_w_shapes.json"
    )


def test_topology_counts_coordinates_groups_and_supports(definition: FrameDefinition) -> None:
    """Require the paper geometry, connectivity, group map, and fixed-base boundary conditions."""

    generated = generate_frame(definition)
    assert len(generated.topology.nodes) == 20
    assert len(generated.topology.members) == 27
    assert len(generated.topology.supports) == 5
    assert Counter(member.member_type for member in generated.topology.members) == {
        "column": 15,
        "beam": 12,
    }
    assert Counter(member.group_id for member in generated.topology.members) == {
        "1": 6,
        "2": 9,
        "3": 4,
        "4": 4,
        "5": 4,
    }
    assert max(node.x_m for node in generated.topology.nodes) == pytest.approx(36.576)
    assert max(node.y_m for node in generated.topology.nodes) == pytest.approx(11.8872)
    assert all(
        support.restrained_dofs == ("ux", "uy", "rz") for support in generated.topology.supports
    )


def test_tags_and_generation_are_deterministic(definition: FrameDefinition) -> None:
    """Generate twice and require stable semantic topology and numerical tag maps."""

    first = generate_frame(definition)
    second = generate_frame(definition)
    assert first == second
    assert sorted(first.node_tags.values()) == list(range(1, 21))
    assert sorted(first.member_tags.values()) == list(range(1, 28))
    assert first.node_tags["N-L0-C0"] == 1
    assert first.member_tags["C-S1-C0"] == 1
    assert first.member_tags["B-L3-B3"] == 27


@pytest.mark.analytical
def test_mass_allocation_reproduces_each_published_story_weight(
    definition: FrameDefinition,
) -> None:
    """Check tributary-width mass fractions and independently recover each story weight."""

    generated = generate_frame(definition)
    expected_fractions = [0.125, 0.25, 0.25, 0.25, 0.125]
    for floor in definition.floors:
        allocations = [mass for mass in generated.node_masses if mass.level == floor.level]
        assert [mass.tributary_fraction for mass in allocations] == pytest.approx(
            expected_fractions
        )
        recovered_weight = (
            sum(mass.mass_kg for mass in allocations) * definition.gravity_acceleration_m_per_s2
        )
        assert recovered_weight == pytest.approx(floor.seismic_weight_n, rel=1e-15)
        assert allocations[0].mass_kg == pytest.approx(allocations[-1].mass_kg)
        assert allocations[1].mass_kg == pytest.approx(allocations[-2].mass_kg)


@pytest.mark.analytical
def test_line_load_totals_are_recovered_without_candidate_self_weight(
    definition: FrameDefinition,
    repository: SectionRepository,
) -> None:
    """Integrate configured floor line loads and keep candidate member weight out of gravity."""

    audit = build_topology_audit(generate_frame(definition), repository)
    assert audit.total_dead_load_n == pytest.approx(1_499_616.0)
    assert audit.total_live_load_n == pytest.approx(1_005_840.0)
    assert audit.total_seismic_weight_n == pytest.approx(14_447_000.0)
    assert audit.total_seismic_mass_kg == pytest.approx(
        14_447_000.0 / definition.gravity_acceleration_m_per_s2
    )
    assert audit.policy.candidate_member_self_weight_added_to_gravity is False
    assert audit.policy.candidate_member_mass_added_to_modal_mass is False


@pytest.mark.analytical
def test_member_lengths_and_wmax_diagnostic(
    definition: FrameDefinition,
    repository: SectionRepository,
) -> None:
    """Independently check group lengths and preserve the observed Wmax mismatch."""

    audit = build_topology_audit(generate_frame(definition), repository)
    lengths = {group.group_id: group.total_length_m for group in audit.group_lengths}
    assert lengths == pytest.approx(
        {"1": 23.7744, "2": 35.6616, "3": 36.576, "4": 36.576, "5": 36.576}
    )
    check = audit.wmax_cross_check
    assert check.heaviest_section == "W14X730"
    assert check.computed_wmax_n == pytest.approx(1_802_196.9873657476)
    assert check.signed_relative_difference == pytest.approx(-0.0870375561715555)
    assert check.disposition == "mismatch_inspect_domain_upper_bound_and_grouping"


def test_invalid_floor_sequence_and_group_map_are_rejected(project_root: Path) -> None:
    """Reject incomplete floors and changed design-variable grouping at the config boundary."""

    path = project_root / "configs" / "frames" / "three_story_frame.json"
    content = json.loads(path.read_text(encoding="utf-8"))
    content["floors"] = content["floors"][:-1]
    with pytest.raises(ValidationError, match="floor levels"):
        FrameDefinition.model_validate(content)

    content = json.loads(path.read_text(encoding="utf-8"))
    content["design_groups"] = ["1", "2", "3", "4", "6"]
    with pytest.raises(ValidationError, match="exactly 1 through 5"):
        FrameDefinition.model_validate(content)


def test_topology_svg_is_deterministic(
    definition: FrameDefinition,
    tmp_path: Path,
) -> None:
    """Render identical SVG bytes twice without clock-dependent metadata."""

    generated = generate_frame(definition)
    first = tmp_path / "first.svg"
    second = tmp_path / "second.svg"
    save_topology_figure(generated, first)
    save_topology_figure(generated, second)
    assert first.read_bytes() == second.read_bytes()
    assert b"\r\n" not in first.read_bytes()
    assert "3-story frame" in first.read_text(encoding="utf-8")
