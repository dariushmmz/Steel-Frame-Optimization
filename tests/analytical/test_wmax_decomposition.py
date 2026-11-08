"""Layer 2/3 checks for the unresolved three-story Wmax decomposition."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from steel_frame_optimization.models.frame import (
    FrameDefinition,
    build_topology_audit,
    generate_frame,
)
from steel_frame_optimization.sections.domains import DomainCatalog
from steel_frame_optimization.sections.repository import SectionRepository

pytestmark = pytest.mark.analytical


def test_wmax_decomposition_and_nonmutating_hypotheses(project_root: Path) -> None:
    """Reproduce the mismatch and test domain, weight, length, unit, and count hypotheses."""

    definition = FrameDefinition.from_json(
        project_root / "configs" / "frames" / "three_story_frame.json"
    )
    repository = SectionRepository.from_json(
        project_root / "data" / "processed" / "aisc_v14_1_w_shapes.json"
    )
    audit = build_topology_audit(generate_frame(definition), repository)
    groups = {group.group_id: group for group in audit.group_lengths}
    assert {
        group_id: (group.member_count, group.total_length_m) for group_id, group in groups.items()
    } == pytest.approx(
        {
            "1": (6, 23.7744),
            "2": (9, 35.6616),
            "3": (4, 36.576),
            "4": (4, 36.576),
            "5": (4, 36.576),
        }
    )
    assert sum(group.member_count for group in groups.values()) == 27
    assert sum(group.total_length_m for group in groups.values()) == pytest.approx(169.164)

    catalog = DomainCatalog.from_json(
        project_root / "configs" / "frames" / "phase2_candidate_domains.json"
    )
    frame = next(item for item in catalog.frames if item.frame_id == "3-story")
    assert all(group.base_family == "all_w" for group in frame.groups)
    heaviest = max(repository.family("all_w"), key=lambda record: record.weight_n_per_m)
    assert heaviest.name == "W14X730"
    assert heaviest.source_weight_lb_per_ft == 730.0
    conversion_n_per_m_per_lb_per_ft = 4.4482216152605 / 0.3048
    assert heaviest.weight_n_per_m == pytest.approx(
        730.0 * conversion_n_per_m_per_lb_per_ft, rel=1e-15
    )
    assert audit.wmax_cross_check.computed_wmax_n == pytest.approx(1_802_196.9874227915)

    reference_n = 1_974_010.0
    required_weight_n_per_m = reference_n / 169.164
    required_weight_lb_per_ft = required_weight_n_per_m / conversion_n_per_m_per_lb_per_ft
    assert required_weight_lb_per_ft == pytest.approx(799.5947779608282)
    assert required_weight_lb_per_ft > heaviest.source_weight_lb_per_ft

    w14x808_hypothesis_n = 808.0 * conversion_n_per_m_per_lb_per_ft * 169.164
    assert (w14x808_hypothesis_n - reference_n) / reference_n == pytest.approx(0.010511852091640003)
    count_hypothesis_n = heaviest.weight_n_per_m * (12 * 9.144 + 19 * 3.9624)
    assert abs(count_hypothesis_n - reference_n) / reference_n < 0.002
    assert (
        sum(group.member_count for group_id, group in groups.items() if group_id in {"1", "2"})
        == 15
    )

    evidence = json.loads(
        (project_root / "docs" / "evidence" / "phase03" / "wmax_decomposition.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["conclusion"]["reproduces_reference_without_conflict"] is False
    assert evidence["member_count_hypothesis"]["conflicts_with_topology"] is True
