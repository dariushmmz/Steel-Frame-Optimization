"""Analytical invariants for nine-story filtering, modal analysis, and Stage 1."""

from __future__ import annotations

from pathlib import Path

import pytest

from steel_frame_optimization.models.frame import FrameDefinition, generate_frame
from steel_frame_optimization.sections.repository import SectionRepository
from steel_frame_optimization.validation.phase11 import (
    NineStoryFilterPolicy,
    build_nine_story_filter,
    build_phase11_artifact,
)

pytestmark = pytest.mark.analytical


def _phase11_paths(project_root: Path) -> dict[str, Path]:
    """Return all explicit Phase 11 input paths."""

    return {
        "frame_path": project_root / "configs/frames/nine_story_frame.json",
        "sections_path": project_root / "data/processed/aisc_v14_1_w_shapes.json",
        "case_path": project_root / "configs/cases/nine_story_case2_gwo.json",
        "stage1_policy_path": project_root / "configs/analysis/stage1_policy.json",
        "filter_policy_path": project_root / "configs/analysis/phase11_nine_story_filter.json",
        "reference_path": project_root / "configs/cases/nine_story_case2_stage1_reference.json",
    }


def test_preliminary_filter_uses_correct_domains_and_freezes_project_baseline(
    project_root: Path,
) -> None:
    """Reproduce all thirteen project counts while retaining the publication diagnostic."""

    generated = generate_frame(
        FrameDefinition.from_json(project_root / "configs/frames/nine_story_frame.json")
    )
    repository = SectionRepository.from_json(
        project_root / "data/processed/aisc_v14_1_w_shapes.json"
    )
    policy = NineStoryFilterPolicy.from_json(
        project_root / "configs/analysis/phase11_nine_story_filter.json"
    )
    result = build_nine_story_filter(generated=generated, repository=repository, policy=policy)
    assert result.column_base_domain_size == 36
    assert result.beam_base_domain_size == 273
    assert result.actual_removed_counts == {
        "1": 1,
        "2": 7,
        "3": 5,
        "4": 0,
        "5": 3,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 45,
        "10": 45,
        "11": 45,
        "12": 45,
        "13": 41,
    }
    assert result.published_counts_match is False
    assert all(
        len(group.rejected_sections) == group.actual_removed_count for group in result.groups
    )
    assert all(group.threshold_design_capacity >= group.demand for group in result.groups)


def test_published_case2_modal_and_stage1_invariants(project_root: Path) -> None:
    """Require positive modes, direct combinations, equilibrium, and granular constraints."""

    result = build_phase11_artifact(**_phase11_paths(project_root))
    assert len(result.modal.periods_s) == 9
    assert all(
        first > second
        for first, second in zip(result.modal.periods_s, result.modal.periods_s[1:], strict=False)
    )
    assert result.modal.periods_s[0] == pytest.approx(3.7395242390567103, rel=1e-10)
    stage1 = result.published_case2_stage1
    assert set(stage1.analyses) == {
        "strength_gravity",
        "strength_seismic_positive",
        "strength_seismic_negative",
        "service_D",
        "service_L",
        "service_D_plus_L",
    }
    assert all(item.combination_evaluated_directly for item in stage1.analyses.values())
    assert all(
        item.equilibrium_relative_residual <= stage1.policy.equilibrium_relative_tolerance
        for item in stage1.analyses.values()
    )
    assert len(stage1.effective_lengths) == 54
    assert len(stage1.constraints) == 452
    assert len(stage1.research_comparisons) == 99
    assert stage1.feasible is True
    assert stage1.failed_constraint_ids == ()
    assert stage1.maximum_utilization == pytest.approx(0.8145215942048479)
