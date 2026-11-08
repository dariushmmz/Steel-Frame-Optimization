"""Layer 1 tests for valid/invalid fixtures and explicit open decisions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from steel_frame_optimization.config.schemas import (
    AnalysisCase,
    DecisionVariant,
    FrameSpec,
    OptimizationCampaign,
    SolverPolicy,
)
from steel_frame_optimization.models.failures import FailureCategory, FailureRecord
from steel_frame_optimization.models.manifests import ArtifactManifest

pytestmark = pytest.mark.unit

SCHEMAS: dict[str, type[BaseModel]] = {
    "frame": FrameSpec,
    "analysis_case": AnalysisCase,
    "decision_variant": DecisionVariant,
    "solver_policy": SolverPolicy,
    "optimization_campaign": OptimizationCampaign,
    "artifact_manifest": ArtifactManifest,
}


@pytest.mark.parametrize(("fixture_name", "model_type"), SCHEMAS.items())
def test_valid_fixture(fixture_name: str, model_type: type[BaseModel], project_root: Path) -> None:
    """Accept each complete valid fixture without coercing unknown fields."""

    path = project_root / "tests" / "fixtures" / "valid" / f"{fixture_name}.json"
    model = model_type.model_validate_json(path.read_text(encoding="utf-8"))
    assert model.model_dump(mode="json")["schema_version"] == "1.0"


@pytest.mark.parametrize(("fixture_name", "model_type"), SCHEMAS.items())
def test_invalid_fixture(
    fixture_name: str, model_type: type[BaseModel], project_root: Path
) -> None:
    """Reject one representative invalid fixture for every required schema."""

    path = project_root / "tests" / "fixtures" / "invalid" / f"{fixture_name}.json"
    with pytest.raises(ValidationError):
        model_type.model_validate_json(path.read_text(encoding="utf-8"))


def test_case_configs_preserve_dec_001_and_leave_dec_002_open(project_root: Path) -> None:
    """Require named 0.025H/0.02H variants without choosing a performance-level mapping."""

    case_directory = project_root / "configs" / "cases"
    primary = AnalysisCase.model_validate_json(
        (case_directory / "ls_primary_0_025H.json").read_text(encoding="utf-8")
    )
    sensitivity = AnalysisCase.model_validate_json(
        (case_directory / "ls_sensitivity_0_02H.json").read_text(encoding="utf-8")
    )
    assert primary.ls_target_ratio == 0.025
    assert sensitivity.ls_target_ratio == 0.02
    assert primary.performance_level_mapping is None
    assert sensitivity.performance_level_mapping is None
    assert primary.mapping_status == "awaiting_structural_engineer_DEC-002"


def test_project_placeholder_and_delegated_configs_validate(project_root: Path) -> None:
    """Validate the unimplemented solver and both delegated campaign configurations."""

    solver = SolverPolicy.model_validate_json(
        (project_root / "configs" / "analysis" / "phase1_unimplemented_solver.json").read_text(
            encoding="utf-8"
        )
    )
    assert solver.status == "placeholder_unimplemented"
    campaign_directory = project_root / "configs" / "optimization"
    three_story = OptimizationCampaign.model_validate_json(
        (campaign_directory / "delegated_3_story_campaign.json").read_text(encoding="utf-8")
    )
    nine_story = OptimizationCampaign.model_validate_json(
        (campaign_directory / "delegated_9_story_campaign.json").read_text(encoding="utf-8")
    )
    assert three_story.structural_evaluation_budget == 10_000
    assert nine_story.structural_evaluation_budget == 20_000
    assert three_story.status == "agent_decided_pending_ratification_DEC-011_DEC-012"


def test_failure_taxonomy_matches_enum_and_phase8_ranking_policy(project_root: Path) -> None:
    """Keep failures typed and record the delegated-owner ASM-016 approval."""

    path = project_root / "configs" / "analysis" / "failure_taxonomy.json"
    taxonomy = json.loads(path.read_text(encoding="utf-8"))
    assert set(taxonomy["categories"]) == {category.value for category in FailureCategory}
    assert (
        taxonomy["optimization_ordering_status"]
        == "approved_by_delegated_technical_owner_2026-10-30"
    )
    record = FailureRecord(
        category=FailureCategory.NOT_IMPLEMENTED,
        stage="phase_1",
        message="Structural evaluation is not implemented",
        recoverable=False,
    )
    assert record.category is FailureCategory.NOT_IMPLEMENTED
