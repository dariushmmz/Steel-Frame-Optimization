"""Integration regression for canonical Phase 11 evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from steel_frame_optimization.serialization import canonical_json_dumps
from steel_frame_optimization.validation.phase11 import Phase11Artifact, build_phase11_artifact

pytestmark = pytest.mark.integration


def test_phase11_artifact_matches_committed_evidence(project_root: Path) -> None:
    """Rebuild all nine-story Stage 1 results and require canonical evidence equality."""

    actual = build_phase11_artifact(
        frame_path=project_root / "configs/frames/nine_story_frame.json",
        sections_path=project_root / "data/processed/aisc_v14_1_w_shapes.json",
        case_path=project_root / "configs/cases/nine_story_case2_gwo.json",
        stage1_policy_path=project_root / "configs/analysis/stage1_policy.json",
        filter_policy_path=project_root / "configs/analysis/phase11_nine_story_filter.json",
        reference_path=project_root / "configs/cases/nine_story_case2_stage1_reference.json",
    )
    evidence_path = project_root / "docs/evidence/phase11/phase11_artifact.json"
    committed = Phase11Artifact.model_validate_json(evidence_path.read_text(encoding="utf-8"))
    assert committed == actual
    assert evidence_path.read_text(encoding="utf-8") == canonical_json_dumps(actual)
