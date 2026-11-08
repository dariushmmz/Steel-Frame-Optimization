"""Verify the bounded zero-cost GitHub Actions execution contract."""

from __future__ import annotations

import json
from pathlib import Path


def test_github_workflow_is_bounded_and_uses_only_standard_linux_runner() -> None:
    """Prevent accidental paid runners, unbounded jobs, or public section data."""

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/phase12-github.yml").read_text(encoding="utf-8")

    assert "runs-on: ubuntu-24.04" in workflow
    assert "timeout-minutes: 330" in workflow
    assert "retention-days: 1" in workflow
    assert "AISC_DATA_GZ_B64_1" in workflow
    assert "AISC_DATA_GZ_B64_2" in workflow
    assert "phase12-checkpoint" in workflow
    assert "large" not in workflow.lower()


def test_github_identity_freezes_chunk_and_scientific_identities() -> None:
    """Keep runner limits separate from the frozen campaign and evaluator identities."""

    root = Path(__file__).resolve().parents[2]
    identity = json.loads(
        (root / "configs/optimization/phase12_github_identity.json").read_text(encoding="utf-8")
    )

    assert identity["chunk_epochs"] == 40
    assert identity["maximum_job_minutes"] == 330
    assert identity["artifact_retention_days"] == 1
    assert set(identity["campaigns"]) == {
        "case1_filtered",
        "case1_no_filter",
        "case2",
    }
    assert all(
        len(campaign["campaign_sha256"]) == 64 and len(campaign["implementation_sha256"]) == 64
        for campaign in identity["campaigns"].values()
    )


def test_processed_section_domain_is_not_tracked_by_public_workflow() -> None:
    """Document that the restricted runtime input is reconstructed from encrypted secrets."""

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/phase12-github.yml").read_text(encoding="utf-8")

    assert "data/processed/aisc_v14_1_w_shapes.json" in workflow
    assert "sha256sum --check --strict" in workflow
