"""Layer 1/2 regression checks for the completed Phase 9 production campaign."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from steel_frame_optimization.serialization import sha256_file

pytestmark = pytest.mark.integration


def test_completed_main_campaign_has_exact_accounting_and_artifacts() -> None:
    """All 30 seeds preserve identities, budgets, reevaluations, and plot hashes."""

    root = Path(__file__).resolve().parents[2]
    evidence_directory = root / "docs" / "evidence" / "phase09" / "main_campaign"
    summary = json.loads(
        (evidence_directory / "main_campaign_summary.json").read_text(encoding="utf-8")
    )
    assert summary["seed_count"] == 30
    assert summary["evaluation_requests_total"] == 300_000
    assert (
        summary["structural_executions_total"] + summary["cache_hits_total"]
        == summary["evaluation_requests_total"]
    )
    assert summary["all_seed_accounting_exact"]
    assert summary["all_champions_independently_reevaluated_equal"]
    assert [record["seed"] for record in summary["seed_records"]] == list(range(30))
    assert all(record["evaluation_requests"] == 10_000 for record in summary["seed_records"])
    assert all(
        record["structural_executions"] + record["cache_hits"] == 10_000
        for record in summary["seed_records"]
    )
    assert all(
        record["independently_reevaluated_champion_equal"] for record in summary["seed_records"]
    )
    assert summary["aggregate"]["feasible_champion_count"] == 30
    assert summary["aggregate"]["infeasible_or_failed_champion_count"] == 0
    failure_counts = summary["aggregate"]["final_population_status_counts"]
    assert failure_counts["invalid_input"] == 0
    assert failure_counts["numerical_failure"] == 0
    assert failure_counts["structural_failure"] == 0
    assert len(summary["convergence_artifacts"]) == 31
    for artifact in summary["convergence_artifacts"]:
        path = evidence_directory / Path(artifact["path"]).name
        assert path.stat().st_size == artifact["size_bytes"]
        assert sha256_file(path) == artifact["sha256"]
