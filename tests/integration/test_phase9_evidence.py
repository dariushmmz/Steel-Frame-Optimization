"""Layer 1/2 checks for committed Phase 9 bounded-pilot evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from steel_frame_optimization.serialization import sha256_file
from steel_frame_optimization.validation.phase9 import Phase9Artifact, phase9_gate_failures

pytestmark = pytest.mark.integration


def test_committed_phase9_pilot_evidence_passes_authorized_gates() -> None:
    """Committed pilots pass while recording zero unapproved main-campaign evaluations."""

    root = Path(__file__).resolve().parents[2]
    artifact = Phase9Artifact.model_validate_json(
        (root / "docs" / "evidence" / "phase09" / "ga_pilot_and_runtime.json").read_text(
            encoding="utf-8"
        )
    )
    assert phase9_gate_failures(artifact) == ()
    assert artifact.main_campaign_status == "blocked_pending_client_penalty_approval"
    assert artifact.main_campaign_evaluations_executed == 0
    assert artifact.domain_sizes == (270, 259, 228, 228, 232)
    assert len(artifact.pilot_runs) == 2
    assert artifact.pilot_aggregate.run_count == 2
    for plot in artifact.convergence_plots:
        path = root / plot.path
        assert path.stat().st_size == plot.size_bytes
        assert sha256_file(path) == plot.sha256
