"""Layer 1 tests for dependency-free optimization convergence reporting."""

from __future__ import annotations

from pathlib import Path

import pytest

from steel_frame_optimization.optimization.campaign import GenerationRecord
from steel_frame_optimization.reporting.optimization import convergence_svg, write_convergence_svg

pytestmark = pytest.mark.unit


def _history() -> tuple[GenerationRecord, ...]:
    """Return a two-point deterministic convergence fixture."""

    return (
        GenerationRecord(
            epoch_index=0,
            evaluation_requests_total=4,
            structural_executions_total=4,
            cache_hits_total=0,
            best_rank_class=1,
            best_ranking_score=2.0,
            best_feasible_objective=None,
            feasible_count=0,
            stage1_infeasible_count=4,
            stage2_infeasible_count=0,
            failed_count=0,
        ),
        GenerationRecord(
            epoch_index=1,
            evaluation_requests_total=8,
            structural_executions_total=7,
            cache_hits_total=1,
            best_rank_class=0,
            best_ranking_score=0.1,
            best_feasible_objective=0.11,
            feasible_count=1,
            stage1_infeasible_count=2,
            stage2_infeasible_count=1,
            failed_count=0,
        ),
    )


def test_convergence_svg_is_deterministic_and_escapes_title(tmp_path: Path) -> None:
    """Repeated rendering has identical bytes and never injects markup from a title."""

    first = convergence_svg(_history(), title="pilot <seed>")
    second = convergence_svg(_history(), title="pilot <seed>")
    assert first == second
    assert "pilot &lt;seed&gt;" in first
    assert "<polyline" in first
    output = tmp_path / "convergence.svg"
    write_convergence_svg(output, _history(), title="pilot <seed>")
    assert output.read_text(encoding="utf-8") == first


def test_convergence_svg_rejects_missing_finite_score() -> None:
    """A typed no-score failure cannot be silently plotted as numeric convergence."""

    invalid = _history()[0].model_copy(update={"best_ranking_score": None})
    with pytest.raises(ValueError, match="finite"):
        convergence_svg((invalid,), title="invalid")
