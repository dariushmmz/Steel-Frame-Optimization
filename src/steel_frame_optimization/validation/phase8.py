"""Generate deterministic Phase 8 objective, dispatch, penalty, and cache evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from steel_frame_optimization.config.schemas import StrictConfigModel
from steel_frame_optimization.optimization.evaluator import (
    EvaluationResult,
    EvaluatorIdentity,
    EvaluatorPaths,
    StructuralEvaluator,
    evaluation_order_key,
)
from steel_frame_optimization.serialization import write_canonical_json


class RepresentativeDesign(StrictConfigModel):
    """Declare one source-visible Phase 8 dispatch and ranking fixture."""

    design_id: str = Field(min_length=1)
    expected_status: Literal["feasible", "stage1_infeasible", "stage2_infeasible"]
    sections_by_group: dict[str, str]

    @model_validator(mode="after")
    def require_five_groups(self) -> RepresentativeDesign:
        """Require exactly the three-story design groups."""

        if set(self.sections_by_group) != {"1", "2", "3", "4", "5"}:
            raise ValueError("representative design requires groups 1 through 5")
        return self


class RepresentativeDesignSet(StrictConfigModel):
    """Validate the three required feasible/infeasible dispatch fixtures."""

    schema_version: Literal["1.0"] = "1.0"
    source: str = Field(min_length=1)
    designs: tuple[RepresentativeDesign, ...] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def require_all_statuses(self) -> RepresentativeDesignSet:
        """Require one fixture for each structural evaluation disposition."""

        statuses = {design.expected_status for design in self.designs}
        if statuses != {"feasible", "stage1_infeasible", "stage2_infeasible"}:
            raise ValueError("representative designs must cover all structural statuses")
        return self


class Phase8Artifact(StrictConfigModel):
    """Persist representative outcomes and deterministic cache/ranking checks."""

    schema_version: Literal["1.0"] = "1.0"
    evaluator_identity: EvaluatorIdentity
    results: tuple[EvaluationResult, ...]
    repeated_request_cache_hit: bool
    repeated_request_result_equal: bool
    deterministic_rank_order: tuple[str, ...]
    all_expected_statuses_match: bool
    all_feasible_scores_below_one: bool
    all_infeasible_scores_at_least_one: bool
    sources: tuple[str, ...]
    limitations: tuple[str, ...]


def build_phase8_artifact(
    *, project_root: Path, representative_designs_path: Path
) -> Phase8Artifact:
    """Evaluate all structural branches and verify cache reuse and rank separation."""

    design_set = RepresentativeDesignSet.model_validate(
        json.loads(representative_designs_path.read_text(encoding="utf-8"))
    )
    evaluator = StructuralEvaluator(EvaluatorPaths.from_project_root(project_root))
    pairs: list[tuple[RepresentativeDesign, EvaluationResult]] = []
    for design in design_set.designs:
        execution = evaluator.evaluate(
            evaluation_id=design.design_id,
            section_names_by_group=design.sections_by_group,
        )
        pairs.append((design, execution.result))
    first_design, first_result = pairs[0]
    repeated = evaluator.evaluate(
        evaluation_id=f"{first_design.design_id}-repeat",
        section_names_by_group=first_design.sections_by_group,
    )
    results = tuple(result for _, result in pairs)
    ordered = sorted(pairs, key=lambda pair: evaluation_order_key(pair[1]))
    feasible_scores = tuple(
        result.penalty.ranking_score
        for result in results
        if result.status == "feasible" and result.penalty is not None
    )
    infeasible_scores = tuple(
        result.penalty.ranking_score
        for result in results
        if result.status in {"stage1_infeasible", "stage2_infeasible"}
        and result.penalty is not None
    )
    return Phase8Artifact(
        evaluator_identity=evaluator.identity,
        results=results,
        repeated_request_cache_hit=repeated.cache_hit,
        repeated_request_result_equal=repeated.result == first_result,
        deterministic_rank_order=tuple(design.design_id for design, _ in ordered),
        all_expected_statuses_match=all(
            design.expected_status == result.status for design, result in pairs
        ),
        all_feasible_scores_below_one=all(score < 1.0 for score in feasible_scores),
        all_infeasible_scores_at_least_one=all(score >= 1.0 for score in infeasible_scores),
        sources=(
            "REF-A Eq. (20), PDF p. 4 / journal p. 80",
            "REF-A Tables 4-5, PDF pp. 7-8 / journal pp. 83-84",
            "ASM-016 normalized additive feasibility-offset policy",
            "approved 273-section project Wmax 1802196.9874227915 N",
        ),
        limitations=(
            "project Wmax remains 8.704% below the paper back-calculation and is not fitted",
            "Stage-1-infeasible designs have no CP uniformity or paper objective; their score "
            "base is explicitly weight-only",
            "client penalty approval remains required before main campaigns",
            "results are not independently engineering-validated or professionally certified",
        ),
    )


def phase8_gate_failures(artifact: Phase8Artifact) -> tuple[str, ...]:
    """Return every failed deterministic dispatch, cache, and ranking gate."""

    checks = (
        (artifact.repeated_request_cache_hit, "identical design did not hit cache"),
        (artifact.repeated_request_result_equal, "cache changed structural result"),
        (artifact.all_expected_statuses_match, "representative status mismatch"),
        (artifact.all_feasible_scores_below_one, "feasible score is not below one"),
        (artifact.all_infeasible_scores_at_least_one, "infeasible score is below one"),
    )
    return tuple(message for passed, message in checks if not passed)


def _parse_args() -> argparse.Namespace:
    """Parse explicit Phase 8 evidence paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--representative-designs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Write canonical Phase 8 evidence and fail on any implementation gate."""

    args = _parse_args()
    artifact = build_phase8_artifact(
        project_root=args.project_root,
        representative_designs_path=args.representative_designs,
    )
    failures = phase8_gate_failures(artifact)
    write_canonical_json(args.output, artifact)
    statuses = ",".join(result.status for result in artifact.results)
    print(
        f"Phase 8 statuses={statuses}, cache_hit={artifact.repeated_request_cache_hit}, "
        f"rank_order={artifact.deterministic_rank_order}, failures={len(failures)}"
    )
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
