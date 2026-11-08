"""Generate deterministic three-story pushover and FEMA-constraint evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from steel_frame_optimization.analysis.convergence import NonlinearSolverPolicy
from steel_frame_optimization.analysis.pushover import (
    DesignPushoverResult,
    PushoverPolicy,
    evaluate_design_pushover,
)
from steel_frame_optimization.config.schemas import StrictConfigModel
from steel_frame_optimization.models.frame import FrameDefinition, generate_frame
from steel_frame_optimization.sections.records import SectionRecord
from steel_frame_optimization.sections.repository import SectionRepository
from steel_frame_optimization.serialization import (
    canonical_json_dumps,
    sha256_config,
)


class PublishedDriftComparison(StrictConfigModel):
    """Retain one transparent REF-A Table 5 drift comparison without a pass claim."""

    design_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    calculated_story_drift_percent: tuple[float, ...]
    published_story_drift_percent: tuple[float, ...]
    signed_differences_percentage_points: tuple[float, ...]
    source: Literal["REF-A Table 5"] = "REF-A Table 5"
    acceptance_role: Literal["research_diagnostic_only"] = "research_diagnostic_only"


class RefinementPoint(StrictConfigModel):
    """Persist the CP response from one predeclared roof-increment refinement."""

    roof_increment_ratio: float = Field(gt=0.0, allow_inf_nan=False)
    cp_base_shear_n: float = Field(gt=0.0, allow_inf_nan=False)
    cp_story_drift_ratios: tuple[float, ...]
    final_curve_point_count: int = Field(gt=0)
    failed_attempt_count: int = Field(ge=0)


class Phase7Artifact(StrictConfigModel):
    """Persist full published-design responses and all numerical Phase 7 gate metrics."""

    schema_version: Literal["1.0"] = "1.0"
    frame_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    policy: PushoverPolicy
    baseline_design_results: tuple[DesignPushoverResult, ...]
    refinement_points: tuple[RefinementPoint, ...]
    maximum_refinement_relative_spread: float = Field(ge=0.0, allow_inf_nan=False)
    maximum_target_landing_error_m: float = Field(ge=0.0, allow_inf_nan=False)
    maximum_lateral_equilibrium_relative_residual: float = Field(ge=0.0, allow_inf_nan=False)
    maximum_vertical_equilibrium_relative_residual: float = Field(ge=0.0, allow_inf_nan=False)
    maximum_constraint_translation_residual_m: float = Field(ge=0.0, allow_inf_nan=False)
    maximum_hinge_force_mismatch_relative: float = Field(ge=0.0, allow_inf_nan=False)
    repeat_result_sha256: tuple[str, str]
    deterministic_repeat_equal: bool
    published_drift_comparisons: tuple[PublishedDriftComparison, ...]
    sources: tuple[str, ...]
    limitations: tuple[str, ...]


def _load_solver_policy(path: Path) -> NonlinearSolverPolicy:
    """Load the locked Phase 6 nonlinear solver policy."""

    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or not isinstance(content.get("solver_policy"), dict):
        raise ValueError("Phase 6 policy must contain a solver_policy object")
    return NonlinearSolverPolicy.model_validate(content["solver_policy"])


def _relative_spread(values: tuple[float, ...]) -> float:
    """Return max-minus-min relative to the finest positive response."""

    return (max(values) - min(values)) / max(abs(values[-1]), 1e-30)


def _refinement_point(result: DesignPushoverResult) -> RefinementPoint:
    """Extract the CP response and fallback count from one complete evaluation."""

    cp = result.final.target_states[-1]
    return RefinementPoint(
        roof_increment_ratio=result.roof_increment_ratio,
        cp_base_shear_n=cp.base_shear_n,
        cp_story_drift_ratios=cp.story_drift_ratios,
        final_curve_point_count=len(result.final.curve),
        failed_attempt_count=sum(
            attempt.return_code != 0
            for segment in result.final.displacement_segments
            for attempt in segment.attempts
        ),
    )


def _published_comparisons(
    results: tuple[DesignPushoverResult, ...], designs: list[object]
) -> tuple[PublishedDriftComparison, ...]:
    """Compare calculated target drifts with the paper's reported percentages."""

    raw_by_id: dict[str, dict[str, object]] = {}
    for raw in designs:
        if not isinstance(raw, dict) or not isinstance(raw.get("design_id"), str):
            raise ValueError("published design entries must be objects with design_id")
        raw_by_id[raw["design_id"]] = raw
    comparisons: list[PublishedDriftComparison] = []
    for result in results:
        raw = raw_by_id[result.design_id]
        published = raw.get("published_story_drift_percent")
        if not isinstance(published, dict):
            raise ValueError("published design must contain drift percentages")
        for state in result.final.target_states:
            values = published.get(state.target.target_id)
            if values is None:
                continue
            if not isinstance(values, list) or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool) for value in values
            ):
                raise ValueError(f"missing published drifts for {state.target.target_id}")
            calculated = tuple(value * 100.0 for value in state.story_drift_ratios)
            reference = tuple(float(value) for value in values)
            comparisons.append(
                PublishedDriftComparison(
                    design_id=result.design_id,
                    target_id=state.target.target_id,
                    calculated_story_drift_percent=calculated,
                    published_story_drift_percent=reference,
                    signed_differences_percentage_points=tuple(
                        actual - expected
                        for actual, expected in zip(calculated, reference, strict=True)
                    ),
                )
            )
    return tuple(comparisons)


def build_phase7_artifact(
    *,
    frame_path: Path,
    sections_path: Path,
    solver_policy_path: Path,
    pushover_policy_path: Path,
    published_designs_path: Path,
) -> Phase7Artifact:
    """Run published designs, refinement, repeatability, and comparison evidence."""

    generated = generate_frame(FrameDefinition.from_json(frame_path))
    repository = SectionRepository.from_json(sections_path)
    solver_policy = _load_solver_policy(solver_policy_path)
    policy = PushoverPolicy.from_json(pushover_policy_path)
    published_content = json.loads(published_designs_path.read_text(encoding="utf-8"))
    if not isinstance(published_content, dict) or not isinstance(
        published_content.get("designs"), list
    ):
        raise ValueError("published design file must contain a designs list")
    designs = published_content["designs"]

    baseline_results: list[DesignPushoverResult] = []
    sections_for_design: dict[str, dict[str, SectionRecord]] = {}
    for raw in designs:
        if not isinstance(raw, dict):
            raise ValueError("published design entries must be objects")
        design_id = raw.get("design_id")
        names = raw.get("sections_by_group")
        if not isinstance(design_id, str) or not isinstance(names, dict):
            raise ValueError("published design requires design_id and sections_by_group")
        sections = {str(group): repository.get(str(name)) for group, name in names.items()}
        sections_for_design[design_id] = sections
        baseline_results.append(
            evaluate_design_pushover(
                generated,
                sections,
                design_id=design_id,
                policy=policy,
                solver_policy=solver_policy,
                roof_increment_ratio=policy.baseline_roof_increment_ratio,
            )
        )

    case2 = next(result for result in baseline_results if result.design_id == "case2_gwo")
    case2_sections = sections_for_design["case2_gwo"]
    refinements: list[DesignPushoverResult] = []
    for increment in policy.refinement_roof_increment_ratios:
        if increment == policy.baseline_roof_increment_ratio:
            refinements.append(case2)
        else:
            refinements.append(
                evaluate_design_pushover(
                    generated,
                    case2_sections,
                    design_id="case2_gwo",
                    policy=policy,
                    solver_policy=solver_policy,
                    roof_increment_ratio=increment,
                )
            )
    refinement_points = tuple(_refinement_point(result) for result in refinements)
    refinement_series = [tuple(point.cp_base_shear_n for point in refinement_points)]
    refinement_series.extend(
        tuple(point.cp_story_drift_ratios[index] for point in refinement_points)
        for index in range(len(refinement_points[-1].cp_story_drift_ratios))
    )
    maximum_refinement_spread = max(_relative_spread(values) for values in refinement_series)

    repeat = evaluate_design_pushover(
        generated,
        case2_sections,
        design_id="case2_gwo",
        policy=policy,
        solver_policy=solver_policy,
        roof_increment_ratio=policy.baseline_roof_increment_ratio,
    )
    repeat_hashes = (sha256_config(case2), sha256_config(repeat))
    results = tuple(baseline_results)
    all_passes = tuple(
        pass_result for result in results for pass_result in (result.preliminary, result.final)
    )
    all_states = tuple(state for pass_result in all_passes for state in pass_result.target_states)
    all_points = tuple(point for pass_result in all_passes for point in pass_result.curve)
    return Phase7Artifact(
        frame_id=generated.definition.frame_id,
        dataset_id=repository.dataset.dataset_id,
        policy=policy,
        baseline_design_results=results,
        refinement_points=refinement_points,
        maximum_refinement_relative_spread=maximum_refinement_spread,
        maximum_target_landing_error_m=max(state.landing_error_m for state in all_states),
        maximum_lateral_equilibrium_relative_residual=max(
            point.lateral_equilibrium_relative_residual for point in all_points
        ),
        maximum_vertical_equilibrium_relative_residual=max(
            point.vertical_equilibrium_relative_residual for point in all_points
        ),
        maximum_constraint_translation_residual_m=max(
            point.maximum_constraint_translation_residual_m for point in all_points
        ),
        maximum_hinge_force_mismatch_relative=max(
            state.maximum_hinge_force_mismatch_relative for state in all_states
        ),
        repeat_result_sha256=repeat_hashes,
        deterministic_repeat_equal=repeat_hashes[0] == repeat_hashes[1],
        published_drift_comparisons=_published_comparisons(results, designs),
        sources=(
            "REF-A PDF pp. 2-3 / journal pp. 78-79, Eqs. (1)-(10)",
            "REF-A PDF pp. 7-8 / journal pp. 83-84, Eqs. (27)-(31), Tables 4-5, Fig. 7",
            "FEMA 356 printed pp. 5-13 to 5-15 and Table 5-6 pp. 5-40 to 5-43",
            "ASM-012 through ASM-015 and ASM-030 through ASM-034",
        ),
        limitations=(
            "REF-A Fig. 7 has no numerical curve ordinates; Table 5 drift values are retained as "
            "research diagnostics and are not fitted acceptance targets",
            "penalty constraints are accepted only with the configured measured "
            "translation-residual gate",
            "results use owner-authorized research assumptions and are not professional "
            "engineering certification",
        ),
    )


def phase7_gate_failures(artifact: Phase7Artifact) -> tuple[str, ...]:
    """Return every failed software/numerical gate without judging design feasibility."""

    policy = artifact.policy
    failures: list[str] = []
    if not all(
        result.preliminary.converged and result.final.converged
        for result in artifact.baseline_design_results
    ):
        failures.append("one or more preliminary/final pushover passes did not converge")
    checks = (
        (
            artifact.maximum_target_landing_error_m,
            policy.target_landing_absolute_tolerance_m,
            "target landing",
        ),
        (
            artifact.maximum_lateral_equilibrium_relative_residual,
            policy.equilibrium_relative_tolerance,
            "lateral equilibrium",
        ),
        (
            artifact.maximum_vertical_equilibrium_relative_residual,
            policy.equilibrium_relative_tolerance,
            "vertical equilibrium",
        ),
        (
            artifact.maximum_constraint_translation_residual_m,
            policy.constraint_translation_absolute_tolerance_m,
            "constraint translation residual",
        ),
        (
            artifact.maximum_hinge_force_mismatch_relative,
            policy.hinge_force_relative_tolerance,
            "hinge/member force consistency",
        ),
        (
            artifact.maximum_refinement_relative_spread,
            policy.refinement_relative_tolerance,
            "refinement spread",
        ),
    )
    failures.extend(label for actual, limit, label in checks if actual > limit)
    if not artifact.deterministic_repeat_equal:
        failures.append("deterministic repeat hashes differ")
    return tuple(failures)


def _parse_args() -> argparse.Namespace:
    """Parse explicit Phase 7 evidence paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--sections", type=Path, required=True)
    parser.add_argument("--solver-policy", type=Path, required=True)
    parser.add_argument("--pushover-policy", type=Path, required=True)
    parser.add_argument("--published-designs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Write canonical evidence and fail if any software/numerical gate is unsatisfied."""

    args = _parse_args()
    artifact = build_phase7_artifact(
        frame_path=args.frame,
        sections_path=args.sections,
        solver_policy_path=args.solver_policy,
        pushover_policy_path=args.pushover_policy,
        published_designs_path=args.published_designs,
    )
    failures = phase7_gate_failures(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output_stream:
        output_stream.write(canonical_json_dumps(artifact))
    print(
        f"Phase 7 designs={len(artifact.baseline_design_results)}, "
        f"landing={artifact.maximum_target_landing_error_m:.3e} m, "
        f"equilibrium={artifact.maximum_lateral_equilibrium_relative_residual:.3e}, "
        f"refinement={artifact.maximum_refinement_relative_spread:.3e}, "
        f"repeat={artifact.deterministic_repeat_equal}, failures={len(failures)}"
    )
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
