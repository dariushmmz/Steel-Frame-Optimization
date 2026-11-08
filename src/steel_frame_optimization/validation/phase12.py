"""Validate nine-story pushover behavior and publication comparisons before GA campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import Field

from steel_frame_optimization.analysis.pushover import (
    DesignPushoverResult,
    PushoverPolicy,
    evaluate_design_pushover,
)
from steel_frame_optimization.config.schemas import StrictConfigModel
from steel_frame_optimization.models.frame import FrameDefinition, generate_frame
from steel_frame_optimization.optimization.evaluator import _load_solver_policy
from steel_frame_optimization.sections.records import SectionRecord
from steel_frame_optimization.sections.repository import SectionRepository
from steel_frame_optimization.serialization import sha256_config, write_canonical_json
from steel_frame_optimization.validation.phase7 import RefinementPoint


class PublishedNineStoryDriftComparison(StrictConfigModel):
    """Retain one REF-A Table 9 drift comparison as a nonfitted research diagnostic."""

    design_id: str
    target_id: str
    calculated_story_drift_percent: tuple[float, ...]
    published_story_drift_percent: tuple[float, ...]
    signed_differences_percentage_points: tuple[float, ...]
    source: str = "REF-A Table 9"
    acceptance_role: str = "research_diagnostic_only"


class Phase12PushoverArtifact(StrictConfigModel):
    """Persist deterministic nine-story component-to-frame nonlinear verification."""

    schema_version: str = "1.0"
    validation_status: str = "verified research software; not engineering certification"
    frame_id: str
    baseline_design_results: tuple[DesignPushoverResult, ...]
    refinement_points: tuple[RefinementPoint, ...]
    maximum_refinement_relative_spread: float = Field(ge=0.0)
    maximum_target_landing_error_m: float = Field(ge=0.0)
    maximum_lateral_equilibrium_relative_residual: float = Field(ge=0.0)
    maximum_vertical_equilibrium_relative_residual: float = Field(ge=0.0)
    maximum_constraint_translation_residual_m: float = Field(ge=0.0)
    maximum_hinge_force_mismatch_relative: float = Field(ge=0.0)
    deterministic_repeat_equal: bool
    repeat_result_sha256: tuple[str, str]
    published_drift_comparisons: tuple[PublishedNineStoryDriftComparison, ...]


def _refinement_point(result: DesignPushoverResult) -> RefinementPoint:
    """Extract the final CP response and nonlinear fallback count."""

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


def _relative_spread(values: tuple[float, ...]) -> float:
    """Return max-minus-min relative to the finest-mesh response."""

    return (max(values) - min(values)) / max(abs(values[-1]), 1.0e-30)


def _comparisons(
    results: tuple[DesignPushoverResult, ...], raw_designs: list[object]
) -> tuple[PublishedNineStoryDriftComparison, ...]:
    """Build target-by-target REF-A Table 9 diagnostics."""

    raw_by_id = {
        str(raw["design_id"]): raw
        for raw in raw_designs
        if isinstance(raw, dict) and "design_id" in raw
    }
    comparisons: list[PublishedNineStoryDriftComparison] = []
    for result in results:
        raw = raw_by_id[result.design_id]
        published = raw.get("published_story_drift_percent")
        if not isinstance(published, dict):
            raise ValueError("published design must provide story-drift percentages")
        for state in result.final.target_states:
            values = published.get(state.target.target_id)
            if values is None:
                continue
            if not isinstance(values, list):
                raise ValueError(f"missing Table 9 values for {state.target.target_id}")
            expected = tuple(float(value) for value in values)
            actual = tuple(100.0 * value for value in state.story_drift_ratios)
            comparisons.append(
                PublishedNineStoryDriftComparison(
                    design_id=result.design_id,
                    target_id=state.target.target_id,
                    calculated_story_drift_percent=actual,
                    published_story_drift_percent=expected,
                    signed_differences_percentage_points=tuple(
                        observed - reference
                        for observed, reference in zip(actual, expected, strict=True)
                    ),
                )
            )
    return tuple(comparisons)


def build_phase12_pushover_artifact(root: Path) -> Phase12PushoverArtifact:
    """Run published designs, a fixed refinement sequence, and exact repeat verification."""

    generated = generate_frame(
        FrameDefinition.from_json(root / "configs/frames/nine_story_frame.json")
    )
    repository = SectionRepository.from_json(root / "data/processed/aisc_v14_1_w_shapes.json")
    solver = _load_solver_policy(root / "configs/analysis/phase6_nonlinear_policy.json")
    policy = PushoverPolicy.from_json(root / "configs/analysis/phase7_pushover_policy.json")
    content = json.loads(
        (root / "configs/cases/nine_story_published_gwo_stage2.json").read_text(encoding="utf-8")
    )
    raw_designs = content.get("designs")
    if not isinstance(raw_designs, list):
        raise ValueError("nine-story published configuration must contain designs")
    baseline: list[DesignPushoverResult] = []
    sections_by_design: dict[str, dict[str, SectionRecord]] = {}
    for raw in raw_designs:
        if not isinstance(raw, dict) or not isinstance(raw.get("sections_by_group"), dict):
            raise ValueError("published design entry is invalid")
        design_id = str(raw["design_id"])
        sections = {
            str(group): repository.get(str(name))
            for group, name in raw["sections_by_group"].items()
        }
        sections_by_design[design_id] = sections
        baseline.append(
            evaluate_design_pushover(
                generated,
                sections,
                design_id=design_id,
                policy=policy,
                solver_policy=solver,
                roof_increment_ratio=policy.baseline_roof_increment_ratio,
            )
        )
    case2 = next(result for result in baseline if result.design_id == "case2_gwo")
    case2_sections = sections_by_design["case2_gwo"]
    refinements = tuple(
        case2
        if ratio == policy.baseline_roof_increment_ratio
        else evaluate_design_pushover(
            generated,
            case2_sections,
            design_id="case2_gwo",
            policy=policy,
            solver_policy=solver,
            roof_increment_ratio=ratio,
        )
        for ratio in policy.refinement_roof_increment_ratios
    )
    points = tuple(_refinement_point(result) for result in refinements)
    series = [tuple(point.cp_base_shear_n for point in points)]
    series.extend(
        tuple(point.cp_story_drift_ratios[index] for point in points)
        for index in range(len(points[-1].cp_story_drift_ratios))
    )
    repeat = evaluate_design_pushover(
        generated,
        case2_sections,
        design_id="case2_gwo",
        policy=policy,
        solver_policy=solver,
        roof_increment_ratio=policy.baseline_roof_increment_ratio,
    )
    hashes = (sha256_config(case2), sha256_config(repeat))
    results = tuple(baseline)
    passes = tuple(item for result in results for item in (result.preliminary, result.final))
    states = tuple(state for item in passes for state in item.target_states)
    curve = tuple(point for item in passes for point in item.curve)
    return Phase12PushoverArtifact(
        frame_id=generated.definition.frame_id,
        baseline_design_results=results,
        refinement_points=points,
        maximum_refinement_relative_spread=max(_relative_spread(values) for values in series),
        maximum_target_landing_error_m=max(state.landing_error_m for state in states),
        maximum_lateral_equilibrium_relative_residual=max(
            point.lateral_equilibrium_relative_residual for point in curve
        ),
        maximum_vertical_equilibrium_relative_residual=max(
            point.vertical_equilibrium_relative_residual for point in curve
        ),
        maximum_constraint_translation_residual_m=max(
            point.maximum_constraint_translation_residual_m for point in curve
        ),
        maximum_hinge_force_mismatch_relative=max(
            state.maximum_hinge_force_mismatch_relative for state in states
        ),
        deterministic_repeat_equal=hashes[0] == hashes[1],
        repeat_result_sha256=hashes,
        published_drift_comparisons=_comparisons(results, raw_designs),
    )


def main() -> int:
    """Write the canonical Phase 12 nine-story nonlinear artifact."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_canonical_json(args.output, build_phase12_pushover_artifact(args.project_root.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
