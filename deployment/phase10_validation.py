"""Build independent three-story validation and model-freeze evidence.

This module intentionally lives outside the production package.  Its calculations exercise the
frozen evaluator through public interfaces while keeping the structural implementation identity
used by the Phase 9 campaign unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import Field

from steel_frame_optimization.analysis.convergence import NonlinearSolverPolicy
from steel_frame_optimization.analysis.pushover import (
    DesignPushoverResult,
    evaluate_design_pushover,
)
from steel_frame_optimization.config.schemas import StrictConfigModel
from steel_frame_optimization.constraints.lrfd import (
    beam_column_interaction_utilization,
    nominal_compression_strength_with_local_buckling,
)
from steel_frame_optimization.objective.functions import drift_uniformity_term
from steel_frame_optimization.optimization.evaluator import EvaluatorPaths, StructuralEvaluator
from steel_frame_optimization.serialization import (
    canonical_json_dumps,
    sha256_config,
    sha256_file,
)
from steel_frame_optimization.validation.phase5 import evaluate_stage1_design


class GravitySensitivityPolicy(StrictConfigModel):
    """Declare the alternative nonlinear gravity factors before execution."""

    label: str = Field(min_length=1)
    dead_factor: float = Field(gt=0.0, allow_inf_nan=False)
    live_factor: float = Field(gt=0.0, allow_inf_nan=False)
    acceptance_role: Literal["diagnostic_only"]


class WeakAxisSensitivityPolicy(StrictConfigModel):
    """Declare the weak-axis capacity diagnostic before execution."""

    label: str = Field(min_length=1)
    acceptance_role: Literal["diagnostic_only"]


class PublishedReference(StrictConfigModel):
    """Store one directly transcribed REF-A Case 2 comparison row."""

    algorithm: Literal["PSO", "GWO"]
    objective: float = Field(gt=0.0, allow_inf_nan=False)
    weight_kn: float = Field(gt=0.0, allow_inf_nan=False)
    uniformity_term: float = Field(gt=0.0, allow_inf_nan=False)
    source: str = Field(min_length=1)


class Phase10Policy(StrictConfigModel):
    """Freeze Phase 10 sources, tolerances, and diagnostic variants."""

    schema_version: Literal["1.0"] = "1.0"
    validation_id: str = Field(min_length=1)
    phase9_summary: Path
    phase9_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    phase9_completion_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    champion_relative_tolerance: float = Field(gt=0.0, allow_inf_nan=False)
    objective_relative_tolerance: float = Field(gt=0.0, allow_inf_nan=False)
    pushover_refinement_ratios: tuple[float, ...] = Field(min_length=3)
    pushover_refinement_relative_tolerance: float = Field(gt=0.0, allow_inf_nan=False)
    no_filter_study: str = Field(min_length=1)
    gravity_sensitivity: GravitySensitivityPolicy
    weak_axis_sensitivity: WeakAxisSensitivityPolicy
    published_case2_references: tuple[PublishedReference, ...] = Field(min_length=2)
    assumption_ids: tuple[str, ...] = Field(min_length=1)

    @classmethod
    def from_json(cls, path: Path) -> Phase10Policy:
        """Load the predeclared Phase 10 validation policy."""

        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


class FreshChampionValidation(StrictConfigModel):
    """Record one seed champion's cache-bypassed fresh-process comparison."""

    seed: int = Field(ge=0)
    sections_by_group: dict[str, str]
    status: str
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    objective: float = Field(gt=0.0, allow_inf_nan=False)
    expected_objective: float = Field(gt=0.0, allow_inf_nan=False)
    weight_kn: float = Field(gt=0.0, allow_inf_nan=False)
    expected_weight_kn: float = Field(gt=0.0, allow_inf_nan=False)
    uniformity_term: float = Field(gt=0.0, allow_inf_nan=False)
    expected_uniformity_term: float = Field(gt=0.0, allow_inf_nan=False)
    stage1_maximum_utilization: float = Field(ge=0.0, allow_inf_nan=False)
    stage2_maximum_utilization: float = Field(ge=0.0, allow_inf_nan=False)
    exact_identity_match: bool
    numerical_match: bool
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class IndependentObjectiveAudit(StrictConfigModel):
    """Record an objective recomputation that does not call production objective functions."""

    structural_weight_n: float = Field(gt=0.0, allow_inf_nan=False)
    expected_structural_weight_n: float = Field(gt=0.0, allow_inf_nan=False)
    uniformity_term: float = Field(ge=0.0, allow_inf_nan=False)
    expected_uniformity_term: float = Field(ge=0.0, allow_inf_nan=False)
    objective: float = Field(gt=0.0, allow_inf_nan=False)
    expected_objective: float = Field(gt=0.0, allow_inf_nan=False)
    maximum_relative_error: float = Field(ge=0.0, allow_inf_nan=False)
    passed: bool


class RefinementPoint(StrictConfigModel):
    """Store compact CP response data for one fixed pushover increment."""

    roof_increment_ratio: float = Field(gt=0.0, allow_inf_nan=False)
    cp_base_shear_n: float = Field(gt=0.0, allow_inf_nan=False)
    cp_story_drift_ratios: tuple[float, ...]
    primary_feasible: bool
    failed_attempt_count: int = Field(ge=0)


class NoFilterOutcome(StrictConfigModel):
    """Store one excluded single-group substitution evaluated without the filter guard."""

    group_id: str = Field(min_length=1)
    section_name: str = Field(min_length=1)
    structural_weight_n: float = Field(gt=0.0, allow_inf_nan=False)
    stage1_feasible: bool
    stage2_executed: bool
    stage2_feasible: bool | None
    maximum_stage1_utilization: float = Field(ge=0.0, allow_inf_nan=False)


class WeakAxisSensitivityResult(StrictConfigModel):
    """Summarize a demand-preserving ry-for-rx column interaction diagnostic."""

    baseline_maximum_utilization: float = Field(ge=0.0, allow_inf_nan=False)
    weak_axis_maximum_utilization: float = Field(ge=0.0, allow_inf_nan=False)
    baseline_failed_count: int = Field(ge=0)
    weak_axis_failed_count: int = Field(ge=0)
    controlling_check_id: str = Field(min_length=1)
    acceptance_role: Literal["diagnostic_only"] = "diagnostic_only"


class GravitySensitivityResult(StrictConfigModel):
    """Summarize the predeclared alternative nonlinear gravity run."""

    label: str = Field(min_length=1)
    baseline_cp_base_shear_n: float = Field(gt=0.0, allow_inf_nan=False)
    sensitivity_cp_base_shear_n: float = Field(gt=0.0, allow_inf_nan=False)
    relative_base_shear_change: float = Field(allow_inf_nan=False)
    baseline_uniformity_term: float = Field(ge=0.0, allow_inf_nan=False)
    sensitivity_uniformity_term: float = Field(ge=0.0, allow_inf_nan=False)
    baseline_primary_feasible: bool
    sensitivity_primary_feasible: bool
    acceptance_role: Literal["diagnostic_only"] = "diagnostic_only"


class PublishedComparison(StrictConfigModel):
    """Compare the project champion with one historical optimizer row."""

    algorithm: str
    published_objective: float
    project_objective: float
    objective_relative_difference: float
    published_weight_kn: float
    project_weight_kn: float
    weight_relative_difference: float
    published_uniformity_term: float
    project_uniformity_term: float
    uniformity_relative_difference: float
    disposition: Literal["research_diagnostic_only"] = "research_diagnostic_only"


class TargetValidation(StrictConfigModel):
    """Store the frozen best design's response and governing utilization at one target."""

    target_id: str = Field(min_length=1)
    base_shear_n: float = Field(gt=0.0, allow_inf_nan=False)
    story_drift_ratios: tuple[float, ...]
    maximum_constraint_utilization: float = Field(ge=0.0, allow_inf_nan=False)


class PublishedCaseAudit(StrictConfigModel):
    """Summarize one preserved REF-A GWO design evaluation from Phase 7."""

    design_id: str = Field(min_length=1)
    sections_by_group: dict[str, str]
    project_primary_feasible: bool
    failed_primary_constraint_count: int = Field(ge=0)
    cp_base_shear_n: float = Field(gt=0.0, allow_inf_nan=False)
    cp_story_drift_ratios: tuple[float, ...]
    maximum_table5_difference_percentage_points: float = Field(ge=0.0, allow_inf_nan=False)
    disposition: Literal["research_diagnostic_only"] = "research_diagnostic_only"


class Phase10Artifact(StrictConfigModel):
    """Persist the complete independent validation and model-freeze record."""

    schema_version: Literal["1.0"] = "1.0"
    validation_id: str
    phase9_summary_sha256: str
    phase9_completion_commit: str
    evaluator_identity: dict[str, str]
    fresh_champion_validations: tuple[FreshChampionValidation, ...]
    all_fresh_champions_match: bool
    independent_objective_audit: IndependentObjectiveAudit
    refinement_points: tuple[RefinementPoint, ...]
    maximum_refinement_relative_spread: float
    refinement_passed: bool
    no_filter_outcomes: tuple[NoFilterOutcome, ...]
    no_filter_status_counts: dict[str, int]
    no_filter_stage1_feasible_count: int
    no_filter_stage2_feasible_count: int
    gravity_sensitivity: GravitySensitivityResult
    weak_axis_sensitivity: WeakAxisSensitivityResult
    best_design_target_results: tuple[TargetValidation, ...]
    published_gwo_case_audits: tuple[PublishedCaseAudit, ...]
    published_case2_comparisons: tuple[PublishedComparison, ...]
    prior_published_case_artifact_sha256: str
    model_freeze_inputs: dict[str, str]
    review_status: Literal["not independently engineering-validated"]
    limitations: tuple[str, ...]


def _relative_error(actual: float, expected: float) -> float:
    """Return scale-normalized absolute error."""

    return abs(actual - expected) / max(abs(expected), 1e-30)


def _load_solver_policy(path: Path) -> NonlinearSolverPolicy:
    """Load the nested nonlinear solver policy without using evaluator internals."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("solver_policy"), dict):
        raise ValueError("nonlinear policy must contain solver_policy")
    return NonlinearSolverPolicy.model_validate(raw["solver_policy"])


def _fresh_champion_worker(
    project_root: str, seed_record: dict[str, object], tolerance: float
) -> dict[str, object]:
    """Evaluate one champion in a newly spawned process with the cache bypassed."""

    root = Path(project_root)
    evaluator = StructuralEvaluator(EvaluatorPaths.from_project_root(root))
    sections_raw = seed_record["champion_sections_by_group"]
    objective_raw = seed_record["champion_objective"]
    if not isinstance(sections_raw, dict) or not isinstance(objective_raw, dict):
        raise TypeError("Phase 9 seed record has invalid champion fields")
    sections = {str(group): str(name) for group, name in sections_raw.items()}
    execution = evaluator.evaluate(
        evaluation_id=f"phase10-seed-{seed_record['seed']}",
        section_names_by_group=sections,
        use_cache=False,
    )
    result = execution.result
    if result.objective is None or result.structural_weight_n is None:
        raise RuntimeError(f"fresh champion seed {seed_record['seed']} has no objective")
    weight_value = seed_record["champion_weight_kn"]
    seed_value = seed_record["seed"]
    if (
        not isinstance(weight_value, (int, float))
        or isinstance(weight_value, bool)
        or not isinstance(seed_value, int)
        or isinstance(seed_value, bool)
    ):
        raise TypeError("Phase 9 seed and weight fields must be numeric")
    if result.stage1.maximum_utilization is None or result.stage2.maximum_utilization is None:
        raise RuntimeError("fresh champion has incomplete structural summaries")
    expected_objective = float(objective_raw["raw_objective"])
    expected_weight_kn = float(weight_value)
    expected_uniformity = float(objective_raw["uniformity_term"])
    expected_cache_key = str(seed_record["champion_cache_key"])
    numerical_match = (
        max(
            _relative_error(result.objective.raw_objective, expected_objective),
            _relative_error(result.structural_weight_n / 1000.0, expected_weight_kn),
            _relative_error(result.objective.uniformity_term, expected_uniformity),
        )
        <= tolerance
    )
    return FreshChampionValidation(
        seed=seed_value,
        sections_by_group=sections,
        status=result.status,
        cache_key=result.cache_key,
        expected_cache_key=expected_cache_key,
        objective=result.objective.raw_objective,
        expected_objective=expected_objective,
        weight_kn=result.structural_weight_n / 1000.0,
        expected_weight_kn=expected_weight_kn,
        uniformity_term=result.objective.uniformity_term,
        expected_uniformity_term=expected_uniformity,
        stage1_maximum_utilization=result.stage1.maximum_utilization,
        stage2_maximum_utilization=result.stage2.maximum_utilization,
        exact_identity_match=result.cache_key == expected_cache_key,
        numerical_match=numerical_match and result.status == seed_record["champion_status"],
        result_sha256=sha256_config(result),
    ).model_dump(mode="json")


def _independent_objective(
    evaluator: StructuralEvaluator, best: dict[str, object], tolerance: float
) -> IndependentObjectiveAudit:
    """Recompute member weight and Eq. (20) directly from primitive data."""

    sections_raw = best["sections_by_group"]
    objective_raw = best["objective"]
    displacements_raw = best["cp_story_displacements_m"]
    if not isinstance(sections_raw, dict) or not isinstance(objective_raw, dict):
        raise TypeError("best champion fields are invalid")
    if not isinstance(displacements_raw, list):
        raise TypeError("best champion CP displacements are invalid")
    nodes = {node.node_id: node for node in evaluator.generated.topology.nodes}
    length_by_group: defaultdict[str, float] = defaultdict(float)
    for member in evaluator.generated.topology.members:
        i_node = nodes[member.i_node]
        j_node = nodes[member.j_node]
        length_by_group[member.group_id] += math.hypot(
            j_node.x_m - i_node.x_m, j_node.y_m - i_node.y_m
        )
    weight = sum(
        length_by_group[group] * evaluator.repository.get(str(name)).weight_n_per_m
        for group, name in sections_raw.items()
    )
    elevations: list[float] = []
    running_elevation = 0.0
    for height in evaluator.generated.definition.story_heights_m:
        running_elevation += height
        elevations.append(running_elevation)
    displacements = tuple(float(value) for value in displacements_raw)
    roof = displacements[-1]
    total_height = elevations[-1]
    ns = len(displacements)
    uniformity = math.sqrt(
        sum(
            ((displacements[index] / roof) * (total_height / elevations[index]) - 1.0) ** 2
            for index in range(ns - 1)
        )
        / ns
    )
    maximum_weight = float(objective_raw["maximum_structural_weight_n"])
    objective = 0.95 * weight / maximum_weight + 0.05 * uniformity
    expected_weight = float(objective_raw["structural_weight_n"])
    expected_uniformity = float(objective_raw["uniformity_term"])
    expected_objective = float(objective_raw["raw_objective"])
    maximum_error = max(
        _relative_error(weight, expected_weight),
        _relative_error(uniformity, expected_uniformity),
        _relative_error(objective, expected_objective),
    )
    return IndependentObjectiveAudit(
        structural_weight_n=weight,
        expected_structural_weight_n=expected_weight,
        uniformity_term=uniformity,
        expected_uniformity_term=expected_uniformity,
        objective=objective,
        expected_objective=expected_objective,
        maximum_relative_error=maximum_error,
        passed=maximum_error <= tolerance,
    )


def _refinement_point(result: DesignPushoverResult) -> RefinementPoint:
    """Extract a compact CP convergence point."""

    cp = next(state for state in result.final.target_states if state.target.target_id == "CP")
    return RefinementPoint(
        roof_increment_ratio=result.roof_increment_ratio,
        cp_base_shear_n=cp.base_shear_n,
        cp_story_drift_ratios=cp.story_drift_ratios,
        primary_feasible=result.primary_feasible,
        failed_attempt_count=sum(
            attempt.return_code != 0
            for segment in result.final.displacement_segments
            for attempt in segment.attempts
        ),
    )


def _maximum_refinement_spread(points: tuple[RefinementPoint, ...]) -> float:
    """Return the largest coarse-to-fine response spread."""

    series = [tuple(point.cp_base_shear_n for point in points)]
    series.extend(
        tuple(point.cp_story_drift_ratios[index] for point in points)
        for index in range(len(points[-1].cp_story_drift_ratios))
    )
    return max((max(values) - min(values)) / max(abs(values[-1]), 1e-30) for values in series)


def _no_filter_study(evaluator: StructuralEvaluator) -> tuple[NoFilterOutcome, ...]:
    """Evaluate every excluded one-group substitution without the evaluator filter guard."""

    best_sections = {"1": "W18X60", "2": "W14X53", "3": "W21X44", "4": "W18X46", "5": "W16X40"}
    generated = evaluator.generated
    outcomes: list[NoFilterOutcome] = []
    for group in evaluator.filter_audit.groups:
        for rejected in group.rejected_sections:
            names = dict(best_sections)
            names[group.group_id] = rejected.section_name
            sections = {key: evaluator.repository.get(value) for key, value in names.items()}
            case = evaluator.base_stage1_case.model_copy(
                update={
                    "case_id": f"phase10-no-filter-g{group.group_id}-{rejected.section_name}",
                    "sections_by_group": names,
                }
            )
            stage1 = evaluate_stage1_design(
                generated=generated,
                sections_by_group=sections,
                case=case,
                policy=evaluator.stage1_policy,
                reference=evaluator.stage1_reference,
            )
            stage2_feasible: bool | None = None
            if stage1.feasible:
                stage2_feasible = evaluate_design_pushover(
                    generated,
                    sections,
                    design_id=case.case_id,
                    policy=evaluator.pushover_policy,
                    solver_policy=evaluator.solver_policy,
                    roof_increment_ratio=evaluator.pushover_policy.baseline_roof_increment_ratio,
                ).primary_feasible
            weight = sum(
                evaluator.group_lengths_m[key] * section.weight_n_per_m
                for key, section in sections.items()
            )
            outcomes.append(
                NoFilterOutcome(
                    group_id=group.group_id,
                    section_name=rejected.section_name,
                    structural_weight_n=weight,
                    stage1_feasible=stage1.feasible,
                    stage2_executed=stage1.feasible,
                    stage2_feasible=stage2_feasible,
                    maximum_stage1_utilization=max(
                        constraint.utilization for constraint in stage1.constraints
                    ),
                )
            )
    return tuple(outcomes)


def _weak_axis_sensitivity(evaluator: StructuralEvaluator) -> WeakAxisSensitivityResult:
    """Recompute champion column interactions with ry while preserving analyzed demands."""

    names = {"1": "W18X60", "2": "W14X53", "3": "W21X44", "4": "W18X46", "5": "W16X40"}
    sections = {group: evaluator.repository.get(name) for group, name in names.items()}
    case = evaluator.base_stage1_case.model_copy(
        update={"case_id": "phase10-weak-axis-diagnostic", "sections_by_group": names}
    )
    stage1 = evaluate_stage1_design(
        generated=evaluator.generated,
        sections_by_group=sections,
        case=case,
        policy=evaluator.stage1_policy,
        reference=evaluator.stage1_reference,
    )
    effective_by_member = {item.member_id: item for item in stage1.effective_lengths}
    column_material = next(
        material
        for material in evaluator.generated.definition.materials
        if material.member_type == "column"
    )
    baseline: list[float] = []
    weak: list[tuple[str, float]] = []
    for constraint in stage1.constraints:
        if constraint.check_type != "column_interaction":
            continue
        section = sections[constraint.group_id]
        effective = effective_by_member[constraint.member_id]
        compression = nominal_compression_strength_with_local_buckling(
            section,
            effective_length_factor=effective.effective_length_factor,
            length_m=effective.member_length_m,
            radius_of_gyration_m=section.ry_m,
            yield_strength_pa=column_material.nominal_yield_strength_pa,
            elastic_modulus_pa=column_material.elastic_modulus_pa,
        )
        utilization = beam_column_interaction_utilization(
            compression_demand_n=float(constraint.details["compression_demand_n"]),
            moment_demand_nm=float(constraint.details["moment_demand_nm"]),
            nominal_compression_strength_n_value=compression.nominal_strength_n,
            nominal_flexural_strength_nm=float(constraint.details["nominal_flexural_strength_nm"]),
            compression_resistance_factor=evaluator.stage1_policy.compression_resistance_factor,
            flexural_resistance_factor=evaluator.stage1_policy.flexural_resistance_factor,
        )
        baseline.append(constraint.utilization)
        weak.append((constraint.check_id, utilization))
    controlling_id, weak_maximum = max(weak, key=lambda item: item[1])
    return WeakAxisSensitivityResult(
        baseline_maximum_utilization=max(baseline),
        weak_axis_maximum_utilization=weak_maximum,
        baseline_failed_count=sum(value > 1.0 for value in baseline),
        weak_axis_failed_count=sum(value > 1.0 for _, value in weak),
        controlling_check_id=controlling_id,
    )


def _gravity_sensitivity(
    evaluator: StructuralEvaluator,
    baseline: DesignPushoverResult,
    policy: GravitySensitivityPolicy,
) -> GravitySensitivityResult:
    """Run the predeclared alternative nonlinear gravity combination."""

    names = {"1": "W18X60", "2": "W14X53", "3": "W21X44", "4": "W18X46", "5": "W16X40"}
    sections = {group: evaluator.repository.get(name) for group, name in names.items()}
    sensitivity_policy = evaluator.pushover_policy.model_copy(
        update={
            "gravity_dead_factor": policy.dead_factor,
            "gravity_live_factor": policy.live_factor,
        }
    )
    sensitivity = evaluate_design_pushover(
        evaluator.generated,
        sections,
        design_id="phase10-gravity-sensitivity",
        policy=sensitivity_policy,
        solver_policy=evaluator.solver_policy,
        roof_increment_ratio=evaluator.pushover_policy.baseline_roof_increment_ratio,
    )
    baseline_cp = next(
        state for state in baseline.final.target_states if state.target.target_id == "CP"
    )
    sensitivity_cp = next(
        state for state in sensitivity.final.target_states if state.target.target_id == "CP"
    )
    heights: list[float] = []
    elevation = 0.0
    for height in evaluator.generated.definition.story_heights_m:
        elevation += height
        heights.append(elevation)
    baseline_uniformity = drift_uniformity_term(
        floor_displacements_m=baseline_cp.story_displacements_m,
        floor_elevations_m=heights,
    )
    sensitivity_uniformity = drift_uniformity_term(
        floor_displacements_m=sensitivity_cp.story_displacements_m,
        floor_elevations_m=heights,
    )
    return GravitySensitivityResult(
        label=policy.label,
        baseline_cp_base_shear_n=baseline_cp.base_shear_n,
        sensitivity_cp_base_shear_n=sensitivity_cp.base_shear_n,
        relative_base_shear_change=(sensitivity_cp.base_shear_n - baseline_cp.base_shear_n)
        / baseline_cp.base_shear_n,
        baseline_uniformity_term=baseline_uniformity,
        sensitivity_uniformity_term=sensitivity_uniformity,
        baseline_primary_feasible=baseline.primary_feasible,
        sensitivity_primary_feasible=sensitivity.primary_feasible,
    )


def _target_validations(result: DesignPushoverResult) -> tuple[TargetValidation, ...]:
    """Extract every target response and its governing constraint utilization."""

    return tuple(
        TargetValidation(
            target_id=state.target.target_id,
            base_shear_n=state.base_shear_n,
            story_drift_ratios=state.story_drift_ratios,
            maximum_constraint_utilization=max(
                constraint.utilization
                for constraint in result.constraints
                if constraint.target_id == state.target.target_id
            ),
        )
        for state in result.final.target_states
    )


def _published_case_audits(phase7_path: Path) -> tuple[PublishedCaseAudit, ...]:
    """Audit all three preserved GWO cases and REF-A Table 5 comparisons."""

    artifact = json.loads(phase7_path.read_text(encoding="utf-8"))
    comparisons = artifact["published_drift_comparisons"]
    results: list[PublishedCaseAudit] = []
    for design in artifact["baseline_design_results"]:
        design_id = str(design["design_id"])
        cp = next(
            state
            for state in design["final"]["target_states"]
            if state["target"]["target_id"] == "CP"
        )
        maximum_difference = max(
            abs(float(value))
            for comparison in comparisons
            if comparison["design_id"] == design_id
            for value in comparison["signed_differences_percentage_points"]
        )
        results.append(
            PublishedCaseAudit(
                design_id=design_id,
                sections_by_group={
                    str(group): str(name)
                    for group, name in design["section_names_by_group"].items()
                },
                project_primary_feasible=bool(design["primary_feasible"]),
                failed_primary_constraint_count=len(design["failed_primary_constraint_ids"]),
                cp_base_shear_n=float(cp["base_shear_n"]),
                cp_story_drift_ratios=tuple(float(value) for value in cp["story_drift_ratios"]),
                maximum_table5_difference_percentage_points=maximum_difference,
            )
        )
    return tuple(results)


def build_phase10_artifact(project_root: Path, policy_path: Path) -> Phase10Artifact:
    """Execute the predeclared validation matrix and return a model-freeze artifact."""

    root = project_root.resolve()
    policy = Phase10Policy.from_json(policy_path)
    summary_path = root / policy.phase9_summary
    if sha256_file(summary_path) != policy.phase9_summary_sha256:
        raise RuntimeError("Phase 9 summary hash differs from the frozen Phase 10 policy")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seed_records = summary.get("seed_records")
    best = summary.get("best_feasible_design")
    if not isinstance(seed_records, list) or len(seed_records) != 30 or not isinstance(best, dict):
        raise ValueError("Phase 9 summary must contain 30 seed records and one best design")
    evaluator = StructuralEvaluator(EvaluatorPaths.from_project_root(root))
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=4, mp_context=context) as executor:
        fresh_raw = tuple(
            executor.map(
                _fresh_champion_worker,
                (str(root) for _ in seed_records),
                seed_records,
                (policy.champion_relative_tolerance for _ in seed_records),
            )
        )
    fresh = tuple(FreshChampionValidation.model_validate(item) for item in fresh_raw)
    independent = _independent_objective(evaluator, best, policy.objective_relative_tolerance)

    best_sections_raw = best["sections_by_group"]
    if not isinstance(best_sections_raw, dict):
        raise TypeError("best champion sections are invalid")
    best_sections = {
        str(group): evaluator.repository.get(str(name)) for group, name in best_sections_raw.items()
    }
    refinements = tuple(
        evaluate_design_pushover(
            evaluator.generated,
            best_sections,
            design_id="phase10-best-champion",
            policy=evaluator.pushover_policy,
            solver_policy=evaluator.solver_policy,
            roof_increment_ratio=ratio,
        )
        for ratio in policy.pushover_refinement_ratios
    )
    refinement_points = tuple(_refinement_point(result) for result in refinements)
    maximum_spread = _maximum_refinement_spread(refinement_points)
    no_filter = _no_filter_study(evaluator)
    no_filter_counts = Counter(
        "stage1_infeasible"
        if not item.stage1_feasible
        else "feasible"
        if item.stage2_feasible
        else "stage2_infeasible"
        for item in no_filter
    )
    published_comparisons = tuple(
        PublishedComparison(
            algorithm=reference.algorithm,
            published_objective=reference.objective,
            project_objective=float(best["objective"]["raw_objective"]),
            objective_relative_difference=(
                float(best["objective"]["raw_objective"]) - reference.objective
            )
            / reference.objective,
            published_weight_kn=reference.weight_kn,
            project_weight_kn=float(best["weight_kn"]),
            weight_relative_difference=(float(best["weight_kn"]) - reference.weight_kn)
            / reference.weight_kn,
            published_uniformity_term=reference.uniformity_term,
            project_uniformity_term=float(best["objective"]["uniformity_term"]),
            uniformity_relative_difference=(
                float(best["objective"]["uniformity_term"]) - reference.uniformity_term
            )
            / reference.uniformity_term,
        )
        for reference in policy.published_case2_references
    )
    paths = EvaluatorPaths.from_project_root(root)
    freeze_inputs = {
        name: sha256_file(path)
        for name, path in paths.model_dump().items()
        if isinstance(path, Path)
    }
    freeze_inputs["phase10_policy"] = sha256_file(policy_path)
    phase7_path = root / "docs" / "evidence" / "phase07" / "three_story_pushover.json"
    return Phase10Artifact(
        validation_id=policy.validation_id,
        phase9_summary_sha256=policy.phase9_summary_sha256,
        phase9_completion_commit=policy.phase9_completion_commit,
        evaluator_identity=evaluator.identity.model_dump(),
        fresh_champion_validations=fresh,
        all_fresh_champions_match=all(
            item.exact_identity_match and item.numerical_match for item in fresh
        ),
        independent_objective_audit=independent,
        refinement_points=refinement_points,
        maximum_refinement_relative_spread=maximum_spread,
        refinement_passed=maximum_spread <= policy.pushover_refinement_relative_tolerance,
        no_filter_outcomes=no_filter,
        no_filter_status_counts=dict(sorted(no_filter_counts.items())),
        no_filter_stage1_feasible_count=sum(item.stage1_feasible for item in no_filter),
        no_filter_stage2_feasible_count=sum(item.stage2_feasible is True for item in no_filter),
        gravity_sensitivity=_gravity_sensitivity(
            evaluator, refinements[-1], policy.gravity_sensitivity
        ),
        weak_axis_sensitivity=_weak_axis_sensitivity(evaluator),
        best_design_target_results=_target_validations(refinements[-1]),
        published_gwo_case_audits=_published_case_audits(phase7_path),
        published_case2_comparisons=published_comparisons,
        prior_published_case_artifact_sha256=sha256_file(phase7_path),
        model_freeze_inputs=freeze_inputs,
        review_status="not independently engineering-validated",
        limitations=(
            "The no-filter study is an exhaustive one-coordinate ablation around the best "
            "champion, not a second stochastic campaign.",
            "REF-A Fig. 7 has no numerical ordinates, so pushover curve comparison remains "
            "qualitative plus target-state metrics.",
            "Sensitivity variants diagnose assumption dependence and are not alternate "
            "certified designs.",
        ),
    )


def main() -> int:
    """Write canonical Phase 10 evidence from an explicit project root and policy."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    artifact = build_phase10_artifact(arguments.project_root, arguments.policy)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(canonical_json_dumps(artifact) + "\n", encoding="utf-8")
    print(
        canonical_json_dumps(
            {"output": str(arguments.output), "sha256": sha256_file(arguments.output)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
