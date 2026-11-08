"""Compose Stage 1 and Stage 2 into a deterministic cacheable evaluation pipeline."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from steel_frame_optimization.analysis.convergence import NonlinearSolverPolicy
from steel_frame_optimization.analysis.pushover import (
    DesignPushoverResult,
    PushoverPolicy,
    evaluate_design_pushover,
)
from steel_frame_optimization.config.schemas import StrictConfigModel
from steel_frame_optimization.constraints.fema import PerformanceConstraint
from steel_frame_optimization.models.failures import FailureCategory, FailureRecord
from steel_frame_optimization.models.frame import FrameDefinition, GeneratedFrame, generate_frame
from steel_frame_optimization.objective.functions import (
    ObjectiveComponents,
    calculate_objective,
    drift_uniformity_term,
    structural_weight_n,
)
from steel_frame_optimization.objective.penalties import (
    NormalizedViolation,
    PenaltyPolicy,
    PenaltyScore,
    ViolationCategory,
    calculate_penalty_score,
    normalized_violation,
)
from steel_frame_optimization.sections.filtering import GroupFilterAudit
from steel_frame_optimization.sections.records import SectionRecord
from steel_frame_optimization.sections.repository import SectionRepository, SectionRepositoryError
from steel_frame_optimization.serialization import canonical_json_dumps, sha256_config, sha256_file
from steel_frame_optimization.validation.phase5 import (
    ConstraintResult,
    PublishedStage1Reference,
    Stage1Artifact,
    Stage1Case,
    Stage1Policy,
    evaluate_stage1_design,
)

EvaluationStatus = Literal[
    "feasible",
    "stage1_infeasible",
    "stage2_infeasible",
    "structural_failure",
    "invalid_input",
    "numerical_failure",
]


class ObjectivePolicy(StrictConfigModel):
    """Freeze Eq. (20), project Wmax, and the source-visible discrepancy."""

    source: Literal["REF-A Eq. (20), PDF p. 4 / journal p. 80"]
    weight_coefficient: float = Field(gt=0.0, allow_inf_nan=False)
    uniformity_coefficient: float = Field(gt=0.0, allow_inf_nan=False)
    maximum_structural_weight_n: float = Field(gt=0.0, allow_inf_nan=False)
    wmax_basis: str = Field(min_length=1)
    wmax_research_reference_n: float = Field(gt=0.0, allow_inf_nan=False)
    uniformity_sum: Literal["levels 1 through ns-1; divisor remains ns"]
    assumption_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_paper_coefficients(self) -> ObjectivePolicy:
        """Require the exact 0.95/0.05 coefficients printed by REF-A."""

        if (self.weight_coefficient, self.uniformity_coefficient) != (0.95, 0.05):
            raise ValueError("objective coefficients differ from REF-A Eq. (20)")
        return self


class EvaluationPolicy(StrictConfigModel):
    """Load the complete Phase 8 objective and penalty policy."""

    schema_version: Literal["1.0"] = "1.0"
    objective: ObjectivePolicy
    penalty: PenaltyPolicy
    stage1_constraint_scope: Literal["full", "gravity_strength_and_service"] = "full"

    @classmethod
    def from_json(cls, path: Path) -> EvaluationPolicy:
        """Load one strict Phase 8 policy file."""

        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


class EvaluatorPaths(StrictConfigModel):
    """Declare every persisted input that participates in evaluator identity."""

    frame: Path
    sections: Path
    preliminary_filter_audit: Path
    stage1_case: Path
    stage1_policy: Path
    stage1_reference: Path
    nonlinear_solver_policy: Path
    pushover_policy: Path
    evaluation_policy: Path
    failure_taxonomy: Path

    @classmethod
    def from_project_root(cls, root: Path) -> EvaluatorPaths:
        """Return the explicit three-story Phase 8 path set under one project root."""

        return cls(
            frame=root / "configs" / "frames" / "three_story_frame.json",
            sections=root / "data" / "processed" / "aisc_v14_1_w_shapes.json",
            preliminary_filter_audit=(
                root / "docs" / "evidence" / "phase05" / "preliminary_filter_audit.json"
            ),
            stage1_case=root / "configs" / "cases" / "three_story_case2_gwo.json",
            stage1_policy=root / "configs" / "analysis" / "stage1_policy.json",
            stage1_reference=(
                root / "configs" / "cases" / "three_story_case2_stage1_reference.json"
            ),
            nonlinear_solver_policy=(
                root / "configs" / "analysis" / "phase6_nonlinear_policy.json"
            ),
            pushover_policy=root / "configs" / "analysis" / "phase7_pushover_policy.json",
            evaluation_policy=(
                root / "configs" / "analysis" / "phase8_objective_penalty_policy.json"
            ),
            failure_taxonomy=root / "configs" / "analysis" / "failure_taxonomy.json",
        )

    @classmethod
    def from_profile(cls, root: Path, profile: str) -> EvaluatorPaths:
        """Return the immutable input set for one supported structural-evaluation profile."""

        if profile == "three_story_case2":
            return cls.from_project_root(root)
        profiles = {
            "nine_story_case1_filtered": (
                "nine_story_case1_gwo.json",
                "phase11_artifact.json",
                "phase12_case1_objective_penalty_policy.json",
            ),
            "nine_story_case1_no_filter": (
                "nine_story_case1_gwo.json",
                "phase12_nine_story_no_filter.json",
                "phase12_case1_objective_penalty_policy.json",
            ),
            "nine_story_case2": (
                "nine_story_case2_gwo.json",
                "phase11_artifact.json",
                "phase12_case2_objective_penalty_policy.json",
            ),
        }
        try:
            case_name, filter_name, evaluation_name = profiles[profile]
        except KeyError as error:
            raise ValueError(f"unknown evaluator profile: {profile}") from error
        filter_path = root / "docs" / "evidence" / "phase11" / filter_name
        if profile == "nine_story_case1_no_filter":
            filter_path = root / "configs" / "analysis" / filter_name
        return cls(
            frame=root / "configs" / "frames" / "nine_story_frame.json",
            sections=root / "data" / "processed" / "aisc_v14_1_w_shapes.json",
            preliminary_filter_audit=filter_path,
            stage1_case=root / "configs" / "cases" / case_name,
            stage1_policy=root / "configs" / "analysis" / "stage1_policy.json",
            stage1_reference=(
                root / "configs" / "cases" / "nine_story_case2_stage1_reference.json"
            ),
            nonlinear_solver_policy=(
                root / "configs" / "analysis" / "phase6_nonlinear_policy.json"
            ),
            pushover_policy=root / "configs" / "analysis" / "phase7_pushover_policy.json",
            evaluation_policy=root / "configs" / "analysis" / evaluation_name,
            failure_taxonomy=root / "configs" / "analysis" / "failure_taxonomy.json",
        )


class OperationalFilterAudit(StrictConfigModel):
    """Expose only the group decisions required by optimization-domain construction."""

    groups: tuple[GroupFilterAudit, ...] = Field(min_length=1)


def _load_operational_filter(path: Path) -> OperationalFilterAudit:
    """Load either a direct filter record or the nested Phase 11 canonical artifact."""

    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError("preliminary filter input must be a JSON object")
    source_path = content.get("base_filter_path")
    if source_path is not None:
        if not isinstance(source_path, str):
            raise ValueError("base_filter_path must be a string")
        source = (path.parents[2] / source_path).resolve()
        expected_hash = content.get("base_filter_sha256")
        if not isinstance(expected_hash, str) or sha256_file(source) != expected_hash:
            raise ValueError("no-filter source identity does not match its declared SHA-256")
        base = _load_operational_filter(source)
        return OperationalFilterAudit(
            groups=tuple(
                group.model_copy(update={"actual_removed_count": 0, "rejected_sections": ()})
                for group in base.groups
            )
        )
    nested = content.get("preliminary_filter", content)
    if not isinstance(nested, dict) or not isinstance(nested.get("groups"), list):
        raise ValueError("preliminary filter input must contain groups")
    return OperationalFilterAudit.model_validate({"groups": nested["groups"]})


def _apply_stage1_constraint_scope(
    artifact: Stage1Artifact, scope: Literal["full", "gravity_strength_and_service"]
) -> Stage1Artifact:
    """Apply the source-defined Case 1 constraint subset after one common analysis path."""

    if scope == "full":
        return artifact
    constraints = tuple(
        item
        for item in artifact.constraints
        if item.load_case.startswith("service_") or item.load_case == "strength_gravity"
    )
    failed = tuple(item.check_id for item in constraints if not item.passed)
    return artifact.model_copy(
        update={
            "constraints": constraints,
            "feasible": not failed,
            "failed_constraint_ids": failed,
            "maximum_utilization": max(item.utilization for item in constraints),
        }
    )


class EvaluatorIdentity(StrictConfigModel):
    """Persist complete model/configuration/code identities used by every cache key."""

    frame_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sections_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preliminary_filter_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage1_case_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage1_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage1_reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    nonlinear_solver_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pushover_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    failure_taxonomy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    implementation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class StageEvaluationSummary(StrictConfigModel):
    """Retain compact stage disposition and a hash of the complete transient result."""

    stage: Literal["stage1", "stage2"]
    executed: bool
    feasible: bool | None
    constraint_count: int = Field(ge=0)
    failed_constraint_count: int = Field(ge=0)
    maximum_utilization: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    complete_result_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_result_fields_when_executed(self) -> StageEvaluationSummary:
        """Require result identity and feasibility exactly when a stage ran."""

        result_fields = (self.feasible, self.maximum_utilization, self.complete_result_sha256)
        if self.executed and any(value is None for value in result_fields):
            raise ValueError("executed stage summary requires feasibility, utilization, and hash")
        if not self.executed and any(value is not None for value in result_fields):
            raise ValueError("unexecuted stage summary cannot contain structural result fields")
        if not self.executed and (self.constraint_count or self.failed_constraint_count):
            raise ValueError("unexecuted stage summary cannot contain constraint counts")
        return self


class EvaluationResult(StrictConfigModel):
    """Provide the compact deterministic contract consumed by the future GA."""

    schema_version: Literal["1.0"] = "1.0"
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_identity: EvaluatorIdentity
    section_names_by_group: dict[str, str]
    section_source_rows_by_group: dict[str, int]
    status: EvaluationStatus
    rank_class: Literal[0, 1, 2]
    structural_weight_n: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    weight_contribution: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    objective: ObjectiveComponents | None = None
    cp_story_displacements_m: tuple[float, ...] | None = None
    stage1: StageEvaluationSummary
    stage2: StageEvaluationSummary
    violations: tuple[NormalizedViolation, ...]
    penalty: PenaltyScore | None = None
    failure: FailureRecord | None = None

    @model_validator(mode="after")
    def validate_disposition(self) -> EvaluationResult:
        """Prevent failed or incomplete evaluations from masquerading as feasible scores."""

        if self.status == "feasible":
            if self.rank_class != 0 or self.objective is None or self.penalty is None:
                raise ValueError("feasible result requires class 0, objective, and score")
            if not self.penalty.feasible or self.failure is not None:
                raise ValueError("feasible result cannot contain failure or infeasible score")
            if self.stage1.feasible is not True or self.stage2.feasible is not True:
                raise ValueError("feasible result requires both structural stages to be feasible")
        elif self.status in {"stage1_infeasible", "stage2_infeasible"}:
            if self.rank_class != 1 or self.penalty is None or self.penalty.feasible:
                raise ValueError("structural infeasibility requires class 1 infeasible score")
            if self.failure is not None:
                raise ValueError("structural constraint failure is represented by violations")
            if self.status == "stage1_infeasible":
                if self.stage1.feasible is not False or self.stage2.executed:
                    raise ValueError("Stage 1 infeasibility must skip Stage 2")
                if (
                    self.objective is not None
                    or self.penalty.objective_basis != "weight_contribution_only"
                ):
                    raise ValueError("Stage 1 infeasibility cannot report the CP objective")
            elif self.stage1.feasible is not True or self.stage2.feasible is not False:
                raise ValueError(
                    "Stage 2 infeasibility requires feasible Stage 1 and failed Stage 2"
                )
        else:
            if self.rank_class != 2 or self.penalty is not None or self.failure is None:
                raise ValueError("failed evaluation requires class 2 and no finite score")
            if self.objective is not None:
                raise ValueError("failed evaluation cannot report a structural objective")
        if self.stage2.executed and not self.stage1.executed:
            raise ValueError("Stage 2 cannot execute before Stage 1")
        return self


class EvaluationExecution(StrictConfigModel):
    """Separate deterministic structural results from the operational cache-hit flag."""

    request_id: str = Field(min_length=1)
    result: EvaluationResult
    cache_hit: bool


class EvaluationCache:
    """Store canonical results in explicit instance state with copy-on-read semantics."""

    def __init__(self) -> None:
        """Create an empty cache without process-global state."""

        self._canonical_results: dict[str, str] = {}

    def get(self, key: str) -> EvaluationResult | None:
        """Return a revalidated copy of a cached result, if present."""

        content = self._canonical_results.get(key)
        return None if content is None else EvaluationResult.model_validate_json(content)

    def put(self, result: EvaluationResult) -> None:
        """Persist one canonical result under its verified internal key."""

        content = canonical_json_dumps(result)
        existing = self._canonical_results.get(result.cache_key)
        if existing is not None and existing != content:
            raise RuntimeError("cache key collision produced different evaluation results")
        self._canonical_results[result.cache_key] = content

    def __len__(self) -> int:
        """Return the number of unique cached evaluations."""

        return len(self._canonical_results)


def _implementation_identity() -> str:
    """Hash every package Python module as a conservative executable-code identity."""

    package_root = Path(__file__).resolve().parents[1]
    source_files = tuple(sorted(package_root.rglob("*.py")))
    return sha256_config(
        {path.relative_to(package_root).as_posix(): sha256_file(path) for path in source_files}
    )


def _load_solver_policy(path: Path) -> NonlinearSolverPolicy:
    """Load the nested verified nonlinear solver policy."""

    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or not isinstance(content.get("solver_policy"), dict):
        raise ValueError("Phase 6 policy must contain solver_policy")
    return NonlinearSolverPolicy.model_validate(content["solver_policy"])


def _group_lengths(generated: GeneratedFrame) -> dict[str, float]:
    """Sum deterministic member centerline lengths by design group."""

    nodes = {node.node_id: node for node in generated.topology.nodes}
    lengths: dict[str, float] = defaultdict(float)
    for member in generated.topology.members:
        i_node = nodes[member.i_node]
        j_node = nodes[member.j_node]
        lengths[member.group_id] += math.hypot(j_node.x_m - i_node.x_m, j_node.y_m - i_node.y_m)
    return dict(lengths)


def _stage1_category(constraint: ConstraintResult) -> ViolationCategory:
    """Map one Stage 1 constraint to its configured penalty severity."""

    if constraint.check_type in {"beam_flexure", "column_interaction"}:
        return "strength"
    if constraint.check_type in {"story_drift", "roof_drift"}:
        return "drift"
    return "target_state_shape"


def _stage2_category(constraint: PerformanceConstraint) -> ViolationCategory:
    """Map one Stage 2 constraint to its configured penalty severity."""

    categories: dict[str, ViolationCategory] = {
        "hinge_rotation": "hinge_rotation",
        "force_controlled_column": "force_controlled_action",
        "story_drift": "drift",
        "hierarchy": "capacity_hierarchy",
    }
    return categories[constraint.check_type]


def _violations_from_stage1(
    artifact: Stage1Artifact, policy: PenaltyPolicy
) -> tuple[NormalizedViolation, ...]:
    """Normalize every failed Stage 1 constraint without losing boundary failures."""

    violations = (
        normalized_violation(
            check_id=item.check_id,
            stage="stage1",
            category=_stage1_category(item),
            utilization=item.utilization,
            failed=not item.passed,
            policy=policy,
        )
        for item in artifact.constraints
        if not item.passed
    )
    return tuple(item for item in violations if item is not None)


def _violations_from_stage2(
    result: DesignPushoverResult, policy: PenaltyPolicy
) -> tuple[NormalizedViolation, ...]:
    """Normalize failed primary Stage 2 constraints and exclude named sensitivities."""

    violations = (
        normalized_violation(
            check_id=item.check_id,
            stage="stage2",
            category=_stage2_category(item),
            utilization=item.utilization,
            failed=not item.passed,
            policy=policy,
        )
        for item in result.constraints
        if item.primary_acceptance and not item.passed
    )
    return tuple(item for item in violations if item is not None)


def _stage1_summary(artifact: Stage1Artifact) -> StageEvaluationSummary:
    """Build a compact identity-bearing summary of one complete Stage 1 result."""

    return StageEvaluationSummary(
        stage="stage1",
        executed=True,
        feasible=artifact.feasible,
        constraint_count=len(artifact.constraints),
        failed_constraint_count=len(artifact.failed_constraint_ids),
        maximum_utilization=artifact.maximum_utilization,
        complete_result_sha256=sha256_config(artifact),
    )


def _stage2_summary(result: DesignPushoverResult) -> StageEvaluationSummary:
    """Build a compact identity-bearing summary of one complete Stage 2 result."""

    primary = tuple(item for item in result.constraints if item.primary_acceptance)
    return StageEvaluationSummary(
        stage="stage2",
        executed=True,
        feasible=result.primary_feasible,
        constraint_count=len(primary),
        failed_constraint_count=len(result.failed_primary_constraint_ids),
        maximum_utilization=max(item.utilization for item in primary),
        complete_result_sha256=sha256_config(result),
    )


def _unexecuted_summary(stage: Literal["stage1", "stage2"]) -> StageEvaluationSummary:
    """Return a structurally empty stage summary with no implied result."""

    return StageEvaluationSummary(
        stage=stage,
        executed=False,
        feasible=None,
        constraint_count=0,
        failed_constraint_count=0,
    )


class StructuralEvaluator:
    """Evaluate one supported frame through Stage 1, conditional Stage 2, and scoring."""

    def __init__(self, paths: EvaluatorPaths, cache: EvaluationCache | None = None) -> None:
        """Load immutable inputs, validate Wmax, and retain only explicit instance state."""

        self.paths = paths
        self.cache = cache if cache is not None else EvaluationCache()
        self.generated = generate_frame(FrameDefinition.from_json(paths.frame))
        self.repository = SectionRepository.from_json(paths.sections)
        self.base_stage1_case = Stage1Case.from_json(paths.stage1_case)
        self.stage1_policy = Stage1Policy.from_json(paths.stage1_policy)
        self.stage1_reference = PublishedStage1Reference.from_json(paths.stage1_reference)
        self.solver_policy = _load_solver_policy(paths.nonlinear_solver_policy)
        self.pushover_policy = PushoverPolicy.from_json(paths.pushover_policy)
        self.evaluation_policy = EvaluationPolicy.from_json(paths.evaluation_policy)
        self.filter_audit = _load_operational_filter(paths.preliminary_filter_audit)
        self.group_lengths_m = _group_lengths(self.generated)
        self.rejected_names_by_group = {
            group.group_id: {item.section_name for item in group.rejected_sections}
            for group in self.filter_audit.groups
        }
        member_type_by_group = {
            group.group_id: group.member_type for group in self.filter_audit.groups
        }
        all_w_maximum = max(self.repository.source_order(), key=lambda item: item.weight_n_per_m)
        w14_maximum = max(self.repository.family("w14"), key=lambda item: item.weight_n_per_m)
        calculated_wmax = sum(
            length
            * (
                w14_maximum.weight_n_per_m
                if self.generated.definition.frame_id == "9-story"
                and member_type_by_group[group_id] == "column"
                else all_w_maximum.weight_n_per_m
            )
            for group_id, length in self.group_lengths_m.items()
        )
        configured_wmax = self.evaluation_policy.objective.maximum_structural_weight_n
        if not math.isclose(calculated_wmax, configured_wmax, rel_tol=1e-12, abs_tol=1e-6):
            raise ValueError("configured Wmax does not match approved domain and frame geometry")
        self.identity = EvaluatorIdentity(
            frame_sha256=sha256_file(paths.frame),
            sections_sha256=sha256_file(paths.sections),
            preliminary_filter_sha256=sha256_file(paths.preliminary_filter_audit),
            stage1_case_sha256=sha256_file(paths.stage1_case),
            stage1_policy_sha256=sha256_file(paths.stage1_policy),
            stage1_reference_sha256=sha256_file(paths.stage1_reference),
            nonlinear_solver_policy_sha256=sha256_file(paths.nonlinear_solver_policy),
            pushover_policy_sha256=sha256_file(paths.pushover_policy),
            evaluation_policy_sha256=sha256_file(paths.evaluation_policy),
            failure_taxonomy_sha256=sha256_file(paths.failure_taxonomy),
            implementation_sha256=_implementation_identity(),
        )

    def cache_key(self, section_names_by_group: Mapping[str, str]) -> str:
        """Hash complete input identity and canonical section names for one design."""

        return sha256_config(
            {
                "evaluator_identity": self.identity,
                "section_names_by_group": {
                    group: section_names_by_group[group] for group in sorted(section_names_by_group)
                },
            }
        )

    def _input_failure(
        self,
        *,
        section_names: dict[str, str],
        cache_key: str,
        message: str,
    ) -> EvaluationResult:
        """Build a typed invalid-input result with no structural score."""

        return EvaluationResult(
            cache_key=cache_key,
            evaluator_identity=self.identity,
            section_names_by_group=section_names,
            section_source_rows_by_group={},
            status="invalid_input",
            rank_class=2,
            stage1=_unexecuted_summary("stage1"),
            stage2=_unexecuted_summary("stage2"),
            violations=(),
            failure=FailureRecord(
                category=FailureCategory.INPUT_INVALID,
                stage="evaluation_input",
                message=message,
                recoverable=False,
                details={"cache_key": cache_key},
            ),
        )

    def _typed_failure(
        self,
        *,
        cache_key: str,
        sections: dict[str, SectionRecord],
        structural_weight: float,
        weight_contribution: float,
        stage1: StageEvaluationSummary,
        stage: Literal["stage1", "stage2"],
        status: Literal["structural_failure", "numerical_failure"],
        message: str,
        category: FailureCategory,
    ) -> EvaluationResult:
        """Build a typed structural or numerical failure without a finite ranking score."""

        return EvaluationResult(
            cache_key=cache_key,
            evaluator_identity=self.identity,
            section_names_by_group={group: item.name for group, item in sections.items()},
            section_source_rows_by_group={
                group: item.source_row_id for group, item in sections.items()
            },
            status=status,
            rank_class=2,
            structural_weight_n=structural_weight,
            weight_contribution=weight_contribution,
            stage1=stage1,
            stage2=_unexecuted_summary("stage2"),
            violations=(),
            failure=FailureRecord(
                category=category,
                stage=stage,
                message=message,
                recoverable=False,
                details={"cache_key": cache_key},
            ),
        )

    def evaluate(
        self,
        *,
        evaluation_id: str,
        section_names_by_group: Mapping[str, str],
        use_cache: bool = True,
    ) -> EvaluationExecution:
        """Run deterministic conditional dispatch and return result plus cache disposition."""

        canonical_names = {
            str(group): str(name).upper() for group, name in section_names_by_group.items()
        }
        key = self.cache_key(canonical_names)
        stable_design_id = f"design-{key[:16]}"
        if use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                return EvaluationExecution(request_id=evaluation_id, result=cached, cache_hit=True)
        expected_groups = set(self.generated.definition.design_groups)
        if set(canonical_names) != expected_groups:
            result = self._input_failure(
                section_names=canonical_names,
                cache_key=key,
                message="section assignment must contain exactly the configured design groups",
            )
            self.cache.put(result)
            return EvaluationExecution(request_id=evaluation_id, result=result, cache_hit=False)
        try:
            sections = {group: self.repository.get(name) for group, name in canonical_names.items()}
        except SectionRepositoryError as error:
            result = self._input_failure(
                section_names=canonical_names,
                cache_key=key,
                message=str(error),
            )
            self.cache.put(result)
            return EvaluationExecution(request_id=evaluation_id, result=result, cache_hit=False)
        below_filter = tuple(
            f"{group}:{section.name}"
            for group, section in sections.items()
            if section.name in self.rejected_names_by_group[group]
        )
        if below_filter:
            result = self._input_failure(
                section_names=canonical_names,
                cache_key=key,
                message=f"section assignment violates preliminary domains: {below_filter}",
            )
            self.cache.put(result)
            return EvaluationExecution(request_id=evaluation_id, result=result, cache_hit=False)

        weight = structural_weight_n(
            total_member_length_m_by_group=self.group_lengths_m,
            sections_by_group=sections,
        )
        objective_policy = self.evaluation_policy.objective
        weight_contribution = (
            objective_policy.weight_coefficient
            * weight
            / objective_policy.maximum_structural_weight_n
        )
        stage1_case = self.base_stage1_case.model_copy(
            update={"case_id": stable_design_id, "sections_by_group": canonical_names}
        )
        try:
            stage1_artifact = evaluate_stage1_design(
                generated=self.generated,
                sections_by_group=sections,
                case=stage1_case,
                policy=self.stage1_policy,
                reference=self.stage1_reference,
            )
            stage1_artifact = _apply_stage1_constraint_scope(
                stage1_artifact, self.evaluation_policy.stage1_constraint_scope
            )
        except RuntimeError as error:
            message = str(error)
            if "equilibrium residual" not in message and "direct combination" not in message:
                raise
            result = self._typed_failure(
                cache_key=key,
                sections=sections,
                structural_weight=weight,
                weight_contribution=weight_contribution,
                stage1=_unexecuted_summary("stage1"),
                stage="stage1",
                status="numerical_failure",
                message=message,
                category=FailureCategory.NUMERICAL_INCONSISTENCY,
            )
            self.cache.put(result)
            return EvaluationExecution(request_id=evaluation_id, result=result, cache_hit=False)
        stage1_summary = _stage1_summary(stage1_artifact)
        stage1_violations = _violations_from_stage1(stage1_artifact, self.evaluation_policy.penalty)
        if not stage1_artifact.feasible:
            penalty = calculate_penalty_score(
                objective_base=weight_contribution,
                objective_basis="weight_contribution_only",
                feasible=False,
                violations=stage1_violations,
                policy=self.evaluation_policy.penalty,
            )
            result = EvaluationResult(
                cache_key=key,
                evaluator_identity=self.identity,
                section_names_by_group=canonical_names,
                section_source_rows_by_group={
                    group: item.source_row_id for group, item in sections.items()
                },
                status="stage1_infeasible",
                rank_class=1,
                structural_weight_n=weight,
                weight_contribution=weight_contribution,
                stage1=stage1_summary,
                stage2=_unexecuted_summary("stage2"),
                violations=stage1_violations,
                penalty=penalty,
            )
            self.cache.put(result)
            return EvaluationExecution(request_id=evaluation_id, result=result, cache_hit=False)

        try:
            stage2_result = evaluate_design_pushover(
                self.generated,
                sections,
                design_id=stable_design_id,
                policy=self.pushover_policy,
                solver_policy=self.solver_policy,
                roof_increment_ratio=self.pushover_policy.baseline_roof_increment_ratio,
            )
        except RuntimeError as error:
            message = str(error)
            if not (
                message.startswith("preliminary pushover failed:")
                or message.startswith("final pushover failed:")
            ):
                raise
            result = self._typed_failure(
                cache_key=key,
                sections=sections,
                structural_weight=weight,
                weight_contribution=weight_contribution,
                stage1=stage1_summary,
                stage="stage2",
                status="numerical_failure",
                message=message,
                category=FailureCategory.NUMERICAL_NONCONVERGENCE,
            )
            self.cache.put(result)
            return EvaluationExecution(request_id=evaluation_id, result=result, cache_hit=False)
        except ValueError as error:
            message = str(error)
            invalid_capacity_messages = {
                "compression eliminates positive column hinge moment capacity",
                "compression eliminates positive FEMA column yield rotation",
            }
            if message not in invalid_capacity_messages:
                raise
            result = self._typed_failure(
                cache_key=key,
                sections=sections,
                structural_weight=weight,
                weight_contribution=weight_contribution,
                stage1=stage1_summary,
                stage="stage2",
                status="structural_failure",
                message=message,
                category=FailureCategory.STRUCTURAL_INFEASIBILITY,
            )
            self.cache.put(result)
            return EvaluationExecution(request_id=evaluation_id, result=result, cache_hit=False)

        cp_state = next(
            state
            for state in stage2_result.final.target_states
            if state.target.performance_level == "CP"
        )
        elevations: list[float] = []
        elevation = 0.0
        for height in self.generated.definition.story_heights_m:
            elevation += height
            elevations.append(elevation)
        uniformity = drift_uniformity_term(
            floor_displacements_m=cp_state.story_displacements_m,
            floor_elevations_m=elevations,
        )
        objective = calculate_objective(
            structural_weight=weight,
            maximum_structural_weight=objective_policy.maximum_structural_weight_n,
            uniformity_term=uniformity,
            weight_coefficient=objective_policy.weight_coefficient,
            uniformity_coefficient=objective_policy.uniformity_coefficient,
        )
        stage2_violations = _violations_from_stage2(stage2_result, self.evaluation_policy.penalty)
        feasible = stage2_result.primary_feasible
        penalty = calculate_penalty_score(
            objective_base=objective.raw_objective,
            objective_basis="full_objective",
            feasible=feasible,
            violations=stage2_violations,
            policy=self.evaluation_policy.penalty,
        )
        result = EvaluationResult(
            cache_key=key,
            evaluator_identity=self.identity,
            section_names_by_group=canonical_names,
            section_source_rows_by_group={
                group: item.source_row_id for group, item in sections.items()
            },
            status="feasible" if feasible else "stage2_infeasible",
            rank_class=0 if feasible else 1,
            structural_weight_n=weight,
            weight_contribution=weight_contribution,
            objective=objective,
            cp_story_displacements_m=cp_state.story_displacements_m,
            stage1=stage1_summary,
            stage2=_stage2_summary(stage2_result),
            violations=stage2_violations,
            penalty=penalty,
        )
        self.cache.put(result)
        return EvaluationExecution(request_id=evaluation_id, result=result, cache_hit=False)


def evaluation_order_key(result: EvaluationResult) -> tuple[int, float, str]:
    """Return deterministic order: feasible, structural infeasible, then typed failures."""

    score = result.penalty.ranking_score if result.penalty is not None else math.inf
    return result.rank_class, score, result.cache_key
