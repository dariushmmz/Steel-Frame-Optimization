"""Run deterministic, checkpointable GA campaigns with process-isolated evaluations."""

from __future__ import annotations

import ctypes
import json
import math
import multiprocessing
import os
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from steel_frame_optimization.config.schemas import StrictConfigModel
from steel_frame_optimization.optimization.evaluator import (
    EvaluationResult,
    EvaluationStatus,
    EvaluatorIdentity,
    EvaluatorPaths,
    StructuralEvaluator,
    evaluation_order_key,
)
from steel_frame_optimization.optimization.genetics import (
    DiscreteGeneticAlgorithm,
    GeneticAlgorithmPolicy,
    Genome,
    restore_pcg64,
    serialize_rng_state,
)
from steel_frame_optimization.sections.domains import CandidateDomain, DomainCatalog
from steel_frame_optimization.serialization import (
    canonical_json_dumps,
    sha256_config,
    sha256_file,
    write_canonical_json,
)


class CampaignApprovalError(RuntimeError):
    """Indicate that a production campaign was stopped before any evaluation."""


class PenaltyApprovalRecord(StrictConfigModel):
    """Authorize one exact machine-readable penalty policy for GA execution."""

    schema_version: Literal["1.0"] = "1.0"
    record_id: Literal["phase9-penalty-approval-v1", "phase12-penalty-approval-v1"]
    approval_status: Literal["approved"]
    authorized_by: Literal["delegated_technical_owner"]
    authorization_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    authorization_scope: Literal[
        "bounded_pilot_and_main_30_seed_campaign",
        "all_approved_three_and_nine_story_campaigns",
    ]
    penalty_policy_id: Literal["normalized-additive-feasibility-offset-v1"]
    penalty_policy_path: str = Field(min_length=1)
    penalty_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    campaign_identity_compatibility_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_ids: tuple[str, ...] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def require_approved_decision_ids(self) -> PenaltyApprovalRecord:
        """Require the exact decision and assumption that freeze the penalty policy."""

        if self.decision_ids != ("DEC-004", "ASM-016"):
            raise ValueError("penalty approval must reference DEC-004 and ASM-016 in order")
        return self

    @classmethod
    def from_json(cls, path: Path) -> PenaltyApprovalRecord:
        """Load and strictly validate the delivered approval record."""

        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class InitialDesign(StrictConfigModel):
    """Name one optional source-visible anchor used by a bounded verification pilot."""

    design_id: str = Field(min_length=1)
    sections_by_group: dict[str, str]


class CampaignPolicy(StrictConfigModel):
    """Freeze one pilot or production campaign, including its exact request budget."""

    schema_version: Literal["1.0"] = "1.0"
    campaign_id: str = Field(min_length=1)
    mode: Literal["bounded_pilot", "main_30_seed"]
    frame_id: Literal["3-story", "9-story"]
    case_id: Literal[
        "3_story_case_2_principal",
        "9_story_case_1_filtered",
        "9_story_case_1_no_filter",
        "9_story_case_2_principal",
    ]
    evaluator_profile: Literal[
        "three_story_case2",
        "nine_story_case1_filtered",
        "nine_story_case1_no_filter",
        "nine_story_case2",
    ] = "three_story_case2"
    population_size: int = Field(ge=2)
    evaluated_population_epochs: int = Field(ge=1)
    evaluation_request_budget_per_seed: int = Field(ge=2)
    seeds: tuple[int, ...] = Field(min_length=1)
    seed_manifest_path: str = Field(min_length=1)
    worker_processes: int = Field(ge=1)
    checkpoint_every_epochs: int = Field(ge=1)
    initial_designs: tuple[InitialDesign, ...] = ()
    ga: GeneticAlgorithmPolicy
    assumption_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_exact_budget_and_campaign_scale(self) -> CampaignPolicy:
        """Require population epochs to account for every evaluation request exactly."""

        expected_budget = self.population_size * self.evaluated_population_epochs
        if self.evaluation_request_budget_per_seed != expected_budget:
            raise ValueError("evaluation request budget must equal population times epochs")
        if self.ga.elite_count >= self.population_size:
            raise ValueError("elite count must be smaller than population size")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("campaign seeds must be unique")
        if self.mode == "main_30_seed":
            expected_population = 50 if self.frame_id == "3-story" else 100
            expected_budget = 10_000 if self.frame_id == "3-story" else 20_000
            if (
                self.population_size != expected_population
                or self.evaluated_population_epochs != 200
            ):
                raise ValueError(
                    "main campaign must use the frame-specific population and 200 epochs"
                )
            if self.evaluation_request_budget_per_seed != expected_budget:
                raise ValueError("main campaign must use the exact frame-specific request budget")
            if self.seeds != tuple(range(30)):
                raise ValueError("main campaign must use declared seeds 0 through 29")
            if self.initial_designs:
                raise ValueError("main paper-comparison campaign cannot inject pilot anchors")
        return self

    @classmethod
    def from_json(cls, path: Path) -> CampaignPolicy:
        """Load and validate one explicit campaign policy."""

        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class GenerationRecord(StrictConfigModel):
    """Persist convergence and disposition counts after one evaluated population."""

    epoch_index: int = Field(ge=0)
    evaluation_requests_total: int = Field(ge=1)
    structural_executions_total: int = Field(ge=1)
    cache_hits_total: int = Field(ge=0)
    best_rank_class: Literal[0, 1, 2]
    best_ranking_score: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    best_feasible_objective: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    feasible_count: int = Field(ge=0)
    stage1_infeasible_count: int = Field(ge=0)
    stage2_infeasible_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)


class CampaignCheckpoint(StrictConfigModel):
    """Contain every deterministic state item required for exact seed-run recovery."""

    schema_version: Literal["1.0"] = "1.0"
    campaign_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_identity: EvaluatorIdentity
    seed: int = Field(ge=0)
    completed_epochs: int = Field(ge=0)
    population: tuple[Genome, ...] = Field(min_length=1)
    rng_state_json: str = Field(min_length=1)
    evaluation_requests: int = Field(ge=0)
    structural_executions: int = Field(ge=0)
    cache_hits: int = Field(ge=0)
    cached_results: tuple[EvaluationResult, ...]
    history: tuple[GenerationRecord, ...]


class WorkerEvaluation(StrictConfigModel):
    """Return one process-isolated result plus measured runtime and process memory."""

    result: EvaluationResult
    elapsed_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    peak_working_set_bytes: int | None = Field(default=None, ge=1)
    process_id: int = Field(ge=1)


class SeedRunResult(StrictConfigModel):
    """Summarize one reproducible seed run and its exact evaluation accounting."""

    schema_version: Literal["1.0"] = "1.0"
    campaign_id: str
    seed: int
    completed_epochs: int
    evaluation_requests: int
    structural_executions: int
    cache_hits: int
    elapsed_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    maximum_worker_peak_working_set_bytes: int | None = Field(default=None, ge=1)
    worker_process_ids: tuple[int, ...]
    champion_genome: Genome
    champion: EvaluationResult
    independently_reevaluated_champion_equal: bool
    final_population: tuple[Genome, ...]
    final_results: tuple[EvaluationResult, ...]
    history: tuple[GenerationRecord, ...]
    checkpoint_path: str


class DesignCodec:
    """Map five stable integer genes to approved filtered section domains."""

    def __init__(self, evaluator: StructuralEvaluator, catalog: DomainCatalog) -> None:
        """Build domains from the approved Phase 5 audit without re-filtering physics."""

        audit_by_group = {group.group_id: group for group in evaluator.filter_audit.groups}
        frame_spec = next(
            frame
            for frame in catalog.frames
            if frame.frame_id == evaluator.generated.definition.frame_id
        )
        spec_by_group = {group.group_id: group for group in frame_spec.groups}
        if set(spec_by_group) != set(audit_by_group):
            raise ValueError("Phase 2 domain catalog and Phase 5 audit group IDs differ")
        group_ids = tuple(sorted(audit_by_group, key=int))
        domains: list[CandidateDomain] = []
        for group_id in group_ids:
            audit = audit_by_group[group_id]
            base_records = evaluator.repository.family(spec_by_group[group_id].base_family)
            rejected = {item.section_name for item in audit.rejected_sections}
            survivors = tuple(item for item in base_records if item.name not in rejected)
            if len(base_records) - len(survivors) != audit.actual_removed_count:
                raise ValueError(f"Phase 5 domain removal mismatch for group {group_id}")
            domains.append(CandidateDomain(group_id=group_id, records=survivors))
        self.group_ids = group_ids
        self.domains = tuple(domains)

    @property
    def domain_sizes(self) -> tuple[int, ...]:
        """Return gene-domain sizes in stable group order."""

        return tuple(len(domain.records) for domain in self.domains)

    def encode(self, sections_by_group: Mapping[str, str]) -> Genome:
        """Encode one complete section assignment into stable integer alleles."""

        if set(sections_by_group) != set(self.group_ids):
            raise ValueError("section assignment must contain every design group")
        return Genome(
            alleles=tuple(
                domain.encode(sections_by_group[group_id])
                for group_id, domain in zip(self.group_ids, self.domains, strict=True)
            )
        )

    def decode(self, genome: Genome) -> dict[str, str]:
        """Decode a legal genome into canonical section names by group."""

        if len(genome.alleles) != len(self.domains):
            raise ValueError("genome length does not match configured groups")
        return {
            group_id: domain.decode(allele).name
            for group_id, domain, allele in zip(
                self.group_ids, self.domains, genome.alleles, strict=True
            )
        }


_WORKER_EVALUATOR: StructuralEvaluator | None = None


def _initialize_worker(project_root: str, evaluator_profile: str) -> None:
    """Create one worker-local evaluator so OpenSees state is never shared across processes."""

    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = StructuralEvaluator(
        EvaluatorPaths.from_profile(Path(project_root), evaluator_profile)
    )


class _WindowsProcessMemoryCounters(ctypes.Structure):
    """Match the Win32 PROCESS_MEMORY_COUNTERS layout used for peak working set."""

    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _peak_working_set_bytes() -> int | None:
    """Read the current process peak working set on Windows; return null elsewhere."""

    if sys.platform != "win32":
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(_WindowsProcessMemoryCounters),
        ctypes.c_ulong,
    )
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    counters = _WindowsProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    process = kernel32.GetCurrentProcess()
    succeeded = psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
    if not succeeded:
        return None
    return int(counters.PeakWorkingSetSize)


def _evaluate_in_worker(payload: tuple[str, dict[str, str]]) -> str:
    """Evaluate one unique design in an initialized spawned worker and return canonical JSON."""

    if _WORKER_EVALUATOR is None:
        raise RuntimeError("campaign worker was not initialized")
    evaluation_id, sections = payload
    started = time.perf_counter()
    execution = _WORKER_EVALUATOR.evaluate(
        evaluation_id=evaluation_id,
        section_names_by_group=sections,
        use_cache=False,
    )
    measured = WorkerEvaluation(
        result=execution.result,
        elapsed_seconds=time.perf_counter() - started,
        peak_working_set_bytes=_peak_working_set_bytes(),
        process_id=os.getpid(),
    )
    return canonical_json_dumps(measured)


def _atomic_write_checkpoint(path: Path, checkpoint: CampaignCheckpoint) -> None:
    """Replace a checkpoint only after its complete canonical temporary file is written."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    write_canonical_json(temporary, checkpoint)
    temporary.replace(path)


def _load_checkpoint(path: Path) -> CampaignCheckpoint:
    """Load and strictly validate one canonical campaign checkpoint."""

    return CampaignCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))


def _rank_key(result: EvaluationResult, genome: Genome) -> tuple[object, ...]:
    """Add alleles to the Phase 8 order key for deterministic stochastic ties."""

    return (*evaluation_order_key(result), genome.alleles)


class CampaignRunner:
    """Coordinate deterministic GA evolution, cache accounting, spawn workers, and recovery."""

    def __init__(self, *, project_root: Path, policy_path: Path) -> None:
        """Load immutable policy/evaluator state and enforce the campaign approval gate."""

        self.project_root = project_root.resolve()
        self.policy_path = policy_path.resolve()
        self.policy = CampaignPolicy.from_json(self.policy_path)
        domain_catalog_path = (
            self.project_root / "configs" / "frames" / "phase2_candidate_domains.json"
        )
        seed_manifest_path = self.project_root / self.policy.seed_manifest_path
        self.evaluator = StructuralEvaluator(
            EvaluatorPaths.from_profile(self.project_root, self.policy.evaluator_profile)
        )
        self.penalty_approval = self._load_penalty_approval()
        # Keep the established campaign identity stable while replacing the excluded
        # human-readable approval document with a strict machine-readable record.
        self.campaign_sha256 = sha256_config(
            {
                "campaign_policy": sha256_file(self.policy_path),
                "domain_catalog": sha256_file(domain_catalog_path),
                "seed_manifest": sha256_file(seed_manifest_path),
                "client_penalty_policy": (
                    self.penalty_approval.campaign_identity_compatibility_sha256
                ),
            }
        )
        self.codec = DesignCodec(self.evaluator, DomainCatalog.from_json(domain_catalog_path))
        self._verify_seed_manifest()
        if (
            self.policy.mode == "main_30_seed"
            and self.penalty_approval.approval_status != "approved"
        ):
            raise CampaignApprovalError("main 30-seed campaign penalty policy is not approved")

    def _load_penalty_approval(self) -> PenaltyApprovalRecord:
        """Require approval of the exact penalty configuration shipped with the project."""

        record_name = (
            "phase9_penalty_approval.json"
            if self.policy.frame_id == "3-story"
            else "phase12_penalty_approval.json"
        )
        approval_path = self.project_root / "configs" / "optimization" / record_name
        try:
            approval = PenaltyApprovalRecord.from_json(approval_path)
        except (OSError, ValueError) as error:
            raise CampaignApprovalError(f"invalid penalty approval record: {error}") from error
        penalty_policy_path = self.project_root / approval.penalty_policy_path
        if sha256_file(penalty_policy_path) != approval.penalty_policy_sha256:
            raise CampaignApprovalError("approved penalty policy hash does not match configuration")
        content = json.loads(penalty_policy_path.read_text(encoding="utf-8"))
        penalty = content.get("penalty")
        if not isinstance(penalty, dict) or penalty.get("policy_id") != approval.penalty_policy_id:
            raise CampaignApprovalError("approved penalty policy ID does not match configuration")
        if self.evaluator.evaluation_policy.penalty.model_dump(mode="json") != penalty:
            raise CampaignApprovalError(
                "evaluator penalty coefficients differ from the approved machine-readable policy"
            )
        return approval

    def _verify_seed_manifest(self) -> None:
        """Require every configured seed to occur in the committed PCG64 seed manifest."""

        path = self.project_root / self.policy.seed_manifest_path
        content = json.loads(path.read_text(encoding="utf-8"))
        if content.get("generator") != "numpy.random.PCG64":
            raise ValueError("seed manifest does not declare numpy.random.PCG64")
        declared = content.get("seeds")
        if not isinstance(declared, list) or not all(isinstance(seed, int) for seed in declared):
            raise ValueError("seed manifest contains invalid seeds")
        if any(seed not in declared for seed in self.policy.seeds):
            raise ValueError("campaign uses a seed absent from the declared manifest")

    def _new_ga(
        self, *, seed: int, rng: np.random.Generator | None = None
    ) -> DiscreteGeneticAlgorithm:
        """Build one GA over the verified group-domain ordering."""

        generator = rng if rng is not None else np.random.Generator(np.random.PCG64(seed))
        return DiscreteGeneticAlgorithm(
            domain_sizes=self.codec.domain_sizes,
            policy=self.policy.ga,
            rng=generator,
        )

    def _initial_population(self, ga: DiscreteGeneticAlgorithm) -> tuple[Genome, ...]:
        """Create the configured anchor-first pilot or fully random main population."""

        anchors = tuple(
            self.codec.encode(item.sections_by_group) for item in self.policy.initial_designs
        )
        return ga.initialize_population(self.policy.population_size, anchors=anchors)

    def _checkpoint_from_state(
        self,
        *,
        seed: int,
        completed_epochs: int,
        population: tuple[Genome, ...],
        ga: DiscreteGeneticAlgorithm,
        requests: int,
        executions: int,
        cache_hits: int,
        result_cache: Mapping[str, EvaluationResult],
        history: Sequence[GenerationRecord],
    ) -> CampaignCheckpoint:
        """Build a deterministically ordered complete recovery record."""

        return CampaignCheckpoint(
            campaign_sha256=self.campaign_sha256,
            evaluator_identity=self.evaluator.identity,
            seed=seed,
            completed_epochs=completed_epochs,
            population=population,
            rng_state_json=serialize_rng_state(ga.rng),
            evaluation_requests=requests,
            structural_executions=executions,
            cache_hits=cache_hits,
            cached_results=tuple(sorted(result_cache.values(), key=lambda item: item.cache_key)),
            history=tuple(history),
        )

    def _restore_state(
        self, checkpoint_path: Path, seed: int
    ) -> tuple[
        DiscreteGeneticAlgorithm,
        tuple[Genome, ...],
        int,
        int,
        int,
        int,
        dict[str, EvaluationResult],
        list[GenerationRecord],
    ]:
        """Validate identities and recover all deterministic state from a checkpoint."""

        checkpoint = _load_checkpoint(checkpoint_path)
        if checkpoint.campaign_sha256 != self.campaign_sha256:
            raise ValueError("checkpoint campaign identity differs from current policy")
        if checkpoint.evaluator_identity != self.evaluator.identity:
            raise ValueError("checkpoint evaluator identity differs from current implementation")
        if checkpoint.seed != seed:
            raise ValueError("checkpoint seed differs from requested seed")
        ga = self._new_ga(seed=seed, rng=restore_pcg64(checkpoint.rng_state_json))
        result_cache = {item.cache_key: item for item in checkpoint.cached_results}
        return (
            ga,
            checkpoint.population,
            checkpoint.completed_epochs,
            checkpoint.evaluation_requests,
            checkpoint.structural_executions,
            checkpoint.cache_hits,
            result_cache,
            list(checkpoint.history),
        )

    def _evaluate_population(
        self,
        *,
        population: tuple[Genome, ...],
        result_cache: dict[str, EvaluationResult],
        executor: ProcessPoolExecutor,
        epoch_index: int,
    ) -> tuple[tuple[EvaluationResult, ...], int, int, tuple[WorkerEvaluation, ...]]:
        """Evaluate each unique cache miss once while accounting for every request."""

        keys_and_sections = tuple(
            (self.evaluator.cache_key(self.codec.decode(genome)), self.codec.decode(genome))
            for genome in population
        )
        new_payloads: list[tuple[str, dict[str, str]]] = []
        scheduled_keys: set[str] = set()
        for key, sections in keys_and_sections:
            if key not in result_cache and key not in scheduled_keys:
                new_payloads.append((f"epoch-{epoch_index}-{key[:16]}", sections))
                scheduled_keys.add(key)
        worker_results = tuple(
            WorkerEvaluation.model_validate_json(content)
            for content in executor.map(_evaluate_in_worker, new_payloads)
        )
        for measured in worker_results:
            expected_key = self.evaluator.cache_key(measured.result.section_names_by_group)
            if measured.result.cache_key != expected_key:
                raise RuntimeError("worker returned a result under the wrong cache identity")
            existing = result_cache.get(expected_key)
            if existing is not None and existing != measured.result:
                raise RuntimeError("identical cache key produced different structural results")
            result_cache[expected_key] = measured.result
        results = tuple(result_cache[key] for key, _ in keys_and_sections)
        executions = len(worker_results)
        cache_hits = len(population) - executions
        return results, executions, cache_hits, worker_results

    def run_seed(
        self,
        *,
        seed: int,
        checkpoint_path: Path,
        resume: bool = False,
        maximum_epochs: int | None = None,
    ) -> SeedRunResult:
        """Run or resume one seed without exceeding its configured population epochs."""

        if seed not in self.policy.seeds:
            raise ValueError("seed is not part of this campaign")
        if maximum_epochs is not None and maximum_epochs < 1:
            raise ValueError("maximum epochs must be positive")
        if resume:
            (
                ga,
                population,
                completed,
                requests,
                executions,
                cache_hits,
                result_cache,
                history,
            ) = self._restore_state(checkpoint_path, seed)
        else:
            ga = self._new_ga(seed=seed)
            population = self._initial_population(ga)
            completed = 0
            requests = 0
            executions = 0
            cache_hits = 0
            result_cache = {}
            history = []
        target_epochs = self.policy.evaluated_population_epochs
        if maximum_epochs is not None:
            target_epochs = min(target_epochs, maximum_epochs)
        if completed > target_epochs:
            raise ValueError("checkpoint is beyond the requested stop epoch")
        started = time.perf_counter()
        worker_measurements: list[WorkerEvaluation] = []
        last_results: tuple[EvaluationResult, ...] | None = None
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=self.policy.worker_processes,
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(str(self.project_root), self.policy.evaluator_profile),
        ) as executor:
            while completed < target_epochs:
                results, new_executions, new_hits, measured = self._evaluate_population(
                    population=population,
                    result_cache=result_cache,
                    executor=executor,
                    epoch_index=completed,
                )
                last_results = results
                worker_measurements.extend(measured)
                requests += len(population)
                executions += new_executions
                cache_hits += new_hits
                best_so_far = min(result_cache.values(), key=evaluation_order_key)
                status_counts = Counter(result.status for result in results)
                history.append(
                    GenerationRecord(
                        epoch_index=completed,
                        evaluation_requests_total=requests,
                        structural_executions_total=executions,
                        cache_hits_total=cache_hits,
                        best_rank_class=best_so_far.rank_class,
                        best_ranking_score=(
                            best_so_far.penalty.ranking_score
                            if best_so_far.penalty is not None
                            else None
                        ),
                        best_feasible_objective=(
                            best_so_far.objective.raw_objective
                            if best_so_far.status == "feasible"
                            and best_so_far.objective is not None
                            else None
                        ),
                        feasible_count=status_counts["feasible"],
                        stage1_infeasible_count=status_counts["stage1_infeasible"],
                        stage2_infeasible_count=status_counts["stage2_infeasible"],
                        failed_count=(
                            status_counts["invalid_input"]
                            + status_counts["structural_failure"]
                            + status_counts["numerical_failure"]
                        ),
                    )
                )
                completed += 1
                if completed < self.policy.evaluated_population_epochs:
                    rank_keys = tuple(
                        _rank_key(result, genome)
                        for genome, result in zip(population, results, strict=True)
                    )
                    population = ga.next_population(population=population, rank_keys=rank_keys)
                    last_results = None
                if (
                    completed % self.policy.checkpoint_every_epochs == 0
                    or completed == target_epochs
                ):
                    checkpoint = self._checkpoint_from_state(
                        seed=seed,
                        completed_epochs=completed,
                        population=population,
                        ga=ga,
                        requests=requests,
                        executions=executions,
                        cache_hits=cache_hits,
                        result_cache=result_cache,
                        history=history,
                    )
                    _atomic_write_checkpoint(checkpoint_path, checkpoint)
        elapsed = time.perf_counter() - started
        if completed == self.policy.evaluated_population_epochs:
            if last_results is None:
                last_results = tuple(
                    result_cache[self.evaluator.cache_key(self.codec.decode(genome))]
                    for genome in population
                )
        else:
            partial_results: list[EvaluationResult] = []
            for genome in population:
                key = self.evaluator.cache_key(self.codec.decode(genome))
                if key in result_cache:
                    partial_results.append(result_cache[key])
            last_results = tuple(partial_results)
        if not result_cache:
            raise RuntimeError("campaign completed no structural evaluations")
        champion = min(result_cache.values(), key=evaluation_order_key)
        champion_genome = self.codec.encode(champion.section_names_by_group)
        independent = StructuralEvaluator(
            EvaluatorPaths.from_profile(self.project_root, self.policy.evaluator_profile)
        ).evaluate(
            evaluation_id=f"seed-{seed}-champion-independent",
            section_names_by_group=champion.section_names_by_group,
            use_cache=False,
        )
        if last_results is None or len(last_results) != len(population):
            final_results = tuple(
                result_cache[self.evaluator.cache_key(self.codec.decode(genome))]
                for genome in population
                if self.evaluator.cache_key(self.codec.decode(genome)) in result_cache
            )
        else:
            final_results = last_results
        peaks = tuple(
            item.peak_working_set_bytes
            for item in worker_measurements
            if item.peak_working_set_bytes is not None
        )
        return SeedRunResult(
            campaign_id=self.policy.campaign_id,
            seed=seed,
            completed_epochs=completed,
            evaluation_requests=requests,
            structural_executions=executions,
            cache_hits=cache_hits,
            elapsed_seconds=elapsed,
            maximum_worker_peak_working_set_bytes=max(peaks) if peaks else None,
            worker_process_ids=tuple(sorted({item.process_id for item in worker_measurements})),
            champion_genome=champion_genome,
            champion=champion,
            independently_reevaluated_champion_equal=independent.result == champion,
            final_population=population,
            final_results=final_results,
            history=tuple(history),
            checkpoint_path=checkpoint_path.as_posix(),
        )

    def run_all(
        self, *, checkpoint_directory: Path, resume_existing: bool = False
    ) -> tuple[SeedRunResult, ...]:
        """Run configured seeds in order, optionally resuming each existing checkpoint."""

        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        results: list[SeedRunResult] = []
        for seed in self.policy.seeds:
            checkpoint_path = checkpoint_directory / f"seed_{seed:02d}.json"
            results.append(
                self.run_seed(
                    seed=seed,
                    checkpoint_path=checkpoint_path,
                    resume=resume_existing and checkpoint_path.exists(),
                )
            )
        return tuple(results)


def status_counts(results: Sequence[EvaluationResult]) -> dict[EvaluationStatus, int]:
    """Return exhaustive typed evaluation disposition counts for reporting."""

    counts = Counter(result.status for result in results)
    statuses: tuple[EvaluationStatus, ...] = (
        "feasible",
        "stage1_infeasible",
        "stage2_infeasible",
        "structural_failure",
        "invalid_input",
        "numerical_failure",
    )
    return {status: counts[status] for status in statuses}


def projected_hours(*, seconds_per_execution: float, executions: int, workers: int) -> float:
    """Return an idealized wall-clock projection with explicit positive inputs."""

    if seconds_per_execution <= 0.0 or not math.isfinite(seconds_per_execution):
        raise ValueError("seconds per execution must be finite and positive")
    if executions <= 0 or workers <= 0:
        raise ValueError("executions and workers must be positive")
    return seconds_per_execution * executions / workers / 3600.0
