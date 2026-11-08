"""Generate Phase 9 bounded-pilot and runtime/storage gate evidence."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Literal

from pydantic import Field

from steel_frame_optimization.config.schemas import StrictConfigModel
from steel_frame_optimization.optimization.campaign import (
    CampaignApprovalError,
    CampaignRunner,
    GenerationRecord,
    SeedRunResult,
    projected_hours,
    status_counts,
)
from steel_frame_optimization.optimization.evaluator import (
    EvaluationResult,
    EvaluationStatus,
    EvaluatorPaths,
    StructuralEvaluator,
)
from steel_frame_optimization.optimization.genetics import Genome
from steel_frame_optimization.reporting.optimization import write_convergence_svg
from steel_frame_optimization.serialization import (
    canonical_json_dumps,
    sha256_config,
    sha256_file,
    write_canonical_json,
)


class SingleEvaluationBenchmark(StrictConfigModel):
    """Record one uncached structural path's elapsed time and compact-result size."""

    design_id: str
    status: str
    elapsed_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    canonical_result_bytes: int = Field(ge=1)


class PilotSeedEvidence(StrictConfigModel):
    """Persist deterministic optimization outcomes separately from timing noise."""

    seed: int
    completed_epochs: int
    evaluation_requests: int
    structural_executions: int
    cache_hits: int
    champion_genome: Genome
    champion: EvaluationResult
    independently_reevaluated_champion_equal: bool
    final_status_counts: dict[EvaluationStatus, int]
    history: tuple[GenerationRecord, ...]
    deterministic_signature_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PilotAggregate(StrictConfigModel):
    """Summarize bounded-run champions without presenting them as a 30-run comparison."""

    run_count: int = Field(ge=1)
    best_feasible_objective: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    all_runs_mean_champion_weight_kn: float = Field(gt=0.0, allow_inf_nan=False)
    all_runs_worst_champion_weight_kn: float = Field(gt=0.0, allow_inf_nan=False)
    feasible_runs_mean_champion_weight_kn: float | None = Field(
        default=None, gt=0.0, allow_inf_nan=False
    )
    feasible_runs_worst_champion_weight_kn: float | None = Field(
        default=None, gt=0.0, allow_inf_nan=False
    )
    feasible_champion_count: int = Field(ge=0)
    infeasible_or_failed_champion_count: int = Field(ge=0)


class EvidenceFile(StrictConfigModel):
    """Identify one deterministic Phase 9 plot by path, size, and exact bytes."""

    path: str
    size_bytes: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RuntimeStorageGate(StrictConfigModel):
    """Record measured Windows costs and explicit main-campaign projections."""

    platform: str
    logical_cpu_count: int | None
    configured_worker_processes: int
    single_evaluations: tuple[SingleEvaluationBenchmark, ...]
    cache_hit_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    pilot_elapsed_seconds_by_seed: dict[str, float]
    maximum_worker_peak_working_set_bytes: int | None = Field(default=None, ge=1)
    pilot_checkpoint_bytes_by_seed: dict[str, int]
    worst_measured_full_evaluation_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    projected_main_hours_per_seed_four_workers: float = Field(gt=0.0, allow_inf_nan=False)
    projected_30_seed_hours_sequential: float = Field(gt=0.0, allow_inf_nan=False)
    projected_max_checkpoint_bytes_per_seed: int = Field(ge=1)
    projection_basis: str


class Phase9Artifact(StrictConfigModel):
    """Combine bounded GA evidence with the unambiguous production-campaign stop."""

    schema_version: Literal["1.0"] = "1.0"
    pilot_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    main_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    domain_sizes: tuple[int, ...]
    pilot_runs: tuple[PilotSeedEvidence, ...]
    pilot_aggregate: PilotAggregate
    convergence_plots: tuple[EvidenceFile, ...]
    runtime_storage_gate: RuntimeStorageGate
    main_campaign_status: Literal["blocked_pending_client_penalty_approval"]
    main_campaign_evaluations_executed: Literal[0] = 0
    all_pilot_champions_independently_equal: bool
    all_accounting_exact: bool
    any_numerical_failures: bool
    sources: tuple[str, ...]
    limitations: tuple[str, ...]


def _deterministic_pilot_evidence(result: SeedRunResult) -> PilotSeedEvidence:
    """Remove wall-clock/process metadata while retaining complete optimization behavior."""

    core = {
        "seed": result.seed,
        "completed_epochs": result.completed_epochs,
        "evaluation_requests": result.evaluation_requests,
        "structural_executions": result.structural_executions,
        "cache_hits": result.cache_hits,
        "champion_genome": result.champion_genome,
        "champion": result.champion,
        "final_status_counts": status_counts(result.final_results),
        "history": result.history,
    }
    return PilotSeedEvidence(
        seed=result.seed,
        completed_epochs=result.completed_epochs,
        evaluation_requests=result.evaluation_requests,
        structural_executions=result.structural_executions,
        cache_hits=result.cache_hits,
        champion_genome=result.champion_genome,
        champion=result.champion,
        independently_reevaluated_champion_equal=(result.independently_reevaluated_champion_equal),
        final_status_counts=status_counts(result.final_results),
        history=result.history,
        deterministic_signature_sha256=sha256_config(core),
    )


def _benchmark_single_paths(
    *, project_root: Path, representative_designs_path: Path
) -> tuple[tuple[SingleEvaluationBenchmark, ...], float]:
    """Measure all three Phase 8 dispatch paths plus one true in-memory cache hit."""

    content = json.loads(representative_designs_path.read_text(encoding="utf-8"))
    designs = content.get("designs")
    if not isinstance(designs, list):
        raise ValueError("representative design file has no designs list")
    evaluator = StructuralEvaluator(EvaluatorPaths.from_project_root(project_root))
    measurements: list[SingleEvaluationBenchmark] = []
    first_sections: dict[str, str] | None = None
    for item in designs:
        if not isinstance(item, dict):
            raise ValueError("representative design entry must be an object")
        design_id = item.get("design_id")
        sections = item.get("sections_by_group")
        if not isinstance(design_id, str) or not isinstance(sections, dict):
            raise ValueError("representative design has invalid ID or sections")
        canonical_sections = {str(group): str(name) for group, name in sections.items()}
        if first_sections is None:
            first_sections = canonical_sections
        started = time.perf_counter()
        result = evaluator.evaluate(
            evaluation_id=f"benchmark-{design_id}",
            section_names_by_group=canonical_sections,
            use_cache=False,
        ).result
        elapsed = time.perf_counter() - started
        measurements.append(
            SingleEvaluationBenchmark(
                design_id=design_id,
                status=result.status,
                elapsed_seconds=elapsed,
                canonical_result_bytes=len(canonical_json_dumps(result).encode("utf-8")),
            )
        )
    if first_sections is None:
        raise ValueError("benchmark requires at least one representative design")
    evaluator.evaluate(
        evaluation_id="benchmark-cache-prime",
        section_names_by_group=first_sections,
        use_cache=False,
    )
    started = time.perf_counter()
    cached = evaluator.evaluate(
        evaluation_id="benchmark-cache-hit",
        section_names_by_group=first_sections,
    )
    cache_elapsed = time.perf_counter() - started
    if not cached.cache_hit:
        raise RuntimeError("benchmark cache-hit request did not hit")
    return tuple(measurements), cache_elapsed


def build_phase9_artifact(
    *,
    project_root: Path,
    pilot_policy_path: Path,
    main_policy_path: Path,
    representative_designs_path: Path,
    checkpoint_directory: Path,
    evidence_directory: Path,
) -> Phase9Artifact:
    """Run authorized pilots, benchmark costs, and prove the main campaign remains blocked."""

    pilot_runner = CampaignRunner(project_root=project_root, policy_path=pilot_policy_path)
    pilot_results = pilot_runner.run_all(checkpoint_directory=checkpoint_directory)
    pilot_evidence = tuple(_deterministic_pilot_evidence(result) for result in pilot_results)
    plot_files: list[EvidenceFile] = []
    for run in pilot_evidence:
        plot_path = evidence_directory / f"convergence_seed_{run.seed:02d}.svg"
        write_convergence_svg(
            plot_path,
            run.history,
            title=f"Phase 9 bounded pilot — seed {run.seed}",
        )
        plot_files.append(
            EvidenceFile(
                path=plot_path.as_posix(),
                size_bytes=plot_path.stat().st_size,
                sha256=sha256_file(plot_path),
            )
        )
    single_evaluations, cache_hit_seconds = _benchmark_single_paths(
        project_root=project_root,
        representative_designs_path=representative_designs_path,
    )
    try:
        CampaignRunner(project_root=project_root, policy_path=main_policy_path)
    except CampaignApprovalError:
        main_blocked = True
    else:
        main_blocked = False
    if not main_blocked:
        raise RuntimeError("main campaign approval gate unexpectedly opened")
    full_times = tuple(
        item.elapsed_seconds
        for item in single_evaluations
        if item.status in {"feasible", "stage2_infeasible"}
    )
    if not full_times:
        raise RuntimeError("runtime gate measured no full Stage 2 evaluation")
    worst_full_seconds = max(full_times)
    largest_result_bytes = max(item.canonical_result_bytes for item in single_evaluations)
    projected_checkpoint_bytes = largest_result_bytes * 10_000
    projected_per_seed = projected_hours(
        seconds_per_execution=worst_full_seconds,
        executions=10_000,
        workers=4,
    )
    peaks = tuple(
        result.maximum_worker_peak_working_set_bytes
        for result in pilot_results
        if result.maximum_worker_peak_working_set_bytes is not None
    )
    checkpoint_sizes = {
        str(result.seed): (checkpoint_directory / f"seed_{result.seed:02d}.json").stat().st_size
        for result in pilot_results
    }
    runtime_gate = RuntimeStorageGate(
        platform=platform.platform(),
        logical_cpu_count=os.cpu_count(),
        configured_worker_processes=pilot_runner.policy.worker_processes,
        single_evaluations=single_evaluations,
        cache_hit_seconds=cache_hit_seconds,
        pilot_elapsed_seconds_by_seed={
            str(result.seed): result.elapsed_seconds for result in pilot_results
        },
        maximum_worker_peak_working_set_bytes=max(peaks) if peaks else None,
        pilot_checkpoint_bytes_by_seed=checkpoint_sizes,
        worst_measured_full_evaluation_seconds=worst_full_seconds,
        projected_main_hours_per_seed_four_workers=projected_per_seed,
        projected_30_seed_hours_sequential=projected_per_seed * 30.0,
        projected_max_checkpoint_bytes_per_seed=projected_checkpoint_bytes,
        projection_basis=(
            "conservative all-requests-uncached projection using the slower measured full Stage 2 "
            "path; ideal four-worker division excludes process overhead and checkpoint I/O"
        ),
    )
    champion_weights_kn = tuple(
        result.champion.structural_weight_n / 1000.0
        for result in pilot_results
        if result.champion.structural_weight_n is not None
    )
    if len(champion_weights_kn) != len(pilot_results):
        raise RuntimeError("pilot champion is missing structural weight")
    feasible_champions = tuple(
        result.champion for result in pilot_results if result.champion.status == "feasible"
    )
    feasible_weights_kn = tuple(
        result.structural_weight_n / 1000.0
        for result in feasible_champions
        if result.structural_weight_n is not None
    )
    feasible_objectives = tuple(
        result.objective.raw_objective
        for result in feasible_champions
        if result.objective is not None
    )
    aggregate = PilotAggregate(
        run_count=len(pilot_results),
        best_feasible_objective=min(feasible_objectives) if feasible_objectives else None,
        all_runs_mean_champion_weight_kn=statistics.fmean(champion_weights_kn),
        all_runs_worst_champion_weight_kn=max(champion_weights_kn),
        feasible_runs_mean_champion_weight_kn=(
            statistics.fmean(feasible_weights_kn) if feasible_weights_kn else None
        ),
        feasible_runs_worst_champion_weight_kn=(
            max(feasible_weights_kn) if feasible_weights_kn else None
        ),
        feasible_champion_count=len(feasible_champions),
        infeasible_or_failed_champion_count=len(pilot_results) - len(feasible_champions),
    )
    all_results = tuple(result for run in pilot_results for result in run.final_results)
    return Phase9Artifact(
        pilot_policy_sha256=sha256_file(pilot_policy_path),
        main_policy_sha256=sha256_file(main_policy_path),
        domain_sizes=pilot_runner.codec.domain_sizes,
        pilot_runs=pilot_evidence,
        pilot_aggregate=aggregate,
        convergence_plots=tuple(plot_files),
        runtime_storage_gate=runtime_gate,
        main_campaign_status="blocked_pending_client_penalty_approval",
        all_pilot_champions_independently_equal=all(
            result.independently_reevaluated_champion_equal for result in pilot_results
        ),
        all_accounting_exact=all(
            result.structural_executions + result.cache_hits == result.evaluation_requests
            for result in pilot_results
        ),
        any_numerical_failures=any(
            result.status in {"structural_failure", "invalid_input", "numerical_failure"}
            for result in all_results
        ),
        sources=(
            "ASM-002 PCG64 seeds and exact evaluation-request budget",
            "ASM-016/DEC-004 feasibility-offset penalty policy",
            "Phase 2 stable weight/source-row domain ordering",
            "Phase 8 evaluator and complete cache identity",
        ),
        limitations=(
            "bounded pilots use two seeds, 8 members and 4 epochs and are not research campaigns",
            "known Phase 8 feasible design is an explicit pilot anchor, never a main-campaign seed",
            "main 30-seed campaign executed zero evaluations pending client penalty approval",
            "wall-clock and memory measurements are Windows-machine-specific",
            "results are not independently engineering-validated or professionally certified",
        ),
    )


def phase9_gate_failures(artifact: Phase9Artifact) -> tuple[str, ...]:
    """Return every failed currently authorized Phase 9 software/numerical gate."""

    checks = (
        (artifact.all_pilot_champions_independently_equal, "champion reevaluation mismatch"),
        (artifact.all_accounting_exact, "evaluation accounting mismatch"),
        (not artifact.any_numerical_failures, "bounded pilot produced a typed evaluation failure"),
        (artifact.main_campaign_evaluations_executed == 0, "unapproved main evaluation executed"),
    )
    return tuple(message for passed, message in checks if not passed)


def _parse_args() -> argparse.Namespace:
    """Parse explicit Phase 9 evidence and checkpoint paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pilot-policy", type=Path, required=True)
    parser.add_argument("--main-policy", type=Path, required=True)
    parser.add_argument("--representative-designs", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Write Phase 9 evidence and fail if any currently authorized gate fails."""

    args = _parse_args()
    artifact = build_phase9_artifact(
        project_root=args.project_root,
        pilot_policy_path=args.pilot_policy,
        main_policy_path=args.main_policy,
        representative_designs_path=args.representative_designs,
        checkpoint_directory=args.checkpoint_directory,
        evidence_directory=args.output.parent,
    )
    failures = phase9_gate_failures(artifact)
    write_canonical_json(args.output, artifact)
    signatures = tuple(run.deterministic_signature_sha256 for run in artifact.pilot_runs)
    print(
        f"Phase 9 pilot_signatures={signatures}, accounting={artifact.all_accounting_exact}, "
        f"main={artifact.main_campaign_status}, failures={len(failures)}"
    )
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
