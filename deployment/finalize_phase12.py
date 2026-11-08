"""Validate and summarize one completed Phase 12 nine-story GA campaign."""

from __future__ import annotations

import csv
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from steel_frame_optimization.optimization.campaign import (
    CampaignCheckpoint,
    CampaignRunner,
    SeedRunResult,
    status_counts,
)
from steel_frame_optimization.optimization.evaluator import evaluation_order_key
from steel_frame_optimization.reporting.optimization import write_convergence_svg
from steel_frame_optimization.serialization import sha256_file


def _required(value: float | None, *, name: str, seed: int) -> float:
    """Return a required champion metric or reject an incomplete result."""

    if value is None:
        raise ValueError(f"seed {seed} champion is missing {name}")
    return value


def _remote_elapsed(path: Path) -> float | None:
    """Read the final durable elapsed-time record when present."""

    if not path.is_file():
        return None
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if line.startswith("complete seed="):
            fields = dict(item.split("=", 1) for item in line.split() if "=" in item)
            return float(fields["elapsed_seconds"])
    return None


def _write_convergence(path: Path, runs: tuple[SeedRunResult, ...], population: int) -> None:
    """Write deterministic cross-seed convergence statistics."""

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("epoch", "requests_per_seed", "feasible_seeds", "best", "mean", "worst"))
        for epoch in range(len(runs[0].history)):
            values = tuple(
                value
                for run in runs
                if (value := run.history[epoch].best_feasible_objective) is not None
            )
            writer.writerow(
                (
                    epoch + 1,
                    (epoch + 1) * population,
                    len(values),
                    min(values) if values else "",
                    statistics.fmean(values) if values else "",
                    max(values) if values else "",
                )
            )


def build_phase12_campaign_summary(
    *,
    project_root: Path,
    policy_path: Path,
    campaign_directory: Path,
    evidence_directory: Path,
) -> dict[str, Any]:
    """Validate identities, accounting, recovery state, champions, and statistics."""

    runner = CampaignRunner(project_root=project_root, policy_path=policy_path)
    evidence_directory.mkdir(parents=True, exist_ok=True)
    runs: list[SeedRunResult] = []
    records: list[dict[str, Any]] = []
    final_statuses: Counter[str] = Counter()
    for seed in runner.policy.seeds:
        checkpoint_path = campaign_directory / "checkpoints" / f"seed_{seed:02d}.json"
        result_path = campaign_directory / "results" / f"seed_{seed:02d}.json"
        checkpoint = CampaignCheckpoint.model_validate_json(
            checkpoint_path.read_text(encoding="utf-8")
        )
        run = SeedRunResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        if checkpoint.campaign_sha256 != runner.campaign_sha256:
            raise ValueError(f"seed {seed} campaign identity differs")
        if checkpoint.evaluator_identity != runner.evaluator.identity:
            raise ValueError(f"seed {seed} evaluator identity differs")
        if checkpoint.seed != seed or run.seed != seed:
            raise ValueError(f"seed {seed} artifact identity differs")
        if checkpoint.completed_epochs != runner.policy.evaluated_population_epochs:
            raise ValueError(f"seed {seed} has incomplete checkpoint epochs")
        if run.completed_epochs != runner.policy.evaluated_population_epochs:
            raise ValueError(f"seed {seed} has incomplete result epochs")
        budget = runner.policy.evaluation_request_budget_per_seed
        if checkpoint.evaluation_requests != budget or run.evaluation_requests != budget:
            raise ValueError(f"seed {seed} does not preserve the exact request budget")
        if checkpoint.structural_executions + checkpoint.cache_hits != budget:
            raise ValueError(f"seed {seed} checkpoint accounting differs")
        if run.structural_executions + run.cache_hits != budget:
            raise ValueError(f"seed {seed} result accounting differs")
        if (
            run.history != checkpoint.history
            or run.final_population != checkpoint.population
            or run.structural_executions != checkpoint.structural_executions
            or run.cache_hits != checkpoint.cache_hits
        ):
            raise ValueError(f"seed {seed} result does not reproduce its checkpoint")
        if len(run.final_results) != runner.policy.population_size:
            raise ValueError(f"seed {seed} final population is incomplete")
        if run.champion != min(checkpoint.cached_results, key=evaluation_order_key):
            raise ValueError(f"seed {seed} champion is not the best cached result")
        if not run.independently_reevaluated_champion_equal:
            raise ValueError(f"seed {seed} champion reevaluation differs")
        counts = status_counts(run.final_results)
        final_statuses.update(counts)
        records.append(
            {
                "seed": seed,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "result_sha256": sha256_file(result_path),
                "evaluation_requests": run.evaluation_requests,
                "structural_executions": run.structural_executions,
                "cache_hits": run.cache_hits,
                "remote_job_elapsed_seconds": _remote_elapsed(
                    campaign_directory / "logs" / f"seed_{seed:02d}.log"
                ),
                "champion_status": run.champion.status,
                "champion_cache_key": run.champion.cache_key,
                "champion_sections_by_group": run.champion.section_names_by_group,
                "champion_weight_kn": _required(
                    run.champion.structural_weight_n, name="weight", seed=seed
                )
                / 1000.0,
                "champion_objective": run.champion.objective,
                "final_population_status_counts": counts,
                "independently_reevaluated_champion_equal": True,
            }
        )
        write_convergence_svg(
            evidence_directory / f"convergence_seed_{seed:02d}.svg",
            run.history,
            title=f"Phase 12 {runner.policy.case_id} — seed {seed}",
        )
        runs.append(run)
    complete = tuple(runs)
    convergence_path = evidence_directory / "convergence_summary.csv"
    _write_convergence(convergence_path, complete, runner.policy.population_size)
    feasible = tuple(run for run in complete if run.champion.status == "feasible")
    best = (
        min(
            feasible,
            key=lambda run: _required(
                run.champion.objective.raw_objective if run.champion.objective else None,
                name="objective",
                seed=run.seed,
            ),
        )
        if feasible
        else None
    )
    all_weights = tuple(
        _required(run.champion.structural_weight_n, name="weight", seed=run.seed) / 1000.0
        for run in complete
    )
    feasible_weights = tuple(
        _required(run.champion.structural_weight_n, name="weight", seed=run.seed) / 1000.0
        for run in feasible
    )
    feasible_objectives = tuple(
        _required(
            run.champion.objective.raw_objective if run.champion.objective else None,
            name="objective",
            seed=run.seed,
        )
        for run in feasible
    )
    return {
        "schema_version": "1.0",
        "campaign_id": runner.policy.campaign_id,
        "campaign_sha256": runner.campaign_sha256,
        "evaluator_identity": runner.evaluator.identity,
        "policy_sha256": sha256_file(policy_path),
        "seed_count": len(complete),
        "evaluation_requests_total": sum(run.evaluation_requests for run in complete),
        "structural_executions_total": sum(run.structural_executions for run in complete),
        "cache_hits_total": sum(run.cache_hits for run in complete),
        "all_seed_accounting_exact": True,
        "all_champions_independently_reevaluated_equal": True,
        "seed_records": records,
        "aggregate": {
            "all_run_champion_weight_kn": {
                "best": min(all_weights),
                "mean": statistics.fmean(all_weights),
                "worst": max(all_weights),
            },
            "feasible_run_champion_weight_kn": (
                {
                    "best": min(feasible_weights),
                    "mean": statistics.fmean(feasible_weights),
                    "worst": max(feasible_weights),
                }
                if feasible_weights
                else None
            ),
            "best_feasible_objective": min(feasible_objectives) if feasible_objectives else None,
            "feasible_champion_count": len(feasible),
            "infeasible_or_failed_champion_count": len(complete) - len(feasible),
            "final_population_status_counts": dict(sorted(final_statuses.items())),
        },
        "best_feasible_design": (
            {
                "seed": best.seed,
                "cache_key": best.champion.cache_key,
                "genome": best.champion_genome,
                "sections_by_group": best.champion.section_names_by_group,
                "objective": best.champion.objective,
                "penalty": best.champion.penalty,
                "weight_kn": _required(
                    best.champion.structural_weight_n, name="weight", seed=best.seed
                )
                / 1000.0,
                "stage1": best.champion.stage1,
                "stage2": best.champion.stage2,
                "cp_story_displacements_m": best.champion.cp_story_displacements_m,
            }
            if best is not None
            else None
        ),
        "convergence_summary": {
            "path": convergence_path.name,
            "size_bytes": convergence_path.stat().st_size,
            "sha256": sha256_file(convergence_path),
        },
        "research_status": "verified research software; not professional engineering certification",
    }
