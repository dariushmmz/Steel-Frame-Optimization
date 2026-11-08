"""Validate and summarize the completed Phase 9 production campaign.

This script deliberately remains outside ``src/steel_frame_optimization``.  The
structural evaluator hashes every package module, so campaign-closeout tooling must
not change the implementation identity embedded in the running checkpoints.
"""

from __future__ import annotations

import argparse
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
from steel_frame_optimization.serialization import sha256_file, write_canonical_json

EXPECTED_SEEDS = tuple(range(30))
EXPECTED_EPOCHS = 200
EXPECTED_REQUESTS_PER_SEED = 10_000


def _required_float(value: float | None, *, name: str, seed: int) -> float:
    """Return one required finite campaign metric or reject an incomplete champion."""

    if value is None:
        raise ValueError(f"seed {seed} champion is missing {name}")
    return value


def _read_complete_log_elapsed_seconds(path: Path) -> float | None:
    """Read the durable remote-job elapsed time from one seed log when present."""

    if not path.is_file():
        return None
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if not line.startswith("complete seed="):
            continue
        fields = dict(item.split("=", 1) for item in line.split() if "=" in item)
        return float(fields["elapsed_seconds"])
    return None


def _write_convergence_summary(path: Path, runs: tuple[SeedRunResult, ...]) -> None:
    """Write deterministic per-epoch feasible-objective comparison data."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "epoch",
                "evaluation_requests_per_seed",
                "seeds_with_feasible_best",
                "minimum_best_feasible_objective",
                "mean_best_feasible_objective",
                "maximum_best_feasible_objective",
            )
        )
        for epoch in range(EXPECTED_EPOCHS):
            values = tuple(
                value
                for run in runs
                if (value := run.history[epoch].best_feasible_objective) is not None
            )
            writer.writerow(
                (
                    epoch + 1,
                    (epoch + 1) * 50,
                    len(values),
                    min(values) if values else "",
                    statistics.fmean(values) if values else "",
                    max(values) if values else "",
                )
            )


def build_campaign_summary(
    *,
    project_root: Path,
    policy_path: Path,
    checkpoint_directory: Path,
    result_directory: Path,
    log_directory: Path,
    evidence_directory: Path,
) -> dict[str, Any]:
    """Validate all identities, accounting, results, and champion reevaluations."""

    runner = CampaignRunner(project_root=project_root, policy_path=policy_path)
    if runner.policy.seeds != EXPECTED_SEEDS:
        raise ValueError("production policy does not declare exactly seeds 0 through 29")

    expected_identity = runner.evaluator.identity
    runs: list[SeedRunResult] = []
    seed_records: list[dict[str, Any]] = []
    final_population_statuses: Counter[str] = Counter()
    for seed in EXPECTED_SEEDS:
        checkpoint_path = checkpoint_directory / f"seed_{seed:02d}.json"
        result_path = result_directory / f"seed_{seed:02d}.json"
        if not checkpoint_path.is_file() or not result_path.is_file():
            raise FileNotFoundError(f"seed {seed} is missing its checkpoint or result")
        checkpoint = CampaignCheckpoint.model_validate_json(
            checkpoint_path.read_text(encoding="utf-8")
        )
        run = SeedRunResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        if checkpoint.seed != seed or run.seed != seed:
            raise ValueError(f"seed {seed} artifact contains the wrong seed identity")
        if checkpoint.campaign_sha256 != runner.campaign_sha256:
            raise ValueError(f"seed {seed} checkpoint campaign identity mismatch")
        if checkpoint.evaluator_identity != expected_identity:
            raise ValueError(f"seed {seed} checkpoint evaluator identity mismatch")
        if any(item.evaluator_identity != expected_identity for item in checkpoint.cached_results):
            raise ValueError(f"seed {seed} cache contains a foreign evaluator identity")
        if (
            checkpoint.completed_epochs != EXPECTED_EPOCHS
            or run.completed_epochs != EXPECTED_EPOCHS
        ):
            raise ValueError(f"seed {seed} did not complete exactly {EXPECTED_EPOCHS} epochs")
        if (
            checkpoint.evaluation_requests != EXPECTED_REQUESTS_PER_SEED
            or run.evaluation_requests != EXPECTED_REQUESTS_PER_SEED
        ):
            raise ValueError(f"seed {seed} did not execute exactly 10,000 requests")
        if checkpoint.structural_executions + checkpoint.cache_hits != EXPECTED_REQUESTS_PER_SEED:
            raise ValueError(f"seed {seed} checkpoint accounting is not exact")
        if run.structural_executions + run.cache_hits != EXPECTED_REQUESTS_PER_SEED:
            raise ValueError(f"seed {seed} result accounting is not exact")
        if (
            run.structural_executions != checkpoint.structural_executions
            or run.cache_hits != checkpoint.cache_hits
            or run.history != checkpoint.history
            or run.final_population != checkpoint.population
        ):
            raise ValueError(f"seed {seed} result does not reproduce checkpoint state")
        if len(run.history) != EXPECTED_EPOCHS or len(run.final_results) != 50:
            raise ValueError(f"seed {seed} has incomplete convergence or final-population data")
        expected_champion = min(checkpoint.cached_results, key=evaluation_order_key)
        if run.champion != expected_champion:
            raise ValueError(f"seed {seed} champion is not the best cached evaluation")
        if not run.independently_reevaluated_champion_equal:
            raise ValueError(f"seed {seed} independent champion reevaluation differs")

        objective = run.champion.objective
        final_counts = status_counts(run.final_results)
        final_population_statuses.update(final_counts)
        seed_records.append(
            {
                "seed": seed,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "result_sha256": sha256_file(result_path),
                "evaluation_requests": run.evaluation_requests,
                "structural_executions": run.structural_executions,
                "cache_hits": run.cache_hits,
                "unique_cached_results": len(checkpoint.cached_results),
                "remote_job_elapsed_seconds": _read_complete_log_elapsed_seconds(
                    log_directory / f"seed_{seed:02d}.log"
                ),
                "champion_status": run.champion.status,
                "champion_cache_key": run.champion.cache_key,
                "champion_genome": run.champion_genome,
                "champion_sections_by_group": run.champion.section_names_by_group,
                "champion_weight_kn": _required_float(
                    run.champion.structural_weight_n, name="weight", seed=seed
                )
                / 1000.0,
                "champion_objective": objective,
                "champion_penalty": run.champion.penalty,
                "final_population_status_counts": final_counts,
                "independently_reevaluated_champion_equal": True,
            }
        )
        write_convergence_svg(
            evidence_directory / f"main_convergence_seed_{seed:02d}.svg",
            run.history,
            title=f"Phase 9 main campaign — seed {seed}",
        )
        runs.append(run)

    complete_runs = tuple(runs)
    _write_convergence_summary(evidence_directory / "main_convergence_summary.csv", complete_runs)
    champion_weights = tuple(
        _required_float(run.champion.structural_weight_n, name="weight", seed=run.seed) / 1000.0
        for run in complete_runs
    )
    feasible_runs = tuple(run for run in complete_runs if run.champion.status == "feasible")
    feasible_weights = tuple(
        _required_float(run.champion.structural_weight_n, name="weight", seed=run.seed) / 1000.0
        for run in feasible_runs
    )
    feasible_objectives = tuple(
        _required_float(
            run.champion.objective.raw_objective if run.champion.objective else None,
            name="objective",
            seed=run.seed,
        )
        for run in feasible_runs
    )
    best_run = (
        min(
            feasible_runs,
            key=lambda run: _required_float(
                run.champion.objective.raw_objective if run.champion.objective else None,
                name="objective",
                seed=run.seed,
            ),
        )
        if feasible_runs
        else None
    )
    plot_paths = tuple(
        evidence_directory / f"main_convergence_seed_{seed:02d}.svg" for seed in EXPECTED_SEEDS
    )
    return {
        "schema_version": "1.0",
        "campaign_id": runner.policy.campaign_id,
        "campaign_sha256": runner.campaign_sha256,
        "evaluator_identity": expected_identity,
        "policy_sha256": sha256_file(policy_path),
        "seed_count": len(complete_runs),
        "evaluation_requests_total": sum(run.evaluation_requests for run in complete_runs),
        "structural_executions_total": sum(run.structural_executions for run in complete_runs),
        "cache_hits_total": sum(run.cache_hits for run in complete_runs),
        "all_seed_accounting_exact": True,
        "all_champions_independently_reevaluated_equal": True,
        "seed_records": seed_records,
        "aggregate": {
            "all_run_champion_weight_kn": {
                "best": min(champion_weights),
                "mean": statistics.fmean(champion_weights),
                "worst": max(champion_weights),
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
            "feasible_champion_count": len(feasible_runs),
            "infeasible_or_failed_champion_count": len(complete_runs) - len(feasible_runs),
            "final_population_status_counts": dict(sorted(final_population_statuses.items())),
        },
        "best_feasible_design": (
            {
                "seed": best_run.seed,
                "cache_key": best_run.champion.cache_key,
                "genome": best_run.champion_genome,
                "sections_by_group": best_run.champion.section_names_by_group,
                "objective": best_run.champion.objective,
                "penalty": best_run.champion.penalty,
                "weight_kn": _required_float(
                    best_run.champion.structural_weight_n,
                    name="weight",
                    seed=best_run.seed,
                )
                / 1000.0,
                "stage1": best_run.champion.stage1,
                "stage2": best_run.champion.stage2,
                "cp_story_displacements_m": best_run.champion.cp_story_displacements_m,
            }
            if best_run is not None
            else None
        ),
        "convergence_artifacts": [
            {
                "path": path.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in plot_paths
        ]
        + [
            {
                "path": (evidence_directory / "main_convergence_summary.csv").as_posix(),
                "size_bytes": (evidence_directory / "main_convergence_summary.csv").stat().st_size,
                "sha256": sha256_file(evidence_directory / "main_convergence_summary.csv"),
            }
        ],
        "research_status": (
            "owner-approved 273-section project baseline; paper comparison remains diagnostic; "
            "not independently engineering-validated or professionally certified"
        ),
    }


def _parse_args() -> argparse.Namespace:
    """Parse explicit campaign artifact locations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Validate the campaign and write one canonical closeout artifact."""

    args = _parse_args()
    summary = build_campaign_summary(
        project_root=args.project_root.resolve(),
        policy_path=args.policy.resolve(),
        checkpoint_directory=args.checkpoints.resolve(),
        result_directory=args.results.resolve(),
        log_directory=args.logs.resolve(),
        evidence_directory=args.output.resolve().parent,
    )
    write_canonical_json(args.output, summary)
    aggregate = summary["aggregate"]
    print(
        "Phase 9 main campaign validated: "
        f"seeds={summary['seed_count']}, requests={summary['evaluation_requests_total']}, "
        f"feasible_champions={aggregate['feasible_champion_count']}, "
        f"best_objective={aggregate['best_feasible_objective']}"
    )


if __name__ == "__main__":
    main()
