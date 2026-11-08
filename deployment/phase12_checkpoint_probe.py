"""Run one bounded Phase 12 checkpoint recovery and emit comparison evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from steel_frame_optimization.optimization.campaign import (
    CampaignCheckpoint,
    CampaignRunner,
)
from steel_frame_optimization.optimization.genetics import serialize_rng_state
from steel_frame_optimization.serialization import sha256_file, write_canonical_json


def load_checkpoint(path: Path) -> CampaignCheckpoint:
    """Load one checkpoint through the strict production schema."""

    return CampaignCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded portability-probe command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--maximum-epochs", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    """Restore, advance, and summarize one identity-checked checkpoint."""

    arguments = build_parser().parse_args()
    checkpoint_path = arguments.checkpoint.resolve()
    initial_sha256 = sha256_file(checkpoint_path)
    initial = load_checkpoint(checkpoint_path)
    runner = CampaignRunner(
        project_root=arguments.project_root.resolve(),
        policy_path=arguments.policy.resolve(),
    )
    restored = runner._restore_state(checkpoint_path, arguments.seed)
    initial_keys = {item.cache_key for item in initial.cached_results}

    run = runner.run_seed(
        seed=arguments.seed,
        checkpoint_path=checkpoint_path,
        resume=True,
        maximum_epochs=arguments.maximum_epochs,
    )
    final = load_checkpoint(checkpoint_path)
    new_results = tuple(item for item in final.cached_results if item.cache_key not in initial_keys)
    evidence = {
        "schema_version": "1.0",
        "initial_checkpoint_sha256": initial_sha256,
        "final_checkpoint_sha256": sha256_file(checkpoint_path),
        "seed": arguments.seed,
        "initial_completed_epochs": initial.completed_epochs,
        "final_completed_epochs": final.completed_epochs,
        "initial_evaluation_requests": initial.evaluation_requests,
        "final_evaluation_requests": final.evaluation_requests,
        "final_structural_executions": final.structural_executions,
        "final_cache_hits": final.cache_hits,
        "campaign_identity_equal": initial.campaign_sha256 == runner.campaign_sha256,
        "evaluator_identity_equal": initial.evaluator_identity == runner.evaluator.identity,
        "population_restored_equal": restored[1] == initial.population,
        "rng_state_restored_equal": (
            serialize_rng_state(restored[0].rng) == initial.rng_state_json
        ),
        "cache_restored_equal": len(restored[6]) == len(initial.cached_results),
        "history_restored_equal": restored[7] == list(initial.history),
        "run_history_equal_checkpoint": run.history == final.history,
        "run_population_equal_checkpoint": run.final_population == final.population,
        "new_results": tuple(item.model_dump(mode="json") for item in new_results),
        "new_history": tuple(
            item.model_dump(mode="json") for item in final.history[initial.completed_epochs :]
        ),
    }
    write_canonical_json(arguments.output.resolve(), evidence)


if __name__ == "__main__":
    main()
