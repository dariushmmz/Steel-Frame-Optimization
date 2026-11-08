"""Run one bounded Phase 12 GA chunk on a GitHub-hosted CPU runner.

Each invocation advances at most 40 epochs, validates the frozen campaign and
evaluator identities, and emits a canonical checkpoint that a later job can
resume.  A terminal result is emitted only after all 200 epochs are complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Final, cast

from steel_frame_optimization.optimization.campaign import (
    CampaignCheckpoint,
    CampaignRunner,
    SeedRunResult,
)
from steel_frame_optimization.optimization.evaluator import evaluation_order_key
from steel_frame_optimization.serialization import write_canonical_json

EXPECTED_EPOCHS: Final = 200
EXPECTED_REQUESTS: Final = 20_000
MAXIMUM_CHUNK_EPOCHS: Final = 40
POLICY_NAMES: Final = {
    "case1_filtered": "phase12_case1_filtered_main.json",
    "case1_no_filter": "phase12_case1_no_filter_main.json",
    "case2": "phase12_case2_main.json",
}


def sha256_file(path: Path) -> str:
    """Return the streaming SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object with a clear type error for malformed inputs."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], value)


def run_chunk(arguments: argparse.Namespace) -> dict[str, Any]:
    """Advance one identity-checked seed to the requested bounded epoch."""

    project_root = arguments.project_root.resolve()
    output_root = arguments.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / "checkpoint.json"
    if arguments.checkpoint_input is not None:
        source = arguments.checkpoint_input.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"prior checkpoint not found: {source}")
        shutil.copy2(source, checkpoint_path)

    identity = _load_json(arguments.identity)
    campaign_identity = cast(dict[str, Any], identity["campaigns"])[arguments.campaign]
    policy_path = project_root / "configs/optimization" / POLICY_NAMES[arguments.campaign]
    runner = CampaignRunner(project_root=project_root, policy_path=policy_path)
    if runner.campaign_sha256 != campaign_identity["campaign_sha256"]:
        raise ValueError("frozen campaign identity differs from GitHub execution policy")
    actual_identity = runner.evaluator.identity.model_dump(mode="json")
    if actual_identity["implementation_sha256"] != campaign_identity["implementation_sha256"]:
        raise ValueError("frozen implementation identity differs from GitHub execution policy")
    if actual_identity["sections_sha256"] != identity["sections_sha256"]:
        raise ValueError("section-domain identity differs from GitHub execution policy")

    starting_epoch = 0
    if checkpoint_path.is_file():
        restored = CampaignCheckpoint.model_validate_json(
            checkpoint_path.read_text(encoding="utf-8")
        )
        starting_epoch = restored.completed_epochs
        if restored.campaign_sha256 != runner.campaign_sha256:
            raise ValueError("prior checkpoint campaign identity differs")
        if restored.evaluator_identity != runner.evaluator.identity:
            raise ValueError("prior checkpoint evaluator identity differs")
        if restored.seed != arguments.seed:
            raise ValueError("prior checkpoint seed differs")
    if arguments.target_epochs <= starting_epoch:
        raise ValueError("target epoch must advance the checkpoint")
    if arguments.target_epochs - starting_epoch > MAXIMUM_CHUNK_EPOCHS:
        raise ValueError("one GitHub job may advance at most 40 epochs")
    if arguments.target_epochs > EXPECTED_EPOCHS:
        raise ValueError("target epoch exceeds the frozen 200-epoch budget")

    started = time.perf_counter()
    run = runner.run_seed(
        seed=arguments.seed,
        checkpoint_path=checkpoint_path,
        resume=checkpoint_path.is_file(),
        maximum_epochs=arguments.target_epochs,
    )
    checkpoint = CampaignCheckpoint.model_validate_json(checkpoint_path.read_text(encoding="utf-8"))
    expected_requests = arguments.target_epochs * runner.policy.population_size
    if checkpoint.completed_epochs != arguments.target_epochs:
        raise RuntimeError("checkpoint did not reach the requested target epoch")
    if checkpoint.evaluation_requests != expected_requests:
        raise RuntimeError("checkpoint request accounting differs from the target epoch")
    if checkpoint.structural_executions + checkpoint.cache_hits != expected_requests:
        raise RuntimeError("checkpoint execution and cache accounting is not exact")
    if run.history != checkpoint.history or run.final_population != checkpoint.population:
        raise RuntimeError("chunk result does not reproduce the checkpoint state")
    champion = min(checkpoint.cached_results, key=evaluation_order_key)
    if run.champion != champion or not run.independently_reevaluated_champion_equal:
        raise RuntimeError("independent chunk champion verification failed")

    terminal = arguments.target_epochs == EXPECTED_EPOCHS
    result_path = output_root / "result.json"
    if terminal:
        if run.evaluation_requests != EXPECTED_REQUESTS:
            raise RuntimeError("terminal result does not contain exactly 20,000 requests")
        write_canonical_json(result_path, run)
        saved = SeedRunResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        if saved != run:
            raise RuntimeError("persisted terminal result differs from in-memory result")
    status = {
        "schema_version": "1.0",
        "success": True,
        "platform": "github_actions",
        "assignment_token": arguments.assignment_token,
        "campaign": arguments.campaign,
        "seed": arguments.seed,
        "python_version": sys.version.split()[0],
        "campaign_sha256": runner.campaign_sha256,
        "evaluator_identity": actual_identity,
        "starting_epoch": starting_epoch,
        "completed_epochs": checkpoint.completed_epochs,
        "evaluation_requests": checkpoint.evaluation_requests,
        "structural_executions": checkpoint.structural_executions,
        "cache_hits": checkpoint.cache_hits,
        "independently_reevaluated_champion_equal": True,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "result_sha256": sha256_file(result_path) if terminal else None,
        "terminal": terminal,
        "elapsed_seconds": time.perf_counter() - started,
    }
    status_path = output_root / "job_status.json"
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(status, sort_keys=True))
    return status


def parse_arguments() -> argparse.Namespace:
    """Parse the bounded GitHub chunk command line."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", choices=tuple(POLICY_NAMES), required=True)
    parser.add_argument("--seed", type=int, choices=range(30), required=True)
    parser.add_argument("--target-epochs", type=int, required=True)
    parser.add_argument("--assignment-token", required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--identity",
        type=Path,
        default=Path("configs/optimization/phase12_github_identity.json"),
    )
    parser.add_argument("--checkpoint-input", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run_chunk(parse_arguments())
