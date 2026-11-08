"""Run one identity-checked Phase 12 seed in the frozen Kaggle runtime.

The outer Kaggle process extracts and verifies the Python 3.12.6 runtime, then
starts this module inside that runtime.  The worker preserves the canonical GA
checkpoint after every epoch and emits an explicit success or failure record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path
from typing import Any, Final, cast

INPUT_ROOT: Final = Path("/kaggle/input")
WORKING_ROOT: Final = Path("/kaggle/working")
RUNTIME_ARCHIVE_NAME: Final = "phase12-python-3.12.6-runtime.tar.gz"
RUNTIME_SHA256: Final = "c382c0867130d68ba3ee81869f7cca382c09a34c3e0007712fe1fd2c731f755d"
EXPECTED_EPOCHS: Final = 200
EXPECTED_REQUESTS: Final = 20_000
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


def require_one(pattern: str, *, root: Path = INPUT_ROOT) -> Path:
    """Return the unique input path matching a recursive pattern."""

    matches = sorted(root.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern!r} under {root}, found {matches}")
    return matches[0]


def _load_manifest() -> dict[str, Any]:
    """Load the recovery manifest supplied with the canonical checkpoints."""

    path = require_one("phase12_recovery_manifest.json")
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise TypeError("recovery manifest must contain a JSON object")
    return cast(dict[str, Any], content)


def _project_root() -> Path:
    """Locate the unique frozen project tree mounted as a private dataset."""

    return require_one("uv.lock").parent


def _runtime_environment(project_root: Path) -> dict[str, str]:
    """Return deterministic environment variables for the runtime worker."""

    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(project_root / "src"),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    return environment


def _copy_recovery_checkpoint(
    *, manifest: dict[str, Any], campaign: str, seed: int, destination: Path
) -> str | None:
    """Copy and hash-check the canonical recovery checkpoint when one exists."""

    entries = manifest.get("checkpoints")
    if not isinstance(entries, list):
        raise TypeError("recovery manifest checkpoints must be a list")
    matching = [
        cast(dict[str, Any], entry)
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("campaign") == campaign
        and entry.get("seed") == seed
    ]
    if not matching:
        return None
    if len(matching) != 1:
        raise RuntimeError(f"manifest has duplicate recovery records for {campaign} seed {seed}")
    record = matching[0]
    expected = str(record["sha256"])
    source_name = f"seed_{seed:02d}.json"
    candidates = [
        path
        for path in INPUT_ROOT.rglob(source_name)
        if campaign in path.parts and "checkpoints" in path.parts
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one recovery checkpoint for {campaign} seed {seed}, found {candidates}"
        )
    source = candidates[0]
    actual = sha256_file(source)
    if actual != expected:
        raise RuntimeError(f"recovery checkpoint hash differs: {actual} != {expected}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return actual


def _run_worker(arguments: argparse.Namespace) -> int:
    """Resume or start one seed and write independently verifiable artifacts."""

    import steel_frame_optimization.optimization.campaign as campaign_module
    from steel_frame_optimization.optimization.campaign import (
        CampaignCheckpoint,
        CampaignRunner,
        SeedRunResult,
    )
    from steel_frame_optimization.optimization.evaluator import evaluation_order_key
    from steel_frame_optimization.serialization import write_canonical_json

    campaign = str(arguments.campaign)
    seed = int(arguments.seed)
    project_root = Path(arguments.project_root).resolve()
    output_root = Path(arguments.output_root).resolve()
    manifest = json.loads(Path(arguments.manifest).read_text(encoding="utf-8"))
    policy_path = project_root / "configs/optimization" / POLICY_NAMES[campaign]
    checkpoint_path = output_root / "checkpoints" / f"seed_{seed:02d}.json"
    result_path = output_root / "results" / f"seed_{seed:02d}.json"
    progress_path = output_root / "logs" / f"seed_{seed:02d}.log"
    status_path = output_root / "job_status.json"
    for directory in (checkpoint_path.parent, result_path.parent, progress_path.parent):
        directory.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    runner = CampaignRunner(project_root=project_root, policy_path=policy_path)
    campaign_record = cast(dict[str, Any], manifest["campaigns"][campaign])
    actual_identity = runner.evaluator.identity.model_dump(mode="json")
    if runner.campaign_sha256 != campaign_record["campaign_sha256"]:
        raise ValueError("frozen campaign identity differs from recovery manifest")
    if actual_identity != campaign_record["evaluator_identity"]:
        raise ValueError("frozen evaluator identity differs from recovery manifest")

    original_write = campaign_module._atomic_write_checkpoint

    def durable_write(path: Path, checkpoint: Any) -> None:
        """Atomically save each epoch and append compact local progress."""

        original_write(path, checkpoint)
        with progress_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(
                f"epoch={checkpoint.completed_epochs} "
                f"requests={checkpoint.evaluation_requests} "
                f"executions={checkpoint.structural_executions} "
                f"hits={checkpoint.cache_hits}\n"
            )

    campaign_module._atomic_write_checkpoint = durable_write
    try:
        run = runner.run_seed(
            seed=seed,
            checkpoint_path=checkpoint_path,
            resume=checkpoint_path.is_file(),
        )
    finally:
        campaign_module._atomic_write_checkpoint = original_write

    checkpoint = CampaignCheckpoint.model_validate_json(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.completed_epochs != EXPECTED_EPOCHS:
        raise RuntimeError("seed checkpoint did not reach 200 evaluated epochs")
    if checkpoint.evaluation_requests != EXPECTED_REQUESTS:
        raise RuntimeError("seed checkpoint did not reach exactly 20,000 requests")
    if checkpoint.structural_executions + checkpoint.cache_hits != EXPECTED_REQUESTS:
        raise RuntimeError("seed checkpoint accounting is not exact")
    if run.history != checkpoint.history or run.final_population != checkpoint.population:
        raise RuntimeError("seed result does not reproduce checkpoint state")
    if run.champion != min(checkpoint.cached_results, key=evaluation_order_key):
        raise RuntimeError("seed champion differs from checkpoint cache ordering")
    if not run.independently_reevaluated_champion_equal:
        raise RuntimeError("independent champion reevaluation differs")
    write_canonical_json(result_path, run)
    saved = SeedRunResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    if saved != run:
        raise RuntimeError("persisted seed result differs from in-memory result")
    status = {
        "schema_version": "1.0",
        "success": True,
        "campaign": campaign,
        "seed": seed,
        "python_version": sys.version.split()[0],
        "campaign_sha256": runner.campaign_sha256,
        "evaluator_identity": actual_identity,
        "completed_epochs": checkpoint.completed_epochs,
        "evaluation_requests": checkpoint.evaluation_requests,
        "structural_executions": checkpoint.structural_executions,
        "cache_hits": checkpoint.cache_hits,
        "independently_reevaluated_champion_equal": True,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "result_sha256": sha256_file(result_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(status, sort_keys=True))
    return 0


def run_job(*, campaign: str, seed: int) -> None:
    """Verify inputs, invoke the Python 3.12.6 worker, and preserve diagnostics."""

    if campaign not in POLICY_NAMES:
        raise ValueError(f"unknown Phase 12 campaign: {campaign}")
    if seed not in range(30):
        raise ValueError("seed must be between 0 and 29")
    output_root = WORKING_ROOT / "phase12" / campaign
    output_root.mkdir(parents=True, exist_ok=True)
    wrapper_status = output_root / "wrapper_status.json"
    runtime_root = WORKING_ROOT / "runtime"
    try:
        runtime_archive = require_one(RUNTIME_ARCHIVE_NAME)
        actual_runtime_hash = sha256_file(runtime_archive)
        if actual_runtime_hash != RUNTIME_SHA256:
            raise RuntimeError(
                f"runtime archive hash differs: {actual_runtime_hash} != {RUNTIME_SHA256}"
            )
        with tarfile.open(runtime_archive, mode="r:gz") as archive:
            archive.extractall(WORKING_ROOT, filter="fully_trusted")
        project_root = _project_root()
        manifest_path = require_one("phase12_recovery_manifest.json")
        manifest = _load_manifest()
        checkpoint_path = output_root / "checkpoints" / f"seed_{seed:02d}.json"
        recovery_hash = _copy_recovery_checkpoint(
            manifest=manifest,
            campaign=campaign,
            seed=seed,
            destination=checkpoint_path,
        )
        runtime_python = runtime_root / "venv/bin/python"
        completed = subprocess.run(
            [
                str(runtime_python),
                str(Path(__file__).resolve()),
                "--worker",
                "--campaign",
                campaign,
                "--seed",
                str(seed),
                "--project-root",
                str(project_root),
                "--output-root",
                str(output_root),
                "--manifest",
                str(manifest_path),
            ],
            cwd=project_root,
            env=_runtime_environment(project_root),
            text=True,
            check=False,
        )
        job_status_path = output_root / "job_status.json"
        success = completed.returncode == 0 and job_status_path.is_file()
        wrapper = {
            "schema_version": "1.0",
            "success": success,
            "campaign": campaign,
            "seed": seed,
            "worker_return_code": completed.returncode,
            "recovery_checkpoint_sha256": recovery_hash,
            "runtime_archive_sha256": actual_runtime_hash,
        }
        wrapper_status.write_text(
            json.dumps(wrapper, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if not success:
            raise RuntimeError(f"seed worker failed with return code {completed.returncode}")
    except Exception:
        error = traceback.format_exc()
        (output_root / "wrapper_error.txt").write_text(error, encoding="utf-8")
        print(error)
    finally:
        shutil.rmtree(runtime_root, ignore_errors=True)


def _parse_arguments() -> argparse.Namespace:
    """Parse the private runtime-worker command line."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--campaign", choices=tuple(POLICY_NAMES), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        parsed = _parse_arguments()
        if not parsed.worker:
            raise ValueError("the module command line is reserved for the frozen worker")
        raise SystemExit(_run_worker(parsed))
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        raise
