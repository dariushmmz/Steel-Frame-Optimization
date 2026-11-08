"""Resume one frozen Phase 12 GA seed with durable Google Drive checkpoints.

The outer process is evaluated by the Colab kernel.  It starts the worker with
the frozen Python 3.12.6 runtime; the worker persists an independently loaded
and recovery-tested checkpoint every ten epochs (1,000 evaluation requests).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Final, cast

EXPECTED_EPOCHS: Final = 200
EXPECTED_REQUESTS: Final = 20_000
DRIVE_INTERVAL_EPOCHS: Final = 10
RETAINED_CHECKPOINTS: Final = 2
POLICY_NAMES: Final = {
    "case1_filtered": "phase12_case1_filtered_main.json",
    "case1_no_filter": "phase12_case1_no_filter_main.json",
    "case2": "phase12_case2_main.json",
}


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_copy_verified(source: Path, destination: Path) -> str:
    """Copy a file atomically and prove that its bytes survived the copy."""

    source_hash = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    if sha256_file(temporary) != source_hash:
        raise RuntimeError(f"temporary Drive copy hash differs for {destination.name}")
    temporary.replace(destination)
    if sha256_file(destination) != source_hash:
        raise RuntimeError(f"persisted Drive copy hash differs for {destination.name}")
    return source_hash


def checkpoint_epoch(path: Path) -> int:
    """Return the epoch encoded in a durable checkpoint filename."""

    stem = path.stem
    marker = "_epoch_"
    if marker not in stem:
        raise ValueError(f"checkpoint filename does not encode an epoch: {path.name}")
    return int(stem.rsplit(marker, 1)[1])


def _run_worker(arguments: argparse.Namespace) -> int:
    """Run one seed and persist validated recovery state to Google Drive."""

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
    drive_root = Path(arguments.drive_root).resolve()
    source_checkpoint = (
        Path(arguments.source_checkpoint).resolve() if arguments.source_checkpoint else None
    )
    expected_source_sha256 = str(arguments.source_sha256 or "")
    manifest = json.loads(Path(arguments.manifest).read_text(encoding="utf-8"))
    campaign_root = drive_root / campaign
    recovery_root = campaign_root / "recovery" / f"seed_{seed:02d}"
    result_root = campaign_root / "results"
    evidence_root = campaign_root / "evidence"
    work_root = Path("/content/phase12-work") / campaign / f"seed_{seed:02d}"
    work_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = work_root / f"seed_{seed:02d}.json"
    policy_path = project_root / "configs/optimization" / POLICY_NAMES[campaign]

    runner = CampaignRunner(project_root=project_root, policy_path=policy_path)
    campaign_record = cast(dict[str, Any], manifest["campaigns"][campaign])
    actual_identity = runner.evaluator.identity.model_dump(mode="json")
    if runner.campaign_sha256 != campaign_record["campaign_sha256"]:
        raise ValueError("frozen campaign identity differs from recovery manifest")
    if actual_identity != campaign_record["evaluator_identity"]:
        raise ValueError("frozen evaluator identity differs from recovery manifest")

    def validate_checkpoint(path: Path) -> CampaignCheckpoint:
        """Load a checkpoint and verify identity, seed and exact accounting."""

        checkpoint = CampaignCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
        if checkpoint.campaign_sha256 != runner.campaign_sha256:
            raise ValueError("durable checkpoint campaign identity differs")
        if checkpoint.evaluator_identity != runner.evaluator.identity:
            raise ValueError("durable checkpoint evaluator identity differs")
        if checkpoint.seed != seed:
            raise ValueError("durable checkpoint seed differs")
        if checkpoint.evaluation_requests != checkpoint.completed_epochs * 100:
            raise ValueError("durable checkpoint request accounting differs")
        if (
            checkpoint.structural_executions + checkpoint.cache_hits
            != checkpoint.evaluation_requests
        ):
            raise ValueError("durable checkpoint execution accounting differs")
        return checkpoint

    def persist_checkpoint(path: Path, checkpoint: CampaignCheckpoint) -> Path:
        """Upload, reread, restore-test and rotate one durable checkpoint."""

        epoch = checkpoint.completed_epochs
        destination = recovery_root / f"seed_{seed:02d}_epoch_{epoch:03d}.json"
        checkpoint_hash = atomic_copy_verified(path, destination)
        persisted = validate_checkpoint(destination)
        restored = runner._restore_state(destination, seed)
        if restored[1] != persisted.population:
            raise RuntimeError("Drive checkpoint population recovery differs")
        if len(restored[6]) != len(persisted.cached_results):
            raise RuntimeError("Drive checkpoint cache recovery differs")
        if restored[7] != list(persisted.history):
            raise RuntimeError("Drive checkpoint history recovery differs")
        record = {
            "schema_version": "1.0",
            "campaign": campaign,
            "seed": seed,
            "completed_epochs": persisted.completed_epochs,
            "evaluation_requests": persisted.evaluation_requests,
            "structural_executions": persisted.structural_executions,
            "cache_hits": persisted.cache_hits,
            "checkpoint_path": str(destination),
            "checkpoint_bytes": destination.stat().st_size,
            "checkpoint_sha256": checkpoint_hash,
            "campaign_sha256": runner.campaign_sha256,
            "evaluator_identity": actual_identity,
            "recovery_probe_passed": True,
            "retention_count": RETAINED_CHECKPOINTS,
        }
        manifest_path = recovery_root / "latest.json"
        manifest_temporary = manifest_path.with_suffix(".json.tmp")
        manifest_temporary.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest_temporary.replace(manifest_path)
        if json.loads(manifest_path.read_text(encoding="utf-8")) != record:
            raise RuntimeError("Drive recovery manifest reread differs")

        candidates = sorted(
            recovery_root.glob(f"seed_{seed:02d}_epoch_*.json"), key=checkpoint_epoch
        )
        for obsolete in candidates[:-RETAINED_CHECKPOINTS]:
            obsolete.unlink()
        remaining = sorted(
            recovery_root.glob(f"seed_{seed:02d}_epoch_*.json"), key=checkpoint_epoch
        )
        if destination not in remaining or len(remaining) > RETAINED_CHECKPOINTS:
            raise RuntimeError("Drive checkpoint rotation invariant failed")
        return destination

    durable_candidates = sorted(
        recovery_root.glob(f"seed_{seed:02d}_epoch_*.json"), key=checkpoint_epoch
    )
    if durable_candidates:
        initial_source = durable_candidates[-1]
        initial = validate_checkpoint(initial_source)
        atomic_copy_verified(initial_source, checkpoint_path)
    elif arguments.start_fresh:
        initial = None
    else:
        if source_checkpoint is None:
            raise RuntimeError("a recovery checkpoint is required unless --start-fresh is used")
        if sha256_file(source_checkpoint) != expected_source_sha256:
            raise RuntimeError("source recovery checkpoint SHA-256 differs")
        source_state = validate_checkpoint(source_checkpoint)
        initial_source = persist_checkpoint(source_checkpoint, source_state)
        atomic_copy_verified(initial_source, checkpoint_path)
        initial = validate_checkpoint(checkpoint_path)

    original_write = campaign_module._atomic_write_checkpoint

    def durable_write(path: Path, checkpoint: CampaignCheckpoint) -> None:
        """Save every epoch locally and every ten epochs durably on Drive."""

        original_write(path, checkpoint)
        if (
            checkpoint.completed_epochs % DRIVE_INTERVAL_EPOCHS == 0
            or checkpoint.completed_epochs == EXPECTED_EPOCHS
        ):
            persist_checkpoint(path, checkpoint)

    started = time.perf_counter()
    campaign_module._atomic_write_checkpoint = durable_write
    try:
        run = runner.run_seed(
            seed=seed,
            checkpoint_path=checkpoint_path,
            resume=checkpoint_path.is_file(),
        )
    finally:
        campaign_module._atomic_write_checkpoint = original_write

    checkpoint = validate_checkpoint(checkpoint_path)
    if checkpoint.completed_epochs != EXPECTED_EPOCHS:
        raise RuntimeError("seed checkpoint did not reach 200 epochs")
    if checkpoint.evaluation_requests != EXPECTED_REQUESTS:
        raise RuntimeError("seed checkpoint did not reach exactly 20,000 requests")
    if run.history != checkpoint.history or run.final_population != checkpoint.population:
        raise RuntimeError("seed result does not reproduce checkpoint state")
    if run.champion != min(checkpoint.cached_results, key=evaluation_order_key):
        raise RuntimeError("seed champion differs from checkpoint cache ordering")
    if not run.independently_reevaluated_champion_equal:
        raise RuntimeError("independent champion reevaluation differs")

    local_result = work_root / f"seed_{seed:02d}_result.json"
    write_canonical_json(local_result, run)
    if SeedRunResult.model_validate_json(local_result.read_text(encoding="utf-8")) != run:
        raise RuntimeError("persisted seed result differs from in-memory result")
    result_path = result_root / f"seed_{seed:02d}.json"
    result_sha256 = atomic_copy_verified(local_result, result_path)
    if SeedRunResult.model_validate_json(result_path.read_text(encoding="utf-8")) != run:
        raise RuntimeError("Drive result differs from in-memory result")
    status = {
        "schema_version": "1.0",
        "success": True,
        "campaign": campaign,
        "seed": seed,
        "python_version": sys.version.split()[0],
        "campaign_sha256": runner.campaign_sha256,
        "evaluator_identity": actual_identity,
        "source_completed_epochs": 0 if initial is None else initial.completed_epochs,
        "completed_epochs": checkpoint.completed_epochs,
        "evaluation_requests": checkpoint.evaluation_requests,
        "structural_executions": checkpoint.structural_executions,
        "cache_hits": checkpoint.cache_hits,
        "independently_reevaluated_champion_equal": True,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "result_sha256": result_sha256,
        "elapsed_seconds": time.perf_counter() - started,
        "drive_interval_evaluation_requests": DRIVE_INTERVAL_EPOCHS * 100,
        "retained_checkpoint_count": RETAINED_CHECKPOINTS,
    }
    evidence_root.mkdir(parents=True, exist_ok=True)
    status_path = evidence_root / f"seed_{seed:02d}_status.json"
    status_temporary = status_path.with_suffix(".json.tmp")
    status_temporary.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    status_temporary.replace(status_path)
    print(json.dumps(status, sort_keys=True))
    return 0


def _worker_parser() -> argparse.ArgumentParser:
    """Build the private frozen-worker argument parser."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--campaign", choices=tuple(POLICY_NAMES), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--source-sha256")
    parser.add_argument("--start-fresh", action="store_true")
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def run_outer() -> None:
    """Validate the Colab request, create a duplicate guard and start the worker."""

    campaign = os.environ["PHASE12_CAMPAIGN"]
    seed = int(os.environ["PHASE12_SEED"])
    start_fresh = os.environ.get("PHASE12_START_FRESH", "0") == "1"
    source_checkpoint = os.environ.get("PHASE12_SOURCE_CHECKPOINT", "")
    source_sha256 = os.environ.get("PHASE12_SOURCE_SHA256", "")
    recovery_run = os.environ.get("PHASE12_RECOVERY_RUN", "0") == "1"
    manifest = Path(os.environ["PHASE12_RECOVERY_MANIFEST"])
    drive_root = Path(
        os.environ.get(
            "PHASE12_DRIVE_ROOT", "/content/drive/MyDrive/steel-frame-phase12-v1/campaigns"
        )
    )
    project_root = Path("/content/phase12-project")
    runtime_python = Path("/kaggle/working/runtime/venv/bin/python")
    if campaign not in POLICY_NAMES or seed not in range(30):
        raise ValueError("invalid Phase 12 campaign or seed")
    if not runtime_python.exists() or not project_root.is_dir():
        raise RuntimeError("frozen runtime or project tree is unavailable")

    lease_root = drive_root / campaign / "leases"
    lease_root.mkdir(parents=True, exist_ok=True)
    lease_path = lease_root / f"seed_{seed:02d}.json"
    lease = {
        "schema_version": "1.0",
        "campaign": campaign,
        "seed": seed,
        "run_token": uuid.uuid4().hex,
        "state": "running",
    }
    try:
        descriptor = os.open(lease_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        existing = json.loads(lease_path.read_text(encoding="utf-8"))
        if not recovery_run or existing.get("campaign") != campaign or existing.get("seed") != seed:
            raise RuntimeError(f"duplicate guard already exists: {existing}") from error
        lease["replaces_run_token"] = existing.get("run_token")
        lease_temporary = lease_path.with_suffix(".json.recovery.tmp")
        lease_temporary.write_text(
            json.dumps(lease, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        lease_temporary.replace(lease_path)
        descriptor = -1
    if descriptor >= 0:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(lease, stream, indent=2, sort_keys=True)
            stream.write("\n")

    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(project_root / "src"),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "MPLBACKEND": "Agg",
        }
    )
    worker_command = [
        str(runtime_python),
        str(Path("/content/colab_phase12_seed_job.py")),
        "--worker",
        "--campaign",
        campaign,
        "--seed",
        str(seed),
        "--project-root",
        str(project_root),
        "--drive-root",
        str(drive_root),
        "--source-checkpoint",
        source_checkpoint,
        "--source-sha256",
        source_sha256,
        "--manifest",
        str(manifest),
    ]
    if start_fresh:
        worker_command.append("--start-fresh")
    completed = subprocess.run(
        worker_command,
        cwd=project_root,
        env=environment,
        check=False,
        text=True,
    )
    lease["state"] = "completed" if completed.returncode == 0 else "failed"
    lease["worker_return_code"] = completed.returncode
    lease_temporary = lease_path.with_suffix(".json.tmp")
    lease_temporary.write_text(json.dumps(lease, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lease_temporary.replace(lease_path)
    if completed.returncode != 0:
        raise RuntimeError(f"Colab seed worker failed with return code {completed.returncode}")


if __name__ == "__main__":
    try:
        if "--worker" in sys.argv:
            parsed = _worker_parser().parse_args()
            raise SystemExit(_run_worker(parsed))
        run_outer()
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        raise
