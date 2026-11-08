"""Validate canonical Modal checkpoints and build a minimal Kaggle recovery bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Final

from steel_frame_optimization.optimization.campaign import CampaignCheckpoint, CampaignRunner
from steel_frame_optimization.serialization import write_canonical_json

SOURCE_PROJECT_COMMIT: Final = "cf628c32f61b72b76fe5627ade9c40adcb25357f"
POLICIES: Final = {
    "case1_filtered": "phase12_case1_filtered_main.json",
    "case1_no_filter": "phase12_case1_no_filter_main.json",
    "case2": "phase12_case2_main.json",
}
EXPECTED_PARTIAL_EPOCHS: Final = {
    "case1_filtered": {4: 149, 5: 94, 8: 82, 10: 124, 14: 49, 15: 33, 16: 21},
    "case1_no_filter": {6: 191, 10: 88, 11: 168, 12: 146, 18: 26, 20: 103, 25: 49},
    "case2": {9: 112, 10: 144, 13: 137, 18: 35, 22: 57, 27: 65},
}
FIXED_ZIP_TIMESTAMP: Final = (2026, 11, 6, 0, 0, 0)


def sha256_file(path: Path) -> str:
    """Return the streaming SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_zip_member(archive: zipfile.ZipFile, source: Path, name: str) -> None:
    """Write one deterministically timestamped member without loading it into memory."""

    information = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIMESTAMP)
    information.compress_type = zipfile.ZIP_DEFLATED
    information.external_attr = 0o100644 << 16
    with source.open("rb") as input_stream, archive.open(information, mode="w") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)


def _checkpoint_record(
    *, runner: CampaignRunner, campaign: str, seed: int, epochs: int, path: Path
) -> dict[str, Any]:
    """Strictly validate one recovery checkpoint and return its manifest record."""

    checkpoint = CampaignCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
    if checkpoint.campaign_sha256 != runner.campaign_sha256:
        raise ValueError(f"{campaign} seed {seed} campaign identity differs")
    if checkpoint.evaluator_identity != runner.evaluator.identity:
        raise ValueError(f"{campaign} seed {seed} evaluator identity differs")
    if checkpoint.seed != seed:
        raise ValueError(f"{campaign} seed {seed} seed field differs")
    if checkpoint.completed_epochs != epochs:
        raise ValueError(f"{campaign} seed {seed} expected epoch {epochs}")
    if checkpoint.evaluation_requests != epochs * runner.policy.population_size:
        raise ValueError(f"{campaign} seed {seed} request accounting differs")
    if checkpoint.structural_executions + checkpoint.cache_hits != checkpoint.evaluation_requests:
        raise ValueError(f"{campaign} seed {seed} execution accounting differs")
    if len(checkpoint.history) != epochs:
        raise ValueError(f"{campaign} seed {seed} history length differs")
    runner._restore_state(path, seed)
    return {
        "campaign": campaign,
        "seed": seed,
        "completed_epochs": checkpoint.completed_epochs,
        "evaluation_requests": checkpoint.evaluation_requests,
        "structural_executions": checkpoint.structural_executions,
        "cache_hits": checkpoint.cache_hits,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "bundle_path": f"checkpoints/{campaign}/{path.name}",
    }


def build_bundle(
    *, project_root: Path, checkpoint_root: Path, output_directory: Path
) -> tuple[Path, Path]:
    """Build and verify the minimal private Kaggle recovery ZIP and manifest."""

    output_directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    campaign_records: dict[str, Any] = {}
    files: list[tuple[Path, str]] = []
    for campaign, policy_name in POLICIES.items():
        runner = CampaignRunner(
            project_root=project_root,
            policy_path=project_root / "configs/optimization" / policy_name,
        )
        campaign_records[campaign] = {
            "policy": f"configs/optimization/{policy_name}",
            "campaign_sha256": runner.campaign_sha256,
            "evaluator_identity": runner.evaluator.identity.model_dump(mode="json"),
            "population_size": runner.policy.population_size,
            "evaluated_population_epochs": runner.policy.evaluated_population_epochs,
            "evaluation_request_budget_per_seed": (
                runner.policy.evaluation_request_budget_per_seed
            ),
        }
        for seed, epochs in EXPECTED_PARTIAL_EPOCHS[campaign].items():
            path = checkpoint_root / campaign / "checkpoints" / f"seed_{seed:02d}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            records.append(
                _checkpoint_record(
                    runner=runner,
                    campaign=campaign,
                    seed=seed,
                    epochs=epochs,
                    path=path,
                )
            )
            files.append((path, f"checkpoints/{campaign}/{path.name}"))
    manifest = {
        "schema_version": "1.0",
        "source": "Modal Volume steel-frame-phase12-v1 canonical checkpoints",
        "source_project_commit": SOURCE_PROJECT_COMMIT,
        "campaigns": campaign_records,
        "checkpoints": records,
        "checkpoint_count": len(records),
        "uncompressed_checkpoint_bytes": sum(int(record["size_bytes"]) for record in records),
    }
    manifest_path = output_directory / "phase12_recovery_manifest.json"
    write_canonical_json(manifest_path, manifest)
    job_script = project_root / "deployment/kaggle_phase12_seed_job.py"
    files.extend(
        (
            (manifest_path, manifest_path.name),
            (job_script, job_script.name),
        )
    )
    bundle_path = output_directory / "phase12_recovery_state.zip"
    with zipfile.ZipFile(
        bundle_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for source, name in files:
            _write_zip_member(archive, source, name)
    with zipfile.ZipFile(bundle_path, mode="r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"recovery bundle ZIP member is corrupt: {bad_member}")
        names = set(archive.namelist())
        if names != {name for _, name in files}:
            raise RuntimeError("recovery bundle member inventory differs")
    return manifest_path, bundle_path


def main() -> None:
    """Parse paths, build the recovery bundle, and print its hashes."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    manifest_path, bundle_path = build_bundle(
        project_root=arguments.project_root.resolve(),
        checkpoint_root=arguments.checkpoint_root.resolve(),
        output_directory=arguments.output_directory.resolve(),
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "bundle": str(bundle_path),
                "bundle_size_bytes": bundle_path.stat().st_size,
                "bundle_sha256": sha256_file(bundle_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
