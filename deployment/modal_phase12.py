"""Run three bounded, checkpointed Phase 12 nine-story campaigns on Modal CPU."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import modal

LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/project")
VOLUME_ROOT = Path("/volume")


def _load_backend_policy() -> dict[str, Any]:
    """Load the backend policy from the checkout or mounted remote project tree."""

    relative = Path("configs/optimization/phase12_modal_backend.json")
    for root in (LOCAL_ROOT, REMOTE_ROOT):
        path = root / relative
        if path.is_file():
            return dict(json.loads(path.read_text(encoding="utf-8")))
    raise FileNotFoundError(
        "phase12_modal_backend.json was not found under the local checkout or /project"
    )


BACKEND = _load_backend_policy()
CAMPAIGNS = {
    "case1_filtered": "phase12_case1_filtered_main.json",
    "case1_no_filter": "phase12_case1_no_filter_main.json",
    "case2": "phase12_case2_main.json",
}

app = modal.App(str(BACKEND["app_name"]))
volume = modal.Volume.from_name(str(BACKEND["volume_name"]), create_if_missing=True, version=2)
image = (
    modal.Image.from_registry(f"python:{BACKEND['python_version']}-slim-bookworm")
    .uv_pip_install(
        "annotated-types==0.8.0",
        "contourpy==1.4.0",
        "cycler==0.12.1",
        "fonttools==4.67.0",
        "kiwisolver==1.5.1",
        "matplotlib==3.11.2",
        "numpy==2.5.4",
        "openseespy==3.8.0.0",
        "openseespylinux==3.8.0.0",
        "packaging==26.3",
        "pillow==13.0.0",
        "pydantic==2.14.1",
        "pydantic-core==2.50.1",
        "pyparsing==3.3.3",
        "python-dateutil==3.0.0",
        "six==1.17.0",
        "typing-extensions==4.16.0",
        "typing-inspection==0.4.4",
        "tzdata==2026.5",
        "xlrd==2.0.2",
    )
    .env(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    .add_local_dir(
        LOCAL_ROOT / "src/steel_frame_optimization",
        remote_path="/root/steel_frame_optimization",
        copy=True,
    )
    .add_local_dir(LOCAL_ROOT / "configs", remote_path="/project/configs", copy=True)
    .add_local_dir(LOCAL_ROOT / "data/processed", remote_path="/project/data/processed", copy=True)
    .add_local_dir(
        LOCAL_ROOT / "docs/evidence/phase11",
        remote_path="/project/docs/evidence/phase11",
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "deployment/finalize_phase12.py",
        remote_path="/project/deployment/finalize_phase12.py",
        copy=True,
    )
)


def _hash(path: Path) -> str:
    """Return one streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _campaign_policy(key: str) -> Path:
    """Return the strict remote campaign policy for one public key."""

    try:
        name = CAMPAIGNS[key]
    except KeyError as error:
        raise ValueError(f"unknown Phase 12 campaign: {key}") from error
    return REMOTE_ROOT / "configs/optimization" / name


def _worst_cost(seed_count: int) -> float:
    """Calculate the configured timeout-bound compute cost for one submission."""

    seconds = int(BACKEND["seed_timeout_seconds"])
    cpu = (
        float(BACKEND["container_cpu_cores"])
        * float(BACKEND["cpu_cost_usd_per_core_second"])
        * seconds
    )
    memory = (
        int(BACKEND["container_memory_mib"])
        / 1024.0
        * float(BACKEND["memory_cost_usd_per_gib_second"])
        * seconds
    )
    return seed_count * (cpu + memory)


def _projected_experiment_cost() -> float:
    """Project all three campaigns from the measured warm recovery benchmark."""

    request_count = (
        int(BACKEND["campaign_count"])
        * int(BACKEND["seeds_per_campaign"])
        * int(BACKEND["evaluation_requests_per_seed"])
    )
    measured_cost = (
        request_count
        * float(BACKEND["measured_seconds_per_request"])
        * (
            float(BACKEND["container_cpu_cores"]) * float(BACKEND["cpu_cost_usd_per_core_second"])
            + int(BACKEND["container_memory_mib"])
            / 1024.0
            * float(BACKEND["memory_cost_usd_per_gib_second"])
        )
    )
    return measured_cost * float(BACKEND["cost_contingency_factor"])


@app.function(image=image, cpu=1.0, memory=512, timeout=300, max_containers=1)
def probe_runtime_paths() -> str:
    """Verify remote mounts and evaluator identity without running a structural evaluation."""

    from steel_frame_optimization.optimization.campaign import CampaignRunner
    from steel_frame_optimization.serialization import sha256_file

    backend_path = REMOTE_ROOT / "configs/optimization/phase12_modal_backend.json"
    policy_path = _campaign_policy("case2")
    runner = CampaignRunner(project_root=REMOTE_ROOT, policy_path=policy_path)
    return json.dumps(
        {
            "remote_project_root": str(REMOTE_ROOT),
            "backend_path": str(backend_path),
            "backend_sha256": sha256_file(backend_path),
            "campaign_policy_path": str(policy_path),
            "campaign_sha256": runner.campaign_sha256,
            "evaluator_identity": runner.evaluator.identity.model_dump(mode="json"),
            "volume_root": str(VOLUME_ROOT),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


@app.function(
    image=image,
    cpu=float(BACKEND["container_cpu_cores"]),
    memory=(int(BACKEND["container_memory_mib"]), int(BACKEND["container_memory_limit_mib"])),
    timeout=int(BACKEND["seed_timeout_seconds"]),
    retries=int(BACKEND["automatic_retries"]),
    max_containers=int(BACKEND["maximum_parallel_seeds"]),
    volumes={str(VOLUME_ROOT): volume},
)
def run_seed(campaign_key: str, seed: int, maximum_epochs: int | None = None) -> str:
    """Resume one isolated seed and commit every epoch before returning."""

    import steel_frame_optimization.optimization.campaign as campaign_module
    from steel_frame_optimization.optimization.campaign import (
        CampaignCheckpoint,
        CampaignRunner,
        SeedRunResult,
    )
    from steel_frame_optimization.serialization import write_canonical_json

    if seed not in range(30):
        raise ValueError("seed must be between 0 and 29")
    policy_path = _campaign_policy(campaign_key)
    base = VOLUME_ROOT / campaign_key
    checkpoint = base / "checkpoints" / f"seed_{seed:02d}.json"
    result_path = base / "results" / f"seed_{seed:02d}.json"
    log_path = base / "logs" / f"seed_{seed:02d}.log"
    for directory in (checkpoint.parent, result_path.parent, log_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    runner = CampaignRunner(project_root=REMOTE_ROOT, policy_path=policy_path)
    if result_path.exists() and maximum_epochs is None:
        saved = SeedRunResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        state = CampaignCheckpoint.model_validate_json(checkpoint.read_text(encoding="utf-8"))
        if state.campaign_sha256 != runner.campaign_sha256:
            raise ValueError("completed checkpoint campaign identity differs")
        if state.evaluator_identity != runner.evaluator.identity:
            raise ValueError("completed checkpoint evaluator identity differs")
        if saved.evaluation_requests != runner.policy.evaluation_request_budget_per_seed:
            raise ValueError("completed result request budget differs")
        return json.dumps(
            {
                "campaign": campaign_key,
                "seed": seed,
                "reused": True,
                "result_sha256": _hash(result_path),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    original_write = campaign_module._atomic_write_checkpoint

    def durable_write(path: Path, state: Any) -> None:
        """Commit the exact checkpoint and a compact progress line."""

        original_write(path, state)
        with log_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(
                f"epoch={state.completed_epochs} requests={state.evaluation_requests} "
                f"executions={state.structural_executions} hits={state.cache_hits}\n"
            )
        volume.commit()

    campaign_module._atomic_write_checkpoint = durable_write
    started = time.perf_counter()
    try:
        run = runner.run_seed(
            seed=seed,
            checkpoint_path=checkpoint,
            resume=checkpoint.exists(),
            maximum_epochs=maximum_epochs,
        )
        if maximum_epochs is None:
            write_canonical_json(result_path, run)
        with log_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(
                f"complete seed={seed} elapsed_seconds={time.perf_counter() - started:.9f} "
                f"requests={run.evaluation_requests} executions={run.structural_executions} "
                f"hits={run.cache_hits} champion={run.champion.cache_key}\n"
            )
        volume.commit()
    finally:
        campaign_module._atomic_write_checkpoint = original_write
    return json.dumps(
        {
            "campaign": campaign_key,
            "seed": seed,
            "maximum_epochs": maximum_epochs,
            "elapsed_seconds": time.perf_counter() - started,
            "evaluation_requests": run.evaluation_requests,
            "structural_executions": run.structural_executions,
            "cache_hits": run.cache_hits,
            "champion_status": run.champion.status,
            "checkpoint_sha256": _hash(checkpoint),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


@app.function(
    image=image,
    cpu=1.0,
    memory=(1024, 2048),
    timeout=1800,
    max_containers=1,
    volumes={str(VOLUME_ROOT): volume},
)
def finalize_campaign(campaign_key: str) -> str:
    """Validate a complete campaign and persist its compact closeout evidence."""

    sys.path.insert(0, str(REMOTE_ROOT))
    from deployment.finalize_phase12 import build_phase12_campaign_summary
    from steel_frame_optimization.serialization import write_canonical_json

    evidence = VOLUME_ROOT / campaign_key / "evidence"
    summary = build_phase12_campaign_summary(
        project_root=REMOTE_ROOT,
        policy_path=_campaign_policy(campaign_key),
        campaign_directory=VOLUME_ROOT / campaign_key,
        evidence_directory=evidence,
    )
    output = evidence / "campaign_summary.json"
    write_canonical_json(output, summary)
    volume.commit()
    return json.dumps(
        {
            "campaign": campaign_key,
            "summary_sha256": _hash(output),
            "aggregate": summary["aggregate"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


@app.local_entrypoint()
def probe() -> None:
    """Print the read-only remote mount and evaluator-identity probe."""

    print(probe_runtime_paths.remote())


@app.local_entrypoint()
def benchmark(campaign_key: str = "case2") -> None:
    """Run one durable epoch for seed zero as the pre-campaign cloud gate."""

    print(run_seed.remote(campaign_key, 0, 1))


@app.local_entrypoint()
def campaign(campaign_key: str, start_seed: int = 0, end_seed: int = 4) -> None:
    """Submit one bounded seed range without waiting for its completion."""

    if campaign_key not in CAMPAIGNS:
        raise ValueError(f"unknown campaign: {campaign_key}")
    if start_seed < 0 or end_seed > 30 or start_seed >= end_seed:
        raise ValueError("range must satisfy 0 <= start < end <= 30")
    experiment_cost = _projected_experiment_cost()
    authorized_total = float(BACKEND["maximum_authorized_total_compute_cost_usd"])
    if experiment_cost > authorized_total:
        raise RuntimeError(
            f"full Phase 12 projection ${experiment_cost:.4f} exceeds the authorized "
            f"aggregate cloud limit ${authorized_total:.4f}; no production seed was submitted"
        )
    cost = _worst_cost(end_seed - start_seed)
    if cost > float(BACKEND["maximum_projected_compute_cost_usd_per_submission"]):
        raise RuntimeError(f"timeout-bound projection ${cost:.4f} exceeds the submission cap")
    print(f"timeout_bound_projection_usd={cost:.6f}")
    seed_range = range(start_seed, end_seed)
    run_seed.spawn_map(
        [campaign_key] * len(seed_range),
        seed_range,
        [None] * len(seed_range),
    )
    print(f"submitted={campaign_key}[{start_seed},{end_seed})")


@app.local_entrypoint()
def finalize(campaign_key: str) -> None:
    """Finalize one complete campaign after all 30 seeds are present."""

    print(finalize_campaign.remote(campaign_key))
