"""Deploy the frozen Phase 9 campaign to bounded, checkpointed Modal CPU workers.

This module deliberately lives outside ``src/steel_frame_optimization`` so adding cloud
orchestration does not change the structural evaluator identity embedded in local checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import modal

LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_PROJECT_ROOT = Path("/project")
VOLUME_MOUNT = Path("/volume")


def _load_backend_policy() -> dict[str, Any]:
    """Load cloud policy from the local checkout or the copied remote project tree."""

    relative_path = Path("configs") / "optimization" / "phase9_modal_backend.json"
    for project_root in (LOCAL_ROOT, REMOTE_PROJECT_ROOT):
        path = project_root / relative_path
        if path.is_file():
            return dict(json.loads(path.read_text(encoding="utf-8")))
    raise FileNotFoundError(
        "phase9_modal_backend.json was not found in the local checkout or /project"
    )


BACKEND_POLICY = _load_backend_policy()
APP_NAME = str(BACKEND_POLICY["app_name"])
VOLUME_NAME = str(BACKEND_POLICY["volume_name"])

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)

image = (
    modal.Image.from_registry(f"python:{BACKEND_POLICY['python_version']}-slim-bookworm")
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
        LOCAL_ROOT / "src" / "steel_frame_optimization",
        remote_path="/root/steel_frame_optimization",
        copy=True,
    )
    .add_local_dir(LOCAL_ROOT / "configs", remote_path="/project/configs", copy=True)
    .add_local_dir(
        LOCAL_ROOT / "data" / "processed",
        remote_path="/project/data/processed",
        copy=True,
    )
    .add_local_dir(
        LOCAL_ROOT / "docs" / "evidence" / "phase05",
        remote_path="/project/docs/evidence/phase05",
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "docs" / "evidence" / "phase08" / "evaluation_pipeline.json",
        remote_path="/project/docs/evidence/phase08/evaluation_pipeline.json",
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "deployment" / "finalize_phase9.py",
        remote_path="/project/deployment/finalize_phase9.py",
        copy=True,
    )
)


def _sha256_file(path: Path) -> str:
    """Hash one local or mounted file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _worst_case_compute_cost(seed_count: int) -> float:
    """Project the maximum requested CPU and memory cost for a seed range."""

    cpu_cost = (
        float(BACKEND_POLICY["container_cpu_cores"])
        * float(BACKEND_POLICY["cpu_cost_usd_per_core_second"])
        * int(BACKEND_POLICY["seed_timeout_seconds"])
        * seed_count
    )
    memory_cost = (
        int(BACKEND_POLICY["container_memory_mib"])
        / 1024.0
        * float(BACKEND_POLICY["memory_cost_usd_per_gib_second"])
        * int(BACKEND_POLICY["seed_timeout_seconds"])
        * seed_count
    )
    return cpu_cost + memory_cost


def _load_representative_designs() -> list[dict[str, Any]]:
    """Load the three committed Phase 8 Linux comparison fixtures."""

    path = REMOTE_PROJECT_ROOT / "configs" / "cases" / "phase8_representative_designs.json"
    content = json.loads(path.read_text(encoding="utf-8"))
    designs = content.get("designs")
    if not isinstance(designs, list):
        raise ValueError("representative design configuration has no designs list")
    return designs


def _compare_json_values(
    actual: Any,
    expected: Any,
    *,
    path: str,
    relative_tolerance: float,
    differences: list[str],
) -> float:
    """Compare nested JSON values and return the largest relative numeric difference."""

    if isinstance(actual, bool) or isinstance(expected, bool):
        if actual != expected:
            differences.append(f"{path}: {actual!r} != {expected!r}")
        return 0.0
    if isinstance(actual, int | float) and isinstance(expected, int | float):
        actual_float = float(actual)
        expected_float = float(expected)
        scale = max(abs(expected_float), 1.0e-12)
        relative_difference = abs(actual_float - expected_float) / scale
        if not math.isclose(
            actual_float,
            expected_float,
            rel_tol=relative_tolerance,
            abs_tol=1.0e-12,
        ):
            differences.append(
                f"{path}: {actual_float:.17g} != {expected_float:.17g} "
                f"(relative_difference={relative_difference:.3e})"
            )
        return relative_difference
    if isinstance(actual, dict) and isinstance(expected, dict):
        maximum_difference = 0.0
        actual_keys = set(actual)
        expected_keys = set(expected)
        if actual_keys != expected_keys:
            differences.append(
                f"{path}: key difference actual_only={sorted(actual_keys - expected_keys)} "
                f"expected_only={sorted(expected_keys - actual_keys)}"
            )
        for key in sorted(actual_keys & expected_keys):
            maximum_difference = max(
                maximum_difference,
                _compare_json_values(
                    actual[key],
                    expected[key],
                    path=f"{path}.{key}",
                    relative_tolerance=relative_tolerance,
                    differences=differences,
                ),
            )
        return maximum_difference
    if isinstance(actual, list) and isinstance(expected, list):
        maximum_difference = 0.0
        if len(actual) != len(expected):
            differences.append(f"{path}: length {len(actual)} != {len(expected)}")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=False)):
            maximum_difference = max(
                maximum_difference,
                _compare_json_values(
                    actual_item,
                    expected_item,
                    path=f"{path}[{index}]",
                    relative_tolerance=relative_tolerance,
                    differences=differences,
                ),
            )
        return maximum_difference
    if actual != expected:
        differences.append(f"{path}: {actual!r} != {expected!r}")
    return 0.0


@app.function(
    image=image,
    cpu=1.0,
    memory=(512, 1024),
    timeout=900,
    max_containers=2,
    volumes={str(VOLUME_MOUNT): volume},
)
def probe_linux() -> str:
    """Evaluate representative paths once and report Linux identity and timing."""

    import numpy
    import openseespy
    import pydantic

    from steel_frame_optimization.optimization.evaluator import (
        EvaluatorPaths,
        StructuralEvaluator,
    )

    evaluator = StructuralEvaluator(EvaluatorPaths.from_project_root(REMOTE_PROJECT_ROOT))
    results: list[dict[str, Any]] = []
    for fixture in _load_representative_designs():
        design_id = str(fixture["design_id"])
        sections = {str(key): str(value) for key, value in fixture["sections_by_group"].items()}
        started = time.perf_counter()
        result = evaluator.evaluate(
            evaluation_id=f"modal-probe-{design_id}",
            section_names_by_group=sections,
            use_cache=False,
        ).result
        results.append(
            {
                "design_id": design_id,
                "elapsed_seconds": time.perf_counter() - started,
                "result": result.model_dump(mode="json"),
            }
        )
    baseline = json.loads(
        (
            REMOTE_PROJECT_ROOT / "docs" / "evidence" / "phase08" / "evaluation_pipeline.json"
        ).read_text(encoding="utf-8")
    )
    differences: list[str] = []
    maximum_relative_difference = 0.0
    for fixture, actual_record, expected_result in zip(
        _load_representative_designs(),
        results,
        baseline["results"],
        strict=True,
    ):
        expected_design_id = str(fixture["design_id"])
        if actual_record["design_id"] != expected_design_id:
            differences.append(
                f"design_id: {actual_record['design_id']!r} != {expected_design_id!r}"
            )
        maximum_relative_difference = max(
            maximum_relative_difference,
            _compare_json_values(
                actual_record["result"],
                expected_result,
                path=expected_design_id,
                relative_tolerance=1.0e-8,
                differences=differences,
            ),
        )
    identity_equal = evaluator.identity.model_dump(mode="json") == baseline["evaluator_identity"]
    if not identity_equal:
        differences.append("evaluator identity differs from committed Phase 8 baseline")
    payload = {
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": numpy.__version__,
        "pydantic": pydantic.__version__,
        "openseespy": getattr(openseespy, "__version__", "not-exposed"),
        "evaluator_identity": evaluator.identity.model_dump(mode="json"),
        "comparison": {
            "baseline": "docs/evidence/phase08/evaluation_pipeline.json",
            "relative_tolerance": 1.0e-8,
            "absolute_tolerance": 1.0e-12,
            "identity_equal": identity_equal,
            "all_results_equal_within_tolerance": not differences,
            "maximum_relative_difference": maximum_relative_difference,
            "differences": differences,
        },
        "results": results,
    }
    evidence_directory = VOLUME_MOUNT / "evidence"
    evidence_directory.mkdir(parents=True, exist_ok=True)
    evidence_path = evidence_directory / "linux_probe.json"
    evidence_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    volume.commit()
    return json.dumps(
        {
            "platform": payload["platform"],
            "python": payload["python"],
            "numpy": payload["numpy"],
            "pydantic": payload["pydantic"],
            "comparison": payload["comparison"],
            "evidence_sha256": _sha256_file(evidence_path),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


@app.function(
    image=image,
    cpu=2.0,
    memory=(768, 1536),
    timeout=1800,
    max_containers=1,
    volumes={str(VOLUME_MOUNT): volume},
)
def benchmark_pilot() -> str:
    """Run one bounded pilot seed to measure spawned-process Modal throughput."""

    from steel_frame_optimization.optimization.campaign import CampaignRunner

    runner = CampaignRunner(
        project_root=REMOTE_PROJECT_ROOT,
        policy_path=(REMOTE_PROJECT_ROOT / "configs" / "optimization" / "phase9_ga_pilot.json"),
    )
    with tempfile.TemporaryDirectory() as directory:
        started = time.perf_counter()
        result = runner.run_seed(
            seed=0,
            checkpoint_path=Path(directory) / "seed_00.json",
        )
        elapsed = time.perf_counter() - started
    payload = {
        "elapsed_seconds": elapsed,
        "evaluation_requests": result.evaluation_requests,
        "structural_executions": result.structural_executions,
        "cache_hits": result.cache_hits,
        "champion_status": result.champion.status,
        "champion_cache_key": result.champion.cache_key,
        "champion_objective": (
            result.champion.objective.raw_objective
            if result.champion.objective is not None
            else None
        ),
        "champion_reevaluation_equal": result.independently_reevaluated_champion_equal,
    }
    evidence_directory = VOLUME_MOUNT / "evidence"
    evidence_directory.mkdir(parents=True, exist_ok=True)
    evidence_path = evidence_directory / "bounded_pilot.json"
    evidence_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    volume.commit()
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )


@app.function(
    image=image,
    cpu=1.0,
    memory=(512, 1024),
    timeout=900,
    max_containers=1,
    volumes={str(VOLUME_MOUNT): volume},
)
def audit_volume_checkpoints() -> str:
    """Validate every uploaded checkpoint against the frozen remote identities."""

    from steel_frame_optimization.optimization.campaign import CampaignCheckpoint, CampaignRunner

    runner = CampaignRunner(
        project_root=REMOTE_PROJECT_ROOT,
        policy_path=(
            REMOTE_PROJECT_ROOT / "configs" / "optimization" / "phase9_three_story_main.json"
        ),
    )
    checkpoint_directory = VOLUME_MOUNT / "checkpoints"
    records: list[dict[str, Any]] = []
    for path in sorted(checkpoint_directory.glob("seed_*.json")):
        checkpoint = CampaignCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
        records.append(
            {
                "name": path.name,
                "sha256": _sha256_file(path),
                "seed": checkpoint.seed,
                "completed_epochs": checkpoint.completed_epochs,
                "evaluation_requests": checkpoint.evaluation_requests,
                "campaign_identity_equal": checkpoint.campaign_sha256 == runner.campaign_sha256,
                "evaluator_identity_equal": checkpoint.evaluator_identity
                == runner.evaluator.identity,
            }
        )
    return json.dumps(
        {
            "campaign_sha256": runner.campaign_sha256,
            "records": records,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


@app.function(
    image=image,
    cpu=1.0,
    memory=(512, 1024),
    timeout=900,
    max_containers=1,
    volumes={str(VOLUME_MOUNT): volume},
)
def probe_checkpoint_recovery() -> str:
    """Restore complete and partial checkpoints without executing structural analyses."""

    from steel_frame_optimization.optimization.campaign import (
        CampaignCheckpoint,
        CampaignRunner,
    )
    from steel_frame_optimization.optimization.genetics import serialize_rng_state

    runner = CampaignRunner(
        project_root=REMOTE_PROJECT_ROOT,
        policy_path=(
            REMOTE_PROJECT_ROOT / "configs" / "optimization" / "phase9_three_story_main.json"
        ),
    )
    records: list[dict[str, Any]] = []
    for seed in (0, 8):
        path = VOLUME_MOUNT / "checkpoints" / f"seed_{seed:02d}.json"
        checkpoint = CampaignCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
        (
            ga,
            population,
            completed_epochs,
            evaluation_requests,
            structural_executions,
            cache_hits,
            result_cache,
            history,
        ) = runner._restore_state(path, seed)
        records.append(
            {
                "seed": seed,
                "completed_epochs": completed_epochs,
                "evaluation_requests": evaluation_requests,
                "structural_executions": structural_executions,
                "cache_hits": cache_hits,
                "population_equal": population == checkpoint.population,
                "rng_state_equal": serialize_rng_state(ga.rng) == checkpoint.rng_state_json,
                "cache_size_equal": len(result_cache) == len(checkpoint.cached_results),
                "history_equal": tuple(history) == checkpoint.history,
                "structural_evaluations_executed": 0,
            }
        )
    return json.dumps(
        {"campaign_sha256": runner.campaign_sha256, "records": records},
        sort_keys=True,
        separators=(",", ":"),
    )


@app.function(
    image=image,
    cpu=float(BACKEND_POLICY["container_cpu_cores"]),
    memory=(
        int(BACKEND_POLICY["container_memory_mib"]),
        int(BACKEND_POLICY["container_memory_limit_mib"]),
    ),
    timeout=int(BACKEND_POLICY["seed_timeout_seconds"]),
    retries=int(BACKEND_POLICY["automatic_retries"]),
    max_containers=int(BACKEND_POLICY["maximum_parallel_seeds"]),
    volumes={str(VOLUME_MOUNT): volume},
)
def run_campaign_seed(seed: int) -> str:
    """Resume one seed sequentially while committing its checkpoint after every epoch."""

    import steel_frame_optimization.optimization.campaign as campaign_module
    from steel_frame_optimization.optimization.campaign import (
        CampaignCheckpoint,
        CampaignRunner,
        SeedRunResult,
    )
    from steel_frame_optimization.serialization import write_canonical_json

    if seed < 0 or seed > 29:
        raise ValueError("main campaign seed must be in [0, 29]")
    checkpoint_directory = VOLUME_MOUNT / "checkpoints"
    result_directory = VOLUME_MOUNT / "results"
    log_directory = VOLUME_MOUNT / "logs"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    result_directory.mkdir(parents=True, exist_ok=True)
    log_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_directory / f"seed_{seed:02d}.json"
    result_path = result_directory / f"seed_{seed:02d}.json"
    log_path = log_directory / f"seed_{seed:02d}.log"

    runner = CampaignRunner(
        project_root=REMOTE_PROJECT_ROOT,
        policy_path=(
            REMOTE_PROJECT_ROOT / "configs" / "optimization" / "phase9_three_story_main.json"
        ),
    )
    if result_path.exists():
        checkpoint = CampaignCheckpoint.model_validate_json(
            checkpoint_path.read_text(encoding="utf-8")
        )
        existing = SeedRunResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        if checkpoint.campaign_sha256 != runner.campaign_sha256:
            raise ValueError(f"completed seed {seed} has a different campaign identity")
        if checkpoint.evaluator_identity != runner.evaluator.identity:
            raise ValueError(f"completed seed {seed} has a different evaluator identity")
        if (
            checkpoint.seed != seed
            or checkpoint.completed_epochs != runner.policy.evaluated_population_epochs
            or checkpoint.evaluation_requests != runner.policy.evaluation_request_budget_per_seed
        ):
            raise ValueError(f"completed seed {seed} checkpoint is not complete")
        if (
            existing.seed != seed
            or existing.campaign_id != runner.policy.campaign_id
            or existing.completed_epochs != checkpoint.completed_epochs
            or existing.evaluation_requests != checkpoint.evaluation_requests
            or existing.structural_executions != checkpoint.structural_executions
            or existing.cache_hits != checkpoint.cache_hits
            or not existing.independently_reevaluated_champion_equal
        ):
            raise ValueError(f"completed seed {seed} result does not match its checkpoint")
        return json.dumps(
            {
                "seed": seed,
                "reused_completed_result": True,
                "evaluation_requests": existing.evaluation_requests,
                "structural_executions": existing.structural_executions,
                "cache_hits": existing.cache_hits,
                "champion_status": existing.champion.status,
                "champion_cache_key": existing.champion.cache_key,
                "champion_reevaluation_equal": True,
                "result_sha256": _sha256_file(result_path),
                "checkpoint_sha256": _sha256_file(checkpoint_path),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    original_write = campaign_module._atomic_write_checkpoint

    def durable_write(path: Path, checkpoint: Any) -> None:
        """Persist each atomic epoch checkpoint and append a durable progress record."""

        original_write(path, checkpoint)
        with log_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(
                f"epoch={checkpoint.completed_epochs} requests={checkpoint.evaluation_requests} "
                f"executions={checkpoint.structural_executions} hits={checkpoint.cache_hits}\n"
            )
        volume.commit()

    campaign_module._atomic_write_checkpoint = durable_write
    started = time.perf_counter()
    try:
        with log_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(f"start seed={seed} resume={checkpoint_path.exists()}\n")
        result = runner.run_seed(
            seed=seed,
            checkpoint_path=checkpoint_path,
            resume=checkpoint_path.exists(),
        )
        write_canonical_json(result_path, result)
        with log_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(
                f"complete seed={seed} elapsed_seconds={time.perf_counter() - started:.9f} "
                f"requests={result.evaluation_requests} executions={result.structural_executions} "
                f"hits={result.cache_hits} champion={result.champion.cache_key}\n"
            )
        volume.commit()
    except BaseException as error:
        with log_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(f"failed seed={seed} type={type(error).__name__} message={error}\n")
        volume.commit()
        raise
    finally:
        campaign_module._atomic_write_checkpoint = original_write
    return json.dumps(
        {
            "seed": seed,
            "elapsed_seconds": time.perf_counter() - started,
            "evaluation_requests": result.evaluation_requests,
            "structural_executions": result.structural_executions,
            "cache_hits": result.cache_hits,
            "champion_status": result.champion.status,
            "champion_cache_key": result.champion.cache_key,
            "champion_reevaluation_equal": result.independently_reevaluated_champion_equal,
            "result_sha256": _sha256_file(result_path),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
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
    volumes={str(VOLUME_MOUNT): volume},
)
def finalize_campaign_evidence() -> str:
    """Validate all production artifacts in place and write compact closeout evidence."""

    sys.path.insert(0, str(REMOTE_PROJECT_ROOT))
    from deployment.finalize_phase9 import build_campaign_summary
    from steel_frame_optimization.serialization import canonical_json_dumps, write_canonical_json

    evidence_directory = VOLUME_MOUNT / "evidence" / "main_campaign"
    evidence_directory.mkdir(parents=True, exist_ok=True)
    output_path = evidence_directory / "main_campaign_summary.json"
    summary = build_campaign_summary(
        project_root=REMOTE_PROJECT_ROOT,
        policy_path=(
            REMOTE_PROJECT_ROOT / "configs" / "optimization" / "phase9_three_story_main.json"
        ),
        checkpoint_directory=VOLUME_MOUNT / "checkpoints",
        result_directory=VOLUME_MOUNT / "results",
        log_directory=VOLUME_MOUNT / "logs",
        evidence_directory=evidence_directory,
    )
    write_canonical_json(output_path, summary)
    volume.commit()
    return canonical_json_dumps(
        {
            "summary_sha256": _sha256_file(output_path),
            "seed_count": summary["seed_count"],
            "evaluation_requests_total": summary["evaluation_requests_total"],
            "structural_executions_total": summary["structural_executions_total"],
            "cache_hits_total": summary["cache_hits_total"],
            "aggregate": summary["aggregate"],
            "best_feasible_design": summary["best_feasible_design"],
        },
    ).rstrip("\n")


@app.local_entrypoint()
def probe() -> None:
    """Print the Linux compatibility probe as one JSON record."""

    print(probe_linux.remote())


@app.local_entrypoint()
def pilot() -> None:
    """Print the bounded spawned-process benchmark as one JSON record."""

    print(benchmark_pilot.remote())


@app.local_entrypoint()
def audit() -> None:
    """Print the persistent-volume checkpoint audit as one JSON record."""

    print(audit_volume_checkpoints.remote())


@app.local_entrypoint()
def recovery() -> None:
    """Print exact complete/partial checkpoint restoration evidence without evaluation."""

    print(probe_checkpoint_recovery.remote())


@app.local_entrypoint()
def seed(seed_number: int) -> None:
    """Run or reuse one explicitly selected production seed."""

    print(run_campaign_seed.remote(seed_number))


@app.local_entrypoint()
def campaign(start_seed: int = 0, end_seed: int = 30) -> None:
    """Submit a durable inclusive-exclusive seed range with bounded parallelism."""

    if start_seed < 0 or end_seed > 30 or start_seed >= end_seed:
        raise ValueError("seed range must satisfy 0 <= start_seed < end_seed <= 30")
    maximum_cost = _worst_case_compute_cost(end_seed - start_seed)
    if maximum_cost > float(BACKEND_POLICY["maximum_projected_compute_cost_usd"]):
        raise RuntimeError(
            f"worst-case compute projection ${maximum_cost:.6f} exceeds configured limit"
        )
    print(f"worst_case_compute_projection_usd={maximum_cost:.6f}")
    run_campaign_seed.spawn_map(range(start_seed, end_seed))
    print(f"submitted_seed_range=[{start_seed},{end_seed})")


@app.local_entrypoint()
def finalize() -> None:
    """Validate the complete volume and print its compact aggregate evidence."""

    print(finalize_campaign_evidence.remote())
