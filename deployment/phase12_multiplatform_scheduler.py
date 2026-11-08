"""Supervise deterministic Phase 12 seeds across Kaggle and Google Colab.

The scheduler owns only dispatch and durable bookkeeping.  Scientific state stays
inside validated campaign checkpoints, and a seed is accepted only after its
persisted result passes strict identity and accounting checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, cast

from steel_frame_optimization.optimization.campaign import CampaignCheckpoint, SeedRunResult

Campaign = Literal["case1_filtered", "case1_no_filter", "case2"]
Platform = Literal["kaggle", "colab", "github", "local"]
StateName = Literal["pending", "submitting", "running", "completed", "failed", "blocked"]

CAMPAIGNS: Final[tuple[Campaign, ...]] = (
    "case1_filtered",
    "case1_no_filter",
    "case2",
)
EXPECTED_EPOCHS: Final = 200
EXPECTED_REQUESTS: Final = 20_000
KAGGLE_STATUS_PATTERN: Final = re.compile(r"KernelWorkerStatus\.([A-Z_]+)")


@dataclass(frozen=True, order=True)
class SeedKey:
    """Identify one immutable campaign/seed experiment."""

    campaign: Campaign
    seed: int

    @property
    def token(self) -> str:
        """Return the stable state-file key."""

        return f"{self.campaign}:{self.seed:02d}"


@dataclass(frozen=True)
class CommandResult:
    """Capture one bounded external command without raising on failure."""

    return_code: int
    output: str


def utc_now() -> str:
    """Return a stable UTC timestamp for scheduler evidence."""

    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    """Return the streaming SHA-256 digest of one artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    """Atomically persist scheduler state using deterministic JSON formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    temporary.replace(path)


def run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout: int = 180,
    environment: dict[str, str] | None = None,
    stdin_text: str | None = None,
) -> CommandResult:
    """Run one bounded command and merge stderr for durable diagnostics."""

    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
            input=stdin_text,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return CommandResult(return_code=-1, output=f"{type(error).__name__}: {error}")
    return CommandResult(
        return_code=completed.returncode,
        output=(completed.stdout + completed.stderr).strip(),
    )


def validate_seed_result(
    *, result_path: Path, status_path: Path, campaign_record: dict[str, Any], key: SeedKey
) -> dict[str, Any]:
    """Validate a terminal seed artifact against its frozen identity and budget."""

    result_bytes = result_path.read_bytes()
    run = SeedRunResult.model_validate_json(result_bytes)
    status = cast(dict[str, Any], json.loads(status_path.read_text(encoding="utf-8")))
    checks = {
        "success": status.get("success") is True,
        "campaign": status.get("campaign") == key.campaign,
        "seed": run.seed == status.get("seed") == key.seed,
        "epochs": run.completed_epochs == status.get("completed_epochs") == EXPECTED_EPOCHS,
        "requests": (
            run.evaluation_requests == status.get("evaluation_requests") == EXPECTED_REQUESTS
        ),
        "accounting": (
            run.structural_executions + run.cache_hits == EXPECTED_REQUESTS
            and run.structural_executions == status.get("structural_executions")
            and run.cache_hits == status.get("cache_hits")
        ),
        "history": (
            len(run.history) == EXPECTED_EPOCHS
            and run.history[-1].epoch_index == EXPECTED_EPOCHS - 1
            and run.history[-1].evaluation_requests_total == EXPECTED_REQUESTS
        ),
        "champion": (
            run.independently_reevaluated_champion_equal
            and status.get("independently_reevaluated_champion_equal") is True
        ),
        "result_sha256": sha256_file(result_path) == status.get("result_sha256"),
        "campaign_sha256": status.get("campaign_sha256") == campaign_record["campaign_sha256"],
        "evaluator_identity": status.get("evaluator_identity")
        == campaign_record["evaluator_identity"],
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"invalid terminal artifact for {key.token}: {', '.join(failed)}")
    return {
        "validated_at": utc_now(),
        "result_sha256": status["result_sha256"],
        "checkpoint_sha256": status["checkpoint_sha256"],
        "evaluation_requests": EXPECTED_REQUESTS,
        "structural_executions": run.structural_executions,
        "cache_hits": run.cache_hits,
        "elapsed_seconds": run.elapsed_seconds,
    }


def choose_next_seed(
    records: dict[str, dict[str, Any]], source_requests: dict[str, int]
) -> SeedKey | None:
    """Choose the most advanced eligible seed while preserving campaign progress."""

    running_campaigns = {
        cast(Campaign, record["campaign"])
        for record in records.values()
        if record["state"] in {"running", "submitting"}
    }
    candidates = [
        SeedKey(cast(Campaign, record["campaign"]), int(record["seed"]))
        for record in records.values()
        if record["state"] == "pending"
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda key: (
            key.campaign not in running_campaigns,
            source_requests.get(key.token, 0),
            -CAMPAIGNS.index(key.campaign),
            -key.seed,
        ),
        reverse=True,
    )
    return candidates[0]


def choose_github_seed(
    records: dict[str, dict[str, Any]], source_requests: dict[str, int]
) -> SeedKey | None:
    """Choose one unclaimed seed that can safely start without a private checkpoint."""

    candidates = [
        SeedKey(cast(Campaign, record["campaign"]), int(record["seed"]))
        for record in records.values()
        if record["state"] == "pending"
        and source_requests.get(f"{record['campaign']}:{int(record['seed']):02d}", 0) == 0
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda key: (CAMPAIGNS.index(key.campaign), key.seed))
    return candidates[0]


def validate_github_status_binding(
    *, status: dict[str, Any], key: SeedKey, record: dict[str, Any]
) -> None:
    """Reject an artifact that is not bound to the scheduler's exclusive lease."""

    checks = {
        "success": status.get("success") is True,
        "platform": status.get("platform") == "github_actions",
        "assignment": status.get("assignment_token") == record["github_assignment_token"],
        "campaign": status.get("campaign") == key.campaign,
        "seed": status.get("seed") == key.seed,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"invalid GitHub lease binding for {key.token}: {', '.join(failed)}")


class Phase12Scheduler:
    """Reconcile remote ownership, validate completions, and fill free slots."""

    def __init__(self, *, project_root: Path, state_root: Path, dry_run: bool = False) -> None:
        """Load frozen scheduler inputs and initialize restart-safe state."""

        self.project_root = project_root.resolve()
        self.state_root = state_root.resolve()
        self.dry_run = dry_run
        self.config = json.loads(
            (self.project_root / "configs/optimization/phase12_scheduler.json").read_text(
                encoding="utf-8"
            )
        )
        self.recovery_manifest = json.loads(
            (
                self.project_root
                / "outputs/runs/phase12_kaggle_recovery_upload/phase12_recovery_manifest.json"
            ).read_text(encoding="utf-8")
        )
        self.state_path = self.state_root / "state.json"
        self.events_path = self.state_root / "events.jsonl"
        self.snapshots_path = self.state_root / "snapshots.jsonl"
        self.state = self._load_or_initialize_state()
        self.source_requests = {
            f"{item['campaign']}:{int(item['seed']):02d}": int(item["evaluation_requests"])
            for item in self.recovery_manifest["checkpoints"]
        }

    def _load_or_initialize_state(self) -> dict[str, Any]:
        """Restore durable state or create the complete 90-seed queue once."""

        if self.state_path.is_file():
            return cast(dict[str, Any], json.loads(self.state_path.read_text(encoding="utf-8")))
        completed = {
            f"{campaign}:{int(seed):02d}"
            for campaign, seeds in self.config["completed_baseline"].items()
            for seed in seeds
        }
        assignments = {
            f"{item['campaign']}:{int(item['seed']):02d}": item
            for item in self.config["initial_assignments"]
        }
        records: dict[str, dict[str, Any]] = {}
        for campaign in self.config["campaigns"]:
            for seed in self.config["seeds"]:
                token = f"{campaign}:{int(seed):02d}"
                assignment = assignments.get(token)
                records[token] = {
                    "campaign": campaign,
                    "seed": int(seed),
                    "state": "completed"
                    if token in completed
                    else ("running" if assignment else "pending"),
                    "platform": None if assignment is None else assignment["platform"],
                    "remote_id": None if assignment is None else assignment["remote_id"],
                    "updated_at": utc_now(),
                    "validation": None,
                    "last_error": None,
                }
        state = {
            "schema_version": "1.0",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "records": records,
            "last_snapshot_at": None,
        }
        atomic_json(self.state_path, state)
        return state

    @property
    def records(self) -> dict[str, dict[str, Any]]:
        """Return mutable seed records in the persisted scheduler state."""

        return cast(dict[str, dict[str, Any]], self.state["records"])

    def _save(self) -> None:
        """Persist the current scheduler state atomically."""

        self.state["updated_at"] = utc_now()
        atomic_json(self.state_path, self.state)

    def _event(self, event: str, **details: object) -> None:
        """Append one compact scheduler event without rewriting history."""

        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"captured_at": utc_now(), "event": event, **details}
        with self.events_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def _campaign_record(self, campaign: Campaign) -> dict[str, Any]:
        """Return one frozen campaign identity from the recovery manifest."""

        return cast(dict[str, Any], self.recovery_manifest["campaigns"][campaign])

    def _kaggle_slug(self, key: SeedKey) -> str:
        """Return the unique remote Kaggle identifier for one seed."""

        campaign = key.campaign.replace("_", "-")
        return f"{self.config['kaggle_owner']}/steel-frame-phase-12-{campaign}-seed-{key.seed:02d}"

    def _kaggle_status(self, remote_id: str) -> tuple[str | None, str]:
        """Query one Kaggle job, returning None for transient CLI failures."""

        result = run_command(["kaggle", "kernels", "status", remote_id], cwd=self.project_root)
        match = KAGGLE_STATUS_PATTERN.search(result.output)
        return (None if match is None else match.group(1), result.output)

    def _download_kaggle_result(self, key: SeedKey, remote_id: str) -> tuple[Path, Path]:
        """Download compact terminal evidence for one Kaggle seed."""

        destination = self.state_root / "completed" / key.campaign / f"seed_{key.seed:02d}"
        destination.mkdir(parents=True, exist_ok=True)
        pattern = (
            rf"(job_status\.json|wrapper_status\.json|results/seed_{key.seed:02d}\.json|"
            rf"logs/seed_{key.seed:02d}\.log)$"
        )
        result = run_command(
            [
                "kaggle",
                "kernels",
                "output",
                remote_id,
                "-p",
                str(destination),
                "--file-pattern",
                pattern,
            ],
            cwd=self.project_root,
            timeout=600,
        )
        if result.return_code != 0:
            raise RuntimeError(f"Kaggle artifact download failed: {result.output}")
        result_paths = list(destination.rglob(f"seed_{key.seed:02d}.json"))
        status_paths = list(destination.rglob("job_status.json"))
        result_paths = [path for path in result_paths if path.parent.name == "results"]
        if len(result_paths) != 1 or len(status_paths) != 1:
            raise RuntimeError("Kaggle terminal output is incomplete or ambiguous")
        return result_paths[0], status_paths[0]

    def _accept_completion(
        self, key: SeedKey, *, platform: Platform, result_path: Path, status_path: Path
    ) -> None:
        """Validate and atomically accept one completed remote seed."""

        validation = validate_seed_result(
            result_path=result_path,
            status_path=status_path,
            campaign_record=self._campaign_record(key.campaign),
            key=key,
        )
        record = self.records[key.token]
        record.update(
            {
                "state": "completed",
                "platform": platform,
                "updated_at": utc_now(),
                "validation": validation,
                "last_error": None,
            }
        )
        self._event("seed_completed", seed=key.token, platform=platform, **validation)
        self._save()

    def _reconcile_staged_results(self) -> None:
        """Accept manually staged terminal artifacts before assigning new work."""

        for record in self.records.values():
            if record["state"] != "pending":
                continue
            key = SeedKey(cast(Campaign, record["campaign"]), int(record["seed"]))
            root = self.state_root / "completed" / key.campaign / f"seed_{key.seed:02d}"
            result_paths = [
                path
                for path in root.rglob(f"seed_{key.seed:02d}.json")
                if path.parent.name == "results"
            ]
            status_paths = list(root.rglob("job_status.json")) + list(
                root.rglob(f"seed_{key.seed:02d}_status.json")
            )
            if len(result_paths) != 1 or len(status_paths) != 1:
                continue
            try:
                self._accept_completion(
                    key,
                    platform="local",
                    result_path=result_paths[0],
                    status_path=status_paths[0],
                )
            except (OSError, ValueError, RuntimeError) as error:
                record["state"] = "blocked"
                record["last_error"] = f"staged result validation failed: {error}"
                self._event("completion_validation_failed", seed=key.token, error=str(error))
                self._save()

    def _reconcile_kaggle(self) -> None:
        """Reconcile all owned Kaggle jobs without inferring remote failure."""

        for token, record in list(self.records.items()):
            if record["platform"] != "kaggle" or record["state"] not in {
                "running",
                "submitting",
            }:
                continue
            key = SeedKey(cast(Campaign, record["campaign"]), int(record["seed"]))
            remote_id = str(record["remote_id"])
            status, raw = self._kaggle_status(remote_id)
            record["last_remote_status"] = status
            record["last_checked_at"] = utc_now()
            if status is None:
                record["last_error"] = f"transient Kaggle status failure: {raw[-500:]}"
                self._event("transient_platform_error", seed=token, platform="kaggle")
                continue
            if status == "RUNNING" or status == "QUEUED":
                record["state"] = "running"
                record["last_error"] = None
            elif status == "COMPLETE":
                try:
                    result_path, status_path = self._download_kaggle_result(key, remote_id)
                    self._accept_completion(
                        key,
                        platform="kaggle",
                        result_path=result_path,
                        status_path=status_path,
                    )
                except (OSError, ValueError, RuntimeError) as error:
                    record["state"] = "blocked"
                    record["last_error"] = str(error)
                    self._event("completion_validation_failed", seed=token, error=str(error))
            elif status in {"ERROR", "CANCELLED"}:
                record["state"] = "failed"
                record["last_error"] = f"terminal Kaggle state: {status}"
                self._event("remote_seed_failed", seed=token, platform="kaggle", status=status)
        self._save()

    def _colab_command(self, shell_command: str, *, timeout: int = 180) -> CommandResult:
        """Run one bounded WSL Colab CLI command with local proxy variables."""

        prefix = (
            "export HTTP_PROXY=socks5h://127.0.0.1:10808 "
            "HTTPS_PROXY=socks5h://127.0.0.1:10808 "
            "ALL_PROXY=socks5h://127.0.0.1:10808; "
        )
        return run_command(
            ["wsl.exe", "-e", "bash", "-lc", prefix + shell_command],
            cwd=self.project_root,
            timeout=timeout,
        )

    def _download_colab_file(self, remote: str, local: Path) -> None:
        """Download one Drive-backed Colab file through the active session."""

        local.parent.mkdir(parents=True, exist_ok=True)
        wsl_local = "/mnt/" + local.drive[0].lower() + local.as_posix()[2:]
        session = self.config["colab_session"]
        command = f"$HOME/.local/bin/colab download -s {session} '{remote}' '{wsl_local}'"
        result = self._colab_command(command, timeout=600)
        if result.return_code != 0 or not local.is_file():
            raise RuntimeError(f"Colab download failed: {result.output}")

    def _reconcile_colab(self) -> None:
        """Reconcile the single verified Colab CPU lane and its Drive result."""

        active = [
            record
            for record in self.records.values()
            if record["platform"] == "colab" and record["state"] in {"running", "submitting"}
        ]
        if not active:
            return
        if len(active) > 1:
            raise RuntimeError("more than one seed owns the verified Colab session")
        record = active[0]
        key = SeedKey(cast(Campaign, record["campaign"]), int(record["seed"]))
        session = self.config["colab_session"]
        status_result = self._colab_command(f"$HOME/.local/bin/colab status -s {session}")
        record["last_checked_at"] = utc_now()
        if status_result.return_code != 0:
            record["last_error"] = f"transient Colab status failure: {status_result.output[-500:]}"
            self._event("transient_platform_error", seed=key.token, platform="colab")
            self._save()
            return
        if "BUSY" in status_result.output:
            record["state"] = "running"
            record["last_error"] = None
            self._save()
            return
        if "not found" in status_result.output.lower():
            sessions = self._colab_command("$HOME/.local/bin/colab sessions")
            if sessions.return_code == 0 and "No active sessions found" in sessions.output:
                self._recover_expired_colab(key, record)
                return
        if "IDLE" not in status_result.output:
            record["last_error"] = f"unrecognized Colab state: {status_result.output[-500:]}"
            self._save()
            return
        base = f"/content/drive/MyDrive/steel-frame-phase12-v1/campaigns/{key.campaign}"
        local = self.state_root / "completed" / key.campaign / f"seed_{key.seed:02d}"
        result_path = local / "results" / f"seed_{key.seed:02d}.json"
        status_path = local / "evidence" / f"seed_{key.seed:02d}_status.json"
        try:
            self._download_colab_file(f"{base}/results/seed_{key.seed:02d}.json", result_path)
            self._download_colab_file(
                f"{base}/evidence/seed_{key.seed:02d}_status.json", status_path
            )
            self._accept_completion(
                key, platform="colab", result_path=result_path, status_path=status_path
            )
        except (OSError, ValueError, RuntimeError) as error:
            record["state"] = "blocked"
            record["last_error"] = (
                "Colab became idle without a valid persisted result; manual recovery required: "
                f"{error}"
            )
            self._event("completion_validation_failed", seed=key.token, error=str(error))
            self._save()

    def _recover_expired_colab(self, key: SeedKey, record: dict[str, Any]) -> None:
        """Recreate an expired Colab session and resume its exclusively owned Drive state."""

        # Reconciliation reaches this method only after both the named session
        # and the server inventory prove that the one owned VM is absent. A
        # previous successful replacement must not disable a later recovery.
        record["recovery_attempted_at"] = utc_now()
        self._save()
        session = self.config["colab_session"]
        self._colab_command(
            "pkill -f '^/home/.*/python .*/colab exec --session " + session + "' || true"
        )
        commands = (f"$HOME/.local/bin/colab new -s {session}",)
        for command in commands:
            result = self._colab_command(command, timeout=1900)
            if result.return_code != 0:
                record["state"] = "blocked"
                record["last_error"] = f"Colab session recovery failed: {result.output[-1000:]}"
                self._event("colab_recovery_failed", seed=key.token)
                self._save()
                return
        record["state"] = "blocked"
        record["last_error"] = (
            "replacement Colab VM is ready; interactive per-VM Drive mount and RETRY_COLAB "
            "are required"
        )
        self._event("colab_drive_authorization_required", seed=key.token, session=session)
        self._save()

    def _bootstrap_colab_owned_seed(
        self, key: SeedKey, record: dict[str, Any], *, wsl_restore_script: str
    ) -> None:
        """Restore immutable inputs and resume one already-owned Drive checkpoint."""

        session = self.config["colab_session"]
        restored = self._colab_command(
            f"$HOME/.local/bin/colab exec -s {session} --file '{wsl_restore_script}' "
            "--timeout 1800",
            timeout=1900,
        )
        if restored.return_code != 0:
            record["state"] = "blocked"
            record["last_error"] = f"Colab runtime restoration failed: {restored.output[-1000:]}"
            self._save()
            return
        manifest = (
            self.project_root
            / "outputs/runs/phase12_kaggle_recovery_upload/phase12_recovery_manifest.json"
        )
        worker_script = self.project_root / "deployment/colab_phase12_seed_job.py"
        recovery_files = (
            (manifest, "/content/phase12_recovery_manifest.json", "manifest"),
            (worker_script, "/content/colab_phase12_seed_job.py", "worker script"),
        )
        for source, destination, label in recovery_files:
            wsl_source = "/mnt/" + source.drive[0].lower() + source.as_posix()[2:]
            upload = self._colab_command(
                f"$HOME/.local/bin/colab upload -s {session} '{wsl_source}' '{destination}'",
                timeout=600,
            )
            if upload.return_code != 0:
                record["state"] = "blocked"
                record["last_error"] = (
                    f"Colab {label} recovery upload failed: {upload.output[-1000:]}"
                )
                self._save()
                return
        self._spawn_colab_launcher(
            key,
            remote_source="/content/recovery-owned-by-drive.json",
            source_sha="RECOVERY",
            start_fresh=False,
            recovery_run=True,
        )
        record["state"] = "running"
        record["last_error"] = None
        self._event("colab_session_recovered", seed=key.token, session=session)
        self._save()

    def _resume_colab_after_manual_drive_auth(self) -> None:
        """Continue a blocked recovery after an operator completes per-VM Drive consent."""

        marker = self.state_root / "RETRY_COLAB"
        if not marker.is_file():
            return
        blocked = [
            record
            for record in self.records.values()
            if record["platform"] == "colab" and record["state"] == "blocked"
        ]
        if len(blocked) != 1:
            return
        record = blocked[0]
        session = self.config["colab_session"]
        status = self._colab_command(f"$HOME/.local/bin/colab status -s {session}")
        if status.return_code != 0 and "not found" in status.output.lower():
            replacement = self._colab_command(
                f"$HOME/.local/bin/colab new -s {session}", timeout=1900
            )
            marker.unlink()
            if replacement.return_code != 0:
                record["last_error"] = (
                    "RETRY_COLAB found an expired session and replacement failed: "
                    f"{replacement.output[-1000:]}"
                )
                self._event(
                    "colab_recovery_failed", seed=record["campaign"] + f":{record['seed']:02d}"
                )
                self._save()
                return
            record["recovery_attempted_at"] = utc_now()
            record["last_error"] = (
                "replacement Colab VM was recreated; interactive per-VM Drive mount and a new "
                "RETRY_COLAB marker are required"
            )
            self._event(
                "colab_drive_authorization_required",
                seed=record["campaign"] + f":{record['seed']:02d}",
                session=session,
            )
            self._save()
            return
        probe = self._colab_command(
            f"$HOME/.local/bin/colab ls -s {session} "
            "/content/drive/MyDrive/steel-frame-phase12-v1/immutable-inputs"
        )
        if probe.return_code != 0 or "phase12-python-3.12.6-runtime.tar.gz" not in probe.output:
            record["last_error"] = "RETRY_COLAB present, but the Drive mount is not verified"
            self._save()
            return
        restore_script = self.project_root / "deployment/colab_phase12_restore_runtime.py"
        wsl_restore_script = (
            "/mnt/" + restore_script.drive[0].lower() + restore_script.as_posix()[2:]
        )
        self._bootstrap_colab_owned_seed(
            key=SeedKey(cast(Campaign, record["campaign"]), int(record["seed"])),
            record=record,
            wsl_restore_script=wsl_restore_script,
        )
        if record["state"] == "running":
            # A later VM expiration must receive one fresh automatic recovery attempt.
            record.pop("recovery_attempted_at", None)
            marker.unlink()

    def _launch_kaggle(self, key: SeedKey) -> None:
        """Create and submit one private Kaggle seed assignment exactly once."""

        record = self.records[key.token]
        remote_id = self._kaggle_slug(key)
        record.update(
            {
                "state": "submitting",
                "platform": "kaggle",
                "remote_id": remote_id,
                "updated_at": utc_now(),
                "last_error": None,
            }
        )
        self._save()
        if self.dry_run:
            self._event("dry_run_dispatch", seed=key.token, platform="kaggle")
            return
        destination = self.state_root / "kernels" / key.campaign / f"seed_{key.seed:02d}"
        prepare = run_command(
            [
                sys.executable,
                str(self.project_root / "deployment/prepare_kaggle_phase12_kernel.py"),
                "--campaign",
                key.campaign,
                "--seed",
                str(key.seed),
                "--destination",
                str(destination),
            ],
            cwd=self.project_root,
        )
        if prepare.return_code != 0:
            record["state"] = "blocked"
            record["last_error"] = f"kernel preparation failed: {prepare.output}"
            self._save()
            return
        pushed = run_command(
            ["kaggle", "kernels", "push", "-p", str(destination)],
            cwd=self.project_root,
            timeout=600,
        )
        if pushed.return_code == 0:
            record["state"] = "running"
            record["launched_at"] = utc_now()
            self._event("seed_dispatched", seed=key.token, platform="kaggle", remote_id=remote_id)
        else:
            record["state"] = "blocked"
            record["last_error"] = (
                "Kaggle submission outcome was not verified; exact slug must be reconciled before "
                f"retry: {pushed.output[-1000:]}"
            )
            self._event("dispatch_unverified", seed=key.token, platform="kaggle")
        self._save()

    def _launch_colab(self, key: SeedKey) -> None:
        """Upload an optional recovery input and start one detached Colab CLI wrapper."""

        record = self.records[key.token]
        source = (
            self.project_root
            / "outputs/runs/phase12_platform_migration"
            / key.campaign
            / "checkpoints"
            / f"seed_{key.seed:02d}.json"
        )
        start_fresh = not source.is_file()
        source_sha = "NONE" if start_fresh else sha256_file(source)
        remote_source = f"/content/{key.campaign}_seed_{key.seed:02d}_source.json"
        if not start_fresh:
            wsl_source = "/mnt/" + source.drive[0].lower() + source.as_posix()[2:]
            upload = self._colab_command(
                f"$HOME/.local/bin/colab upload -s {self.config['colab_session']} "
                f"'{wsl_source}' '{remote_source}'",
                timeout=1200,
            )
            if upload.return_code != 0:
                record["state"] = "blocked"
                record["last_error"] = f"Colab recovery upload failed: {upload.output[-1000:]}"
                self._save()
                return
        record.update(
            {
                "state": "submitting",
                "platform": "colab",
                "remote_id": self.config["colab_session"],
                "updated_at": utc_now(),
                "last_error": None,
            }
        )
        self._save()
        if self.dry_run:
            self._event("dry_run_dispatch", seed=key.token, platform="colab")
            return
        self._spawn_colab_launcher(
            key,
            remote_source=remote_source,
            source_sha=source_sha,
            start_fresh=start_fresh,
            recovery_run=False,
        )
        record["state"] = "running"
        record["launched_at"] = utc_now()
        self._save()

    def _spawn_colab_launcher(
        self,
        key: SeedKey,
        *,
        remote_source: str,
        source_sha: str,
        start_fresh: bool,
        recovery_run: bool,
    ) -> None:
        """Start the local Colab CLI bridge for an already-owned seed."""

        logs = self.state_root / "colab_launchers"
        logs.mkdir(parents=True, exist_ok=True)
        stdout = (logs / f"{key.campaign}_seed_{key.seed:02d}.stdout.log").open(
            "a", encoding="utf-8"
        )
        stderr = (logs / f"{key.campaign}_seed_{key.seed:02d}.stderr.log").open(
            "a", encoding="utf-8"
        )
        script = self.project_root / "deployment/run_colab_phase12_seed.sh"
        wsl_script = "/mnt/" + script.drive[0].lower() + script.as_posix()[2:]
        shell = (
            f"'{wsl_script}' {key.campaign} {key.seed} '{remote_source}' "
            f"'{source_sha}' '{self.config['colab_session']}' {1 if start_fresh else 0} "
            f"{1 if recovery_run else 0}"
        )
        process = subprocess.Popen(
            ["wsl.exe", "-e", "bash", "-lc", shell],
            cwd=self.project_root,
            stdout=stdout,
            stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        stdout.close()
        stderr.close()
        record = self.records[key.token]
        record["launcher_pid"] = process.pid
        self._event(
            "seed_dispatched",
            seed=key.token,
            platform="colab",
            launcher_pid=process.pid,
            start_fresh=start_fresh,
            recovery_run=recovery_run,
        )

    def _github_command(self, arguments: list[str], *, timeout: int = 180) -> CommandResult:
        """Run the authenticated GitHub CLI without exposing its stored credential."""

        executable = self.project_root / str(self.config["github_cli"])
        if not executable.is_file():
            return CommandResult(return_code=-1, output=f"GitHub CLI not found: {executable}")
        credential = run_command(
            ["git", "credential-manager-core", "get"],
            cwd=self.project_root,
            environment={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            stdin_text="protocol=https\nhost=github.com\n\n",
        )
        token = next(
            (
                line.removeprefix("password=")
                for line in credential.output.splitlines()
                if line.startswith("password=")
            ),
            None,
        )
        if credential.return_code != 0 or token is None:
            return CommandResult(return_code=-1, output="GitHub credential is unavailable")
        return run_command(
            [str(executable), *arguments],
            cwd=self.project_root,
            timeout=timeout,
            environment={**os.environ, "GH_TOKEN": token},
        )

    def _find_github_run(self, assignment_token: str) -> str | None:
        """Return the workflow run ID bearing one unique scheduler assignment token."""

        result = self._github_command(
            [
                "run",
                "list",
                "--repo",
                str(self.config["github_repository"]),
                "--workflow",
                str(self.config["github_workflow"]),
                "--limit",
                "30",
                "--json",
                "databaseId,displayTitle",
            ]
        )
        if result.return_code != 0:
            return None
        runs = json.loads(result.output)
        matches = [
            str(item["databaseId"])
            for item in runs
            if str(item.get("displayTitle", "")).endswith(assignment_token)
        ]
        if len(matches) > 1:
            raise RuntimeError(f"duplicate GitHub workflow runs for {assignment_token}")
        return matches[0] if matches else None

    def _dispatch_github_chunk(
        self, key: SeedKey, *, previous_run_id: str | None, target_epochs: int
    ) -> None:
        """Dispatch exactly one bounded workflow chunk for an already leased seed."""

        record = self.records[key.token]
        lease_token = str(record["lease_token"])
        assignment_token = f"{lease_token}-e{target_epochs:03d}"
        record.update(
            {
                "state": "submitting",
                "platform": "github",
                "remote_id": None,
                "github_previous_run_id": previous_run_id,
                "github_target_epochs": target_epochs,
                "github_assignment_token": assignment_token,
                "updated_at": utc_now(),
                "last_error": None,
            }
        )
        self._save()
        if self.dry_run:
            self._event(
                "dry_run_dispatch",
                seed=key.token,
                platform="github",
                target_epochs=target_epochs,
            )
            return
        arguments = [
            "workflow",
            "run",
            str(self.config["github_workflow"]),
            "--repo",
            str(self.config["github_repository"]),
            "--ref",
            str(self.config["github_branch"]),
            "-f",
            f"campaign={key.campaign}",
            "-f",
            f"seed={key.seed:02d}",
            "-f",
            f"target_epochs={target_epochs}",
            "-f",
            f"assignment_token={assignment_token}",
        ]
        if previous_run_id is not None:
            arguments.extend(["-f", f"previous_run_id={previous_run_id}"])
        dispatched = self._github_command(arguments)
        if dispatched.return_code != 0:
            record["last_error"] = (
                "GitHub dispatch outcome was not verified; the assignment remains leased: "
                f"{dispatched.output[-1000:]}"
            )
            self._event("dispatch_unverified", seed=key.token, platform="github")
            self._save()
            return
        self._event(
            "github_chunk_submitted",
            seed=key.token,
            assignment_token=assignment_token,
            target_epochs=target_epochs,
        )
        self._save()

    def _launch_github(self, key: SeedKey) -> None:
        """Lease a fresh seed and submit its first free hosted-runner chunk."""

        record = self.records[key.token]
        record["lease_token"] = f"{key.campaign}-{key.seed:02d}-{uuid.uuid4().hex[:12]}"
        record["launched_at"] = utc_now()
        self._dispatch_github_chunk(
            key,
            previous_run_id=None,
            target_epochs=int(self.config["github_chunk_epochs"]),
        )

    def _download_github_artifact(self, key: SeedKey, run_id: str) -> Path:
        """Download one workflow artifact into a run-specific immutable directory."""

        destination = self.state_root / "github" / key.campaign / f"seed_{key.seed:02d}" / run_id
        if destination.exists():
            checkpoint = destination / "checkpoint.json"
            if checkpoint.is_file():
                return destination
        destination.mkdir(parents=True, exist_ok=True)
        result = self._github_command(
            [
                "run",
                "download",
                run_id,
                "--repo",
                str(self.config["github_repository"]),
                "--name",
                "phase12-checkpoint",
                "--dir",
                str(destination),
            ],
            timeout=900,
        )
        if result.return_code != 0:
            raise RuntimeError(f"GitHub artifact download failed: {result.output}")
        return destination

    def _validate_github_chunk(
        self, key: SeedKey, record: dict[str, Any], artifact_root: Path
    ) -> tuple[CampaignCheckpoint, dict[str, Any]]:
        """Validate identity, hash, deterministic state, and exact chunk accounting."""

        checkpoint_path = artifact_root / "checkpoint.json"
        status_path = artifact_root / "job_status.json"
        if not checkpoint_path.is_file() or not status_path.is_file():
            raise RuntimeError("GitHub artifact does not contain checkpoint and status files")
        checkpoint = CampaignCheckpoint.model_validate_json(
            checkpoint_path.read_text(encoding="utf-8")
        )
        status = cast(dict[str, Any], json.loads(status_path.read_text(encoding="utf-8")))
        validate_github_status_binding(status=status, key=key, record=record)
        campaign_record = self._campaign_record(key.campaign)
        checks = {
            "seed": status.get("seed") == checkpoint.seed == key.seed,
            "campaign_identity": status.get("campaign_sha256")
            == checkpoint.campaign_sha256
            == campaign_record["campaign_sha256"],
            "evaluator_identity": status.get("evaluator_identity")
            == checkpoint.evaluator_identity.model_dump(mode="json")
            == campaign_record["evaluator_identity"],
            "target_epoch": status.get("completed_epochs")
            == checkpoint.completed_epochs
            == record["github_target_epochs"],
            "requests": status.get("evaluation_requests")
            == checkpoint.evaluation_requests
            == checkpoint.completed_epochs * 100,
            "accounting": checkpoint.structural_executions + checkpoint.cache_hits
            == checkpoint.evaluation_requests,
            "history": len(checkpoint.history) == checkpoint.completed_epochs
            and checkpoint.history[-1].evaluation_requests_total == checkpoint.evaluation_requests,
            "checkpoint_sha256": status.get("checkpoint_sha256") == sha256_file(checkpoint_path),
            "champion": status.get("independently_reevaluated_champion_equal") is True,
        }
        failed = sorted(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError(f"invalid GitHub chunk for {key.token}: {', '.join(failed)}")
        return checkpoint, status

    def _reconcile_github(self) -> None:
        """Reconcile the single free GitHub lane and chain validated chunks safely."""

        for token, record in list(self.records.items()):
            if record["platform"] != "github" or record["state"] not in {
                "running",
                "submitting",
            }:
                continue
            key = SeedKey(cast(Campaign, record["campaign"]), int(record["seed"]))
            remote_id = record.get("remote_id")
            if remote_id is None:
                try:
                    remote_id = self._find_github_run(str(record["github_assignment_token"]))
                except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
                    record["last_error"] = f"transient GitHub run discovery failure: {error}"
                    continue
                if remote_id is None:
                    record["last_error"] = "GitHub submission accepted; run ID not visible yet"
                    continue
                record["remote_id"] = remote_id
                record["state"] = "running"
                record["last_error"] = None
                self._event("github_run_identified", seed=token, run_id=remote_id)
                self._save()
            viewed = self._github_command(
                [
                    "run",
                    "view",
                    str(remote_id),
                    "--repo",
                    str(self.config["github_repository"]),
                    "--json",
                    "status,conclusion,url",
                ]
            )
            if viewed.return_code != 0:
                record["last_error"] = f"transient GitHub status failure: {viewed.output[-500:]}"
                self._event("transient_platform_error", seed=token, platform="github")
                continue
            remote = json.loads(viewed.output)
            record["last_remote_status"] = remote.get("status")
            record["last_checked_at"] = utc_now()
            if remote.get("status") != "completed":
                record["state"] = "running"
                record["last_error"] = None
                continue
            if remote.get("conclusion") != "success":
                record["state"] = "blocked"
                record["last_error"] = (
                    f"GitHub chunk ended with conclusion {remote.get('conclusion')}; "
                    "the seed lease and prior checkpoint are preserved"
                )
                self._event("remote_seed_blocked", seed=token, platform="github")
                continue
            try:
                artifact_root = self._download_github_artifact(key, str(remote_id))
                checkpoint, _ = self._validate_github_chunk(key, record, artifact_root)
                record["github_last_validated_run_id"] = str(remote_id)
                record["github_last_validated_epoch"] = checkpoint.completed_epochs
                record["github_last_checkpoint_sha256"] = sha256_file(
                    artifact_root / "checkpoint.json"
                )
                self.source_requests[key.token] = checkpoint.evaluation_requests
                if checkpoint.completed_epochs == EXPECTED_EPOCHS:
                    self._accept_completion(
                        key,
                        platform="github",
                        result_path=artifact_root / "result.json",
                        status_path=artifact_root / "job_status.json",
                    )
                else:
                    target = min(
                        EXPECTED_EPOCHS,
                        checkpoint.completed_epochs + int(self.config["github_chunk_epochs"]),
                    )
                    self._dispatch_github_chunk(
                        key, previous_run_id=str(remote_id), target_epochs=target
                    )
            except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
                record["state"] = "blocked"
                record["last_error"] = f"GitHub artifact validation failed: {error}"
                self._event("completion_validation_failed", seed=token, error=str(error))
        self._save()

    def _fill_slots(self) -> None:
        """Keep verified Kaggle, Colab, and free GitHub capacity uniquely occupied."""

        kaggle_active = sum(
            record["platform"] == "kaggle"
            and record["state"] in {"running", "submitting", "blocked"}
            for record in self.records.values()
        )
        while kaggle_active < int(self.config["kaggle_slots"]):
            key = choose_next_seed(self.records, self.source_requests)
            if key is None:
                break
            self._launch_kaggle(key)
            kaggle_active += 1
        colab_active = any(
            record["platform"] == "colab"
            and record["state"] in {"running", "submitting", "blocked"}
            for record in self.records.values()
        )
        if not colab_active and int(self.config["colab_slots"]) > 0:
            key = choose_next_seed(self.records, self.source_requests)
            if key is not None:
                self._launch_colab(key)
        github_active = sum(
            record["platform"] == "github"
            and record["state"] in {"running", "submitting", "blocked"}
            for record in self.records.values()
        )
        if bool(self.config.get("github_enabled", False)):
            while github_active < int(self.config.get("github_slots", 0)):
                key = choose_github_seed(self.records, self.source_requests)
                if key is None:
                    break
                self._launch_github(key)
                github_active += 1

    def _snapshot(self, *, force: bool = False) -> None:
        """Append a conservative 30-minute progress and storage snapshot."""

        previous = self.state.get("last_snapshot_at")
        if previous and not force:
            elapsed = datetime.now(UTC) - datetime.fromisoformat(str(previous))
            if elapsed.total_seconds() < int(self.config["snapshot_seconds"]):
                return
        counts = {
            state: sum(record["state"] == state for record in self.records.values())
            for state in ("completed", "running", "submitting", "pending", "failed", "blocked")
        }
        committed = 0
        by_campaign: dict[str, dict[str, int]] = {}
        for campaign in CAMPAIGNS:
            campaign_records = [
                record for record in self.records.values() if record["campaign"] == campaign
            ]
            completed_count = sum(record["state"] == "completed" for record in campaign_records)
            campaign_committed = sum(
                EXPECTED_REQUESTS
                if record["state"] == "completed"
                else self.source_requests.get(f"{campaign}:{int(record['seed']):02d}", 0)
                for record in campaign_records
            )
            committed += campaign_committed
            by_campaign[campaign] = {
                "completed_seeds": completed_count,
                "committed_requests_lower_bound": campaign_committed,
            }
        storage = sum(path.stat().st_size for path in self.state_root.rglob("*") if path.is_file())
        snapshot = {
            "captured_at": utc_now(),
            "counts": counts,
            "by_campaign": by_campaign,
            "committed_requests_lower_bound": committed,
            "total_requests": len(self.records) * EXPECTED_REQUESTS,
            "scheduler_storage_bytes": storage,
            "active": [
                {
                    "campaign": record["campaign"],
                    "seed": record["seed"],
                    "platform": record["platform"],
                    "remote_id": record["remote_id"],
                    "state": record["state"],
                }
                for record in self.records.values()
                if record["state"] in {"running", "submitting", "blocked"}
            ],
        }
        self.snapshots_path.parent.mkdir(parents=True, exist_ok=True)
        with self.snapshots_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(snapshot, sort_keys=True) + "\n")
        self.state["last_snapshot_at"] = snapshot["captured_at"]
        self._save()

    def cycle(self, *, snapshot: bool = False) -> None:
        """Run one bounded reconcile, dispatch and evidence cycle."""

        self._reconcile_staged_results()
        self._resume_colab_after_manual_drive_auth()
        self._reconcile_kaggle()
        self._reconcile_colab()
        self._reconcile_github()
        self._fill_slots()
        self._snapshot(force=snapshot)

    def run_forever(self) -> None:
        """Run bounded cycles until every seed is complete or a stop file appears."""

        stop_path = self.state_root / "STOP"
        while not stop_path.exists():
            self.cycle()
            if all(record["state"] == "completed" for record in self.records.values()):
                self._snapshot(force=True)
                return
            time.sleep(int(self.config["poll_seconds"]))


def acquire_singleton_lock(path: Path) -> Any:
    """Acquire and retain a nonblocking Windows byte-range process lock."""

    import msvcrt

    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    if path.stat().st_size == 0:
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    try:
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as error:
        stream.close()
        raise RuntimeError("another Phase 12 scheduler already owns the singleton lock") from error
    return stream


def main() -> None:
    """Parse CLI arguments and run one or repeated scheduler cycles."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--state-root", type=Path, default=Path("outputs/runs/phase12_scheduler"))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    project_root = arguments.project_root.resolve()
    state_root = (
        arguments.state_root
        if arguments.state_root.is_absolute()
        else project_root / arguments.state_root
    )
    lock = acquire_singleton_lock(state_root / "scheduler.lock")
    try:
        scheduler = Phase12Scheduler(
            project_root=project_root,
            state_root=state_root,
            dry_run=arguments.dry_run,
        )
        if arguments.once:
            scheduler.cycle(snapshot=arguments.snapshot)
        else:
            scheduler.run_forever()
    finally:
        lock.close()


if __name__ == "__main__":
    main()
