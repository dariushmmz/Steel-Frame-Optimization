"""Verify restart-safe Phase 12 scheduling without launching cloud work."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_scheduler_module() -> Any:
    """Load the deployment script while registering its dataclass module."""

    path = Path(__file__).resolve().parents[2] / "deployment/phase12_multiplatform_scheduler.py"
    specification = importlib.util.spec_from_file_location(path.stem, path)
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load Phase 12 scheduler")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


SCHEDULER = _load_scheduler_module()
Phase12Scheduler = SCHEDULER.Phase12Scheduler
SeedKey = SCHEDULER.SeedKey
choose_next_seed = SCHEDULER.choose_next_seed
choose_github_seed = SCHEDULER.choose_github_seed
validate_github_status_binding = SCHEDULER.validate_github_status_binding


def _scheduler(tmp_path: Path) -> Any:
    """Return a dry-run scheduler backed by an isolated persistent state file."""

    root = Path(__file__).resolve().parents[2]
    return Phase12Scheduler(project_root=root, state_root=tmp_path, dry_run=True)


def test_scheduler_prefers_advanced_seed_without_duplicate_ownership() -> None:
    """Assign the most advanced eligible seed while excluding every owned seed."""

    records: dict[str, dict[str, Any]] = {
        "case1_filtered:04": {
            "campaign": "case1_filtered",
            "seed": 4,
            "state": "running",
        },
        "case1_no_filter:11": {
            "campaign": "case1_no_filter",
            "seed": 11,
            "state": "pending",
        },
        "case2:13": {"campaign": "case2", "seed": 13, "state": "pending"},
    }

    selected = choose_next_seed(
        records,
        {"case1_filtered:04": 14_900, "case1_no_filter:11": 16_800, "case2:13": 13_700},
    )

    assert selected == SeedKey("case1_no_filter", 11)
    assert selected.token != "case1_filtered:04"


def test_github_selects_only_fresh_unclaimed_seed() -> None:
    """Never transfer a private recovery checkpoint through the public repository."""

    records: dict[str, dict[str, Any]] = {
        "case1_filtered:08": {
            "campaign": "case1_filtered",
            "seed": 8,
            "state": "pending",
        },
        "case1_no_filter:11": {
            "campaign": "case1_no_filter",
            "seed": 11,
            "state": "pending",
        },
        "case2:13": {"campaign": "case2", "seed": 13, "state": "running"},
    }

    selected = choose_github_seed(
        records,
        {"case1_filtered:08": 0, "case1_no_filter:11": 13_700, "case2:13": 0},
    )

    assert selected == SeedKey("case1_filtered", 8)


def test_github_chunk_validation_rejects_wrong_assignment(
    tmp_path: Path,
) -> None:
    """Bind every downloaded artifact to its persistent scheduler lease."""

    scheduler = _scheduler(tmp_path)
    key = SeedKey("case2", 13)
    record = scheduler.records[key.token]
    record.update(
        {
            "state": "running",
            "platform": "github",
            "github_assignment_token": "expected-e040",
            "github_target_epochs": 40,
        }
    )
    status = {
        "success": True,
        "platform": "github_actions",
        "assignment_token": "wrong",
        "campaign": "case2",
        "seed": 13,
    }

    with pytest.raises(ValueError, match="assignment"):
        validate_github_status_binding(status=status, key=key, record=record)


def test_github_dry_run_leases_seed_before_dispatch(tmp_path: Path) -> None:
    """Persist exclusive ownership before any remote workflow can be submitted."""

    scheduler = _scheduler(tmp_path)
    key = SeedKey("case1_filtered", 8)

    scheduler._launch_github(key)

    record = scheduler.records[key.token]
    assert record["state"] == "submitting"
    assert record["platform"] == "github"
    assert record["remote_id"] is None
    assert record["github_target_epochs"] == 40
    assert record["lease_token"].startswith("case1_filtered-08-")


def test_scheduler_restart_preserves_ownership_and_queue(tmp_path: Path) -> None:
    """Reload the atomic state without creating a second assignment."""

    first = _scheduler(tmp_path)
    first.records["case1_no_filter:11"].update(
        {
            "state": "running",
            "platform": "kaggle",
            "remote_id": "dariushmzx/example",
        }
    )
    first._save()

    restored = _scheduler(tmp_path)

    assert restored.records["case1_no_filter:11"]["state"] == "running"
    assert restored.records["case1_no_filter:11"]["remote_id"] == "dariushmzx/example"
    assert len(restored.records) == 90


def test_transient_kaggle_failure_does_not_release_or_duplicate_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain ownership when a status API call cannot prove remote termination."""

    scheduler = _scheduler(tmp_path)
    monkeypatch.setattr(
        scheduler,
        "_kaggle_status",
        lambda remote_id: (None, "temporary TLS EOF"),
    )

    scheduler._reconcile_kaggle()

    active = scheduler.records["case2:10"]
    assert active["state"] == "running"
    assert active["platform"] == "kaggle"
    assert "transient Kaggle" in active["last_error"]


def test_verified_completion_opens_slot_and_dispatches_next_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed owner frees exactly one slot for the next eligible seed."""

    scheduler = _scheduler(tmp_path)
    completed = scheduler.records["case1_filtered:04"]
    completed.update({"state": "completed", "validation": {"evaluation_requests": 20_000}})
    dispatched: list[Any] = []

    def capture(key: Any) -> None:
        dispatched.append(key)
        scheduler.records[key.token].update(
            {"state": "running", "platform": "kaggle", "remote_id": "test"}
        )

    monkeypatch.setattr(scheduler, "_launch_kaggle", capture)
    scheduler._fill_slots()

    assert len(dispatched) == 2
    assert dispatched[0] == SeedKey("case1_no_filter", 11)
    assert dispatched[1] == SeedKey("case2", 13)
    assert len({key.token for key in dispatched}) == len(dispatched)


def test_snapshot_uses_exact_completed_and_checkpoint_accounting(tmp_path: Path) -> None:
    """Report only durable requests and always count a completed seed as 20,000."""

    scheduler = _scheduler(tmp_path)
    scheduler.records["case1_filtered:04"]["state"] = "completed"
    scheduler._snapshot(force=True)

    snapshot = scheduler.snapshots_path.read_text(encoding="utf-8").splitlines()[-1]

    assert '"total_requests": 1800000' in snapshot
    assert '"committed_requests_lower_bound"' in snapshot


def test_expired_colab_session_rebuild_waits_for_drive_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bootstrap a fresh VM and resume the same Drive owner without opening a new seed."""

    scheduler = _scheduler(tmp_path)
    record = scheduler.records["case1_filtered:10"]
    record["recovery_attempted_at"] = "previous-successful-replacement"
    commands: list[str] = []

    def command(shell_command: str, *, timeout: int = 180) -> Any:
        commands.append(shell_command)
        return SCHEDULER.CommandResult(return_code=0, output="ok")

    monkeypatch.setattr(scheduler, "_colab_command", command)

    scheduler._recover_expired_colab(SeedKey("case1_filtered", 10), record)

    assert record["state"] == "blocked"
    assert "Drive mount" in record["last_error"]
    assert any("colab new" in item for item in commands)
    assert not any("drivemount" in item for item in commands)


def test_manual_drive_authorization_retries_only_the_blocked_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Continue the same seed after a new VM's unavoidable interactive Drive consent."""

    scheduler = _scheduler(tmp_path)
    record = scheduler.records["case1_filtered:10"]
    record["state"] = "blocked"
    record["recovery_attempted_at"] = "prior-expiration"
    (tmp_path / "RETRY_COLAB").write_text("approved\n", encoding="utf-8")
    resumed: list[str] = []

    monkeypatch.setattr(
        scheduler,
        "_colab_command",
        lambda command, timeout=180: SCHEDULER.CommandResult(
            return_code=0, output="phase12-python-3.12.6-runtime.tar.gz"
        ),
    )

    def bootstrap(key: Any, record: dict[str, Any], *, wsl_restore_script: str) -> None:
        resumed.append(key.token)
        record["state"] = "running"

    monkeypatch.setattr(scheduler, "_bootstrap_colab_owned_seed", bootstrap)

    scheduler._resume_colab_after_manual_drive_auth()

    assert resumed == ["case1_filtered:10"]
    assert "recovery_attempted_at" not in record
    assert not (tmp_path / "RETRY_COLAB").exists()


def test_retry_marker_recreates_an_expired_blocked_colab_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recreate one missing blocked VM without releasing or duplicating its seed lease."""

    scheduler = _scheduler(tmp_path)
    record = scheduler.records["case1_filtered:10"]
    record.update(
        {
            "state": "blocked",
            "platform": "colab",
            "remote_id": "phase12-filtered",
            "recovery_attempted_at": "prior-expiration",
        }
    )
    marker = tmp_path / "RETRY_COLAB"
    marker.write_text("approved\n", encoding="utf-8")
    commands: list[str] = []

    def command(shell_command: str, *, timeout: int = 180) -> Any:
        commands.append(shell_command)
        if " status " in shell_command:
            return SCHEDULER.CommandResult(return_code=1, output="Session not found")
        return SCHEDULER.CommandResult(return_code=0, output="Session READY")

    monkeypatch.setattr(scheduler, "_colab_command", command)

    scheduler._resume_colab_after_manual_drive_auth()

    assert record["state"] == "blocked"
    assert record["platform"] == "colab"
    assert record["remote_id"] == "phase12-filtered"
    assert "Drive mount" in record["last_error"]
    assert any("colab new" in item for item in commands)
    assert not marker.exists()


def test_replacement_colab_bootstrap_uploads_worker_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Install both metadata and the executable worker on every replacement VM."""

    scheduler = _scheduler(tmp_path)
    record = scheduler.records["case1_filtered:10"]
    commands: list[str] = []
    launches: list[str] = []

    def command(shell_command: str, *, timeout: int = 180) -> Any:
        commands.append(shell_command)
        return SCHEDULER.CommandResult(return_code=0, output="ok")

    def launch(
        key: Any,
        *,
        remote_source: str,
        source_sha: str,
        start_fresh: bool,
        recovery_run: bool,
    ) -> None:
        launches.append(key.token)

    monkeypatch.setattr(scheduler, "_colab_command", command)
    monkeypatch.setattr(scheduler, "_spawn_colab_launcher", launch)

    scheduler._bootstrap_colab_owned_seed(
        SeedKey("case1_filtered", 10),
        record,
        wsl_restore_script="/mnt/f/project/deployment/colab_phase12_restore_runtime.py",
    )

    assert any("/content/phase12_recovery_manifest.json" in item for item in commands)
    assert any("/content/colab_phase12_seed_job.py" in item for item in commands)
    assert launches == ["case1_filtered:10"]
    assert record["state"] == "running"
