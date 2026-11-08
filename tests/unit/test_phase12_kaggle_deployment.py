"""Verify deterministic generation of private Phase 12 Kaggle seed assignments."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load_deployment_module(name: str) -> ModuleType:
    """Load one repository deployment script without making deployment a package."""

    path = Path(__file__).resolve().parents[2] / "deployment" / name
    specification = importlib.util.spec_from_file_location(path.stem, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load deployment module {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_phase12_kaggle_assignment_is_private_cpu_and_seed_specific(tmp_path: Path) -> None:
    """Keep each frozen seed in a unique private CPU-only Kaggle kernel."""

    module = _load_deployment_module("prepare_kaggle_phase12_kernel.py")
    metadata_path = module.prepare_kernel(campaign="case1_no_filter", seed=6, destination=tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    wrapper = (tmp_path / "run_seed.py").read_text(encoding="utf-8")

    assert metadata["id"].endswith("case1-no-filter-seed-06")
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is False
    assert metadata["enable_internet"] is False
    assert any(
        source.endswith("steel-frame-phase12-recovery-state")
        for source in metadata["dataset_sources"]
    )
    assert any(
        source.endswith("steel-frame-phase-12-python-runtime")
        for source in metadata["kernel_sources"]
    )
    assert "campaign='case1_no_filter', seed=6" in wrapper


@pytest.mark.parametrize("seed", (-1, 30))
def test_phase12_kaggle_assignment_rejects_out_of_range_seed(tmp_path: Path, seed: int) -> None:
    """Reject assignments outside the immutable Phase 12 seed manifest."""

    module = _load_deployment_module("prepare_kaggle_phase12_kernel.py")
    with pytest.raises(ValueError, match="between 0 and 29"):
        module.prepare_kernel(campaign="case2", seed=seed, destination=tmp_path)


def test_phase12_kaggle_recovery_copy_requires_manifest_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Copy only the uniquely identified checkpoint declared by the manifest."""

    module = _load_deployment_module("kaggle_phase12_seed_job.py")
    source = tmp_path / "input" / "checkpoints" / "case2" / "seed_09.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"checkpoint":"frozen"}\n', encoding="utf-8")
    monkeypatch.setattr(module, "INPUT_ROOT", tmp_path / "input")
    expected_hash = module.sha256_file(source)
    manifest = {"checkpoints": [{"campaign": "case2", "seed": 9, "sha256": expected_hash}]}
    destination = tmp_path / "working" / "seed_09.json"

    actual_hash = module._copy_recovery_checkpoint(
        manifest=manifest, campaign="case2", seed=9, destination=destination
    )

    assert actual_hash == expected_hash
    assert destination.read_bytes() == source.read_bytes()


def test_phase12_kaggle_recovery_copy_rejects_changed_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stop recovery before execution when transferred checkpoint bytes differ."""

    module = _load_deployment_module("kaggle_phase12_seed_job.py")
    source = tmp_path / "input" / "checkpoints" / "case2" / "seed_09.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"checkpoint":"changed"}\n', encoding="utf-8")
    monkeypatch.setattr(module, "INPUT_ROOT", tmp_path / "input")
    manifest = {"checkpoints": [{"campaign": "case2", "seed": 9, "sha256": "0" * 64}]}

    with pytest.raises(RuntimeError, match="hash differs"):
        module._copy_recovery_checkpoint(
            manifest=manifest,
            campaign="case2",
            seed=9,
            destination=tmp_path / "working" / "seed_09.json",
        )


def test_phase12_colab_atomic_copy_preserves_checkpoint_bytes(tmp_path: Path) -> None:
    """Accept a Drive-style checkpoint copy only after its hash rereads unchanged."""

    module = _load_deployment_module("colab_phase12_seed_job.py")
    source = tmp_path / "source.json"
    destination = tmp_path / "drive" / "seed_10_epoch_124.json"
    source.write_text('{"checkpoint":"frozen"}\n', encoding="utf-8")

    copied_hash = module.atomic_copy_verified(source, destination)

    assert copied_hash == module.sha256_file(source)
    assert destination.read_bytes() == source.read_bytes()
    assert not destination.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize(
    ("name", "expected_epoch"),
    (("seed_10_epoch_124.json", 124), ("seed_10_epoch_200.json", 200)),
)
def test_phase12_colab_checkpoint_epoch_is_explicit(name: str, expected_epoch: int) -> None:
    """Order retained recovery files only by their explicit epoch suffix."""

    module = _load_deployment_module("colab_phase12_seed_job.py")

    assert module.checkpoint_epoch(Path(name)) == expected_epoch
