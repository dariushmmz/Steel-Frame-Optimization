"""Prepare a private one-seed Kaggle kernel without changing campaign inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

CAMPAIGNS: Final = ("case1_filtered", "case1_no_filter", "case2")


def _slug(campaign: str, seed: int) -> str:
    """Return a stable private Kaggle kernel slug for one campaign seed."""

    return f"steel-frame-phase-12-{campaign.replace('_', '-')}-seed-{seed:02d}"


def prepare_kernel(*, campaign: str, seed: int, destination: Path) -> Path:
    """Write one thin assignment wrapper and its private kernel metadata."""

    if campaign not in CAMPAIGNS:
        raise ValueError(f"unknown campaign: {campaign}")
    if seed not in range(30):
        raise ValueError("seed must be between 0 and 29")
    destination.mkdir(parents=True, exist_ok=True)
    wrapper = destination / "run_seed.py"
    wrapper.write_text(
        "\n".join(
            (
                '"""Launch one frozen Phase 12 seed from the private support dataset."""',
                "",
                "import importlib.util",
                "from pathlib import Path",
                "",
                "matches = sorted(Path('/kaggle/input').rglob('kaggle_phase12_seed_job.py'))",
                "if len(matches) != 1:",
                "    raise RuntimeError(f'expected one seed job module, found {matches}')",
                "spec = importlib.util.spec_from_file_location('phase12_seed_job', matches[0])",
                "if spec is None or spec.loader is None:",
                "    raise RuntimeError('cannot load the Phase 12 seed job module')",
                "module = importlib.util.module_from_spec(spec)",
                "spec.loader.exec_module(module)",
                f"module.run_job(campaign={campaign!r}, seed={seed})",
                "",
            )
        ),
        encoding="utf-8",
        newline="\n",
    )
    metadata = {
        "id": f"dariushmzx/{_slug(campaign, seed)}",
        "title": f"Steel Frame Phase 12 {campaign} Seed {seed:02d}",
        "code_file": wrapper.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [
            "dariushmzx/steel-frame-phase12-frozen-project",
            "dariushmzx/steel-frame-phase12-recovery-state",
        ],
        "competition_sources": [],
        "kernel_sources": ["dariushmzx/steel-frame-phase-12-python-runtime"],
    }
    metadata_path = destination / "kernel-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata_path


def main() -> None:
    """Create the requested generated kernel directory."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", choices=CAMPAIGNS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    arguments = parser.parse_args()
    path = prepare_kernel(
        campaign=arguments.campaign,
        seed=arguments.seed,
        destination=arguments.destination,
    )
    print(path)


if __name__ == "__main__":
    main()
