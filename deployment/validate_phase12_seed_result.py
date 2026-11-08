"""Validate one persisted Phase 12 seed result against the frozen manifest."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, cast


def _load_scheduler() -> Any:
    """Load the sibling scheduler without relying on deployment package installation."""

    path = Path(__file__).with_name("phase12_multiplatform_scheduler.py")
    specification = importlib.util.spec_from_file_location(path.stem, path)
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the Phase 12 scheduler module")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def main() -> None:
    """Parse artifact paths, run strict validation, and print compact evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign", choices=("case1_filtered", "case1_no_filter", "case2"), required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.seed not in range(30):
        raise ValueError("seed must be between 0 and 29")
    manifest = cast(dict[str, Any], json.loads(arguments.manifest.read_text(encoding="utf-8")))
    scheduler = _load_scheduler()
    key = scheduler.SeedKey(arguments.campaign, arguments.seed)
    validation = scheduler.validate_seed_result(
        result_path=arguments.result,
        status_path=arguments.status,
        campaign_record=manifest["campaigns"][arguments.campaign],
        key=key,
    )
    print(json.dumps({"seed": key.token, "valid": True, **validation}, indent=2))


if __name__ == "__main__":
    main()
