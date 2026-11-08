"""Provide reproducibility commands and the deterministic Phase 8 evaluator."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from steel_frame_optimization import __version__
from steel_frame_optimization.models.manifests import ArtifactManifest
from steel_frame_optimization.optimization.campaign import CampaignApprovalError, CampaignRunner
from steel_frame_optimization.optimization.evaluator import EvaluatorPaths, StructuralEvaluator
from steel_frame_optimization.provenance import (
    collect_runtime_provenance,
    verify_artifact_manifest,
)
from steel_frame_optimization.serialization import canonical_json_dumps, write_canonical_json


def _parse_datetime(value: str) -> datetime:
    """Parse an ISO-8601 timestamp while preserving its explicit offset."""

    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected an ISO-8601 timestamp") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a UTC offset")
    return timestamp


def _parse_section_assignment(value: str) -> tuple[str, str]:
    """Parse one explicit ``GROUP=SECTION`` command-line assignment."""

    try:
        group, section = value.split("=", maxsplit=1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected GROUP=SECTION") from error
    if not group or not section:
        raise argparse.ArgumentTypeError("expected nonempty GROUP=SECTION")
    return group, section


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser used by console and module entry points."""

    parser = argparse.ArgumentParser(
        prog="steel-frame-opt",
        description=(
            "Evaluate and optimize the delivered three-story steel frame with the "
            "two-stage LRFD-PBD workflow."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    provenance = subparsers.add_parser("provenance", help="record runtime provenance")
    provenance.add_argument("--project-root", type=Path, default=Path.cwd())
    provenance.add_argument("--output", type=Path)
    provenance.add_argument("--run-id")
    provenance.add_argument("--created-at-utc", type=_parse_datetime)

    subparsers.add_parser("test-unit", help="run layer-1 unit tests")
    subparsers.add_parser("test-integration", help="run integration tests")
    evaluate = subparsers.add_parser("evaluate", help="run one deterministic structural evaluation")
    evaluate.add_argument("--project-root", type=Path, default=Path.cwd())
    evaluate.add_argument("--evaluation-id", required=True)
    evaluate.add_argument(
        "--section",
        type=_parse_section_assignment,
        action="append",
        required=True,
        help="design assignment GROUP=SECTION; repeat for every group",
    )
    evaluate.add_argument("--output", type=Path)
    campaign = subparsers.add_parser("campaign", help="run a configured GA pilot or campaign")
    campaign.add_argument("--project-root", type=Path, default=Path.cwd())
    campaign.add_argument("--config", type=Path, required=True)
    campaign.add_argument("--checkpoint-dir", type=Path, required=True)
    campaign.add_argument("--output", type=Path)
    campaign.add_argument(
        "--resume",
        action="store_true",
        help="resume every seed with an existing identity-checked checkpoint",
    )
    subparsers.add_parser("report", help="generate a structural report (future phase)")
    verification = subparsers.add_parser(
        "verify-artifacts", help="verify files against an artifact manifest"
    )
    verification.add_argument("manifest", type=Path)
    verification.add_argument("--root", type=Path, default=Path.cwd())
    return parser


def _run_pytest(arguments: Sequence[str]) -> int:
    """Run pytest in the active locked interpreter without invoking uv or an implicit sync."""

    result = subprocess.run(
        [sys.executable, "-m", "pytest", *arguments],
        check=False,
    )
    return result.returncode


def _not_implemented(capability: str, required_phase: str) -> int:
    """Return an explicit non-zero failure for a future-phase capability."""

    print(
        f"not implemented: {capability} requires {required_phase}; "
        "this capability is not available in the delivered release",
        file=sys.stderr,
    )
    return 2


def _verify_artifacts(manifest_path: Path, root: Path) -> int:
    """Load one strict artifact manifest and report every deterministic mismatch."""

    try:
        manifest = ArtifactManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"artifact verification failed: {error}", file=sys.stderr)
        return 3
    mismatches = verify_artifact_manifest(manifest, root)
    if mismatches:
        for mismatch in mismatches:
            print(f"artifact verification failed: {mismatch}", file=sys.stderr)
        return 4
    print(f"verified {len(manifest.files)} artifact(s)")
    return 0


def _evaluate_design(arguments: argparse.Namespace) -> int:
    """Run one Phase 8 evaluation and emit its canonical compact result."""

    assignments = dict(arguments.section)
    if len(assignments) != len(arguments.section):
        print("evaluation failed: duplicate design group assignment", file=sys.stderr)
        return 3
    evaluator = StructuralEvaluator(EvaluatorPaths.from_project_root(arguments.project_root))
    execution = evaluator.evaluate(
        evaluation_id=arguments.evaluation_id,
        section_names_by_group=assignments,
    )
    if arguments.output is None:
        print(canonical_json_dumps(execution), end="")
    else:
        write_canonical_json(arguments.output, execution)
    return 3 if execution.result.rank_class == 2 else 0


def _run_campaign(arguments: argparse.Namespace) -> int:
    """Run configured seeds while enforcing the owner-controlled production approval gate."""

    try:
        runner = CampaignRunner(
            project_root=arguments.project_root,
            policy_path=arguments.config,
        )
        results = runner.run_all(
            checkpoint_directory=arguments.checkpoint_dir,
            resume_existing=arguments.resume,
        )
    except CampaignApprovalError as error:
        print(f"campaign blocked: {error}", file=sys.stderr)
        return 5
    except (OSError, ValueError, RuntimeError) as error:
        print(f"campaign failed: {error}", file=sys.stderr)
        return 3
    if arguments.output is None:
        print(canonical_json_dumps(results), end="")
    else:
        write_canonical_json(arguments.output, results)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments, print help, and return a process exit code."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "provenance":
        record = collect_runtime_provenance(
            arguments.project_root,
            run_id=arguments.run_id,
            created_at_utc=arguments.created_at_utc,
        )
        if arguments.output is None:
            print(canonical_json_dumps(record), end="")
        else:
            write_canonical_json(arguments.output, record)
        return 0
    if arguments.command == "test-unit":
        return _run_pytest(("tests/unit", "-m", "unit"))
    if arguments.command == "test-integration":
        return _run_pytest(("tests/integration", "-m", "integration"))
    if arguments.command == "evaluate":
        return _evaluate_design(arguments)
    if arguments.command == "campaign":
        return _run_campaign(arguments)
    if arguments.command == "report":
        return _not_implemented("structural report generation", "Phase 14")
    if arguments.command == "verify-artifacts":
        return _verify_artifacts(arguments.manifest, arguments.root)
    parser.print_help()
    return 0
