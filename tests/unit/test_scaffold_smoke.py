"""Layer 1 smoke tests for the pinned interpreter, imports, and CLI."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

import steel_frame_optimization

pytestmark = pytest.mark.unit

SUBPACKAGES = [
    "analysis",
    "constraints",
    "config",
    "sections",
    "models",
    "objective",
    "optimization",
    "validation",
    "reporting",
]
STUB_MODULES = [
    "analysis.modal",
    "analysis.elastic",
    "analysis.hinges",
    "analysis.pushover",
    "analysis.convergence",
    "constraints.lrfd",
    "constraints.serviceability",
    "constraints.fema",
    "constraints.hierarchy",
]


def test_python_version_is_owner_pinned() -> None:
    """Reject execution outside the exact owner-approved interpreter version."""

    assert sys.version_info[:3] == (3, 12, 6)


def test_package_version() -> None:
    """Keep the package version accessible to the CLI and later provenance records."""

    assert steel_frame_optimization.__version__ == "0.1.0"


@pytest.mark.parametrize("name", SUBPACKAGES + STUB_MODULES)
def test_submodule_imports(name: str) -> None:
    """Ensure architecture placeholders import without hidden setup or side effects."""

    importlib.import_module(f"steel_frame_optimization.{name}")


def test_openseespy_imports() -> None:
    """Import and wipe OpenSeesPy only; this is not structural validation."""

    ops = importlib.import_module("openseespy.opensees")
    ops.wipe()


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the module entry point in an isolated child interpreter."""

    return subprocess.run(
        [sys.executable, "-m", "steel_frame_optimization", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_help() -> None:
    """Expose usable top-level help with a successful process status."""

    result = _run_cli("--help")
    assert result.returncode == 0
    assert "steel-frame-opt" in result.stdout


def test_cli_version() -> None:
    """Expose the package version without importing future structural code."""

    result = _run_cli("--version")
    assert result.returncode == 0
    assert "0.1.0" in result.stdout


def test_cli_provenance_with_declared_variable_fields() -> None:
    """Emit parseable provenance without relying on an implicit clock or random run ID."""

    result = _run_cli(
        "provenance",
        "--run-id",
        "cli-test",
        "--created-at-utc",
        "2026-10-27T00:00:00+00:00",
    )
    assert result.returncode == 0
    assert '"run_id":"cli-test"' in result.stdout
    assert '"uv_version":"uv 0.6.14 (a4cec56dc 2025-04-09)"' in result.stdout
