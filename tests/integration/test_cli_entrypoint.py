"""Integration test for the installed console-script entry point."""

from __future__ import annotations

import subprocess

import pytest

pytestmark = pytest.mark.integration


def test_installed_console_help() -> None:
    """Run the installed executable and verify successful command discovery."""

    result = subprocess.run(
        ["steel-frame-opt", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "verify-artifacts" in result.stdout
