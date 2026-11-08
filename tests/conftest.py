"""Provide shared pytest fixtures for repository-aware tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    """Return the repository root independently of the current working directory."""

    return Path(__file__).resolve().parents[1]
