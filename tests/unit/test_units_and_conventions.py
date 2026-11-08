"""Layer 1 tests for unit boundaries, dimensions, conventions, and tolerances."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from steel_frame_optimization.config.conventions import (
    CANONICAL_COORDINATE_CONVENTION,
    AxialSign,
    Axis,
    RotationSign,
)
from steel_frame_optimization.config.units import (
    CANONICAL_UNITS,
    Dimension,
    UnitError,
    convert_value,
    to_canonical,
)

pytestmark = pytest.mark.unit

ROUND_TRIP_RELATIVE_TOLERANCE = 1e-12
ROUND_TRIP_ABSOLUTE_TOLERANCE = 1e-15


@pytest.mark.parametrize(
    ("dimension", "canonical", "alternate", "value"),
    [
        (Dimension.DIMENSIONLESS, "1", "1", 1.25),
        (Dimension.LENGTH, "m", "ft", 12.345),
        (Dimension.FORCE, "N", "kip", 98_765.0),
        (Dimension.MASS, "kg", "lbm", 4321.0),
        (Dimension.TIME, "s", "ms", 12.5),
        (Dimension.STRESS, "Pa", "ksi", 345_000_000.0),
        (Dimension.ANGLE, "rad", "deg", math.pi / 3.0),
    ],
)
def test_round_trip_for_every_canonical_unit(
    dimension: Dimension,
    canonical: str,
    alternate: str,
    value: float,
) -> None:
    """Round-trip one alternate unit for every selected canonical dimension."""

    assert CANONICAL_UNITS[dimension] == canonical
    converted = convert_value(value, canonical, alternate)
    restored = convert_value(converted, alternate, canonical)
    assert math.isclose(
        restored,
        value,
        rel_tol=ROUND_TRIP_RELATIVE_TOLERANCE,
        abs_tol=ROUND_TRIP_ABSOLUTE_TOLERANCE,
    )


def test_to_canonical_returns_value_and_symbol() -> None:
    """Make canonicalization explicit at data-ingestion boundaries."""

    value, symbol = to_canonical(1.0, "kN")
    assert value == 1000.0
    assert symbol == "N"


def test_incompatible_dimensions_fail() -> None:
    """Reject mixed dimensional arithmetic at the conversion boundary."""

    with pytest.raises(UnitError, match="incompatible dimensions"):
        convert_value(1.0, "m", "N")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_fail(value: float) -> None:
    """Reject non-finite values before they enter configurations or analyses."""

    with pytest.raises(UnitError, match="finite"):
        convert_value(value, "m", "mm")


def test_unknown_unit_fails() -> None:
    """Require every unit symbol to be registered and reviewable."""

    with pytest.raises(UnitError, match="unknown unit"):
        convert_value(1.0, "furlong", "m")


def test_coordinate_and_sign_convention_is_explicit() -> None:
    """Freeze storage orientation without resolving future hinge engineering policy."""

    convention = CANONICAL_COORDINATE_CONVENTION
    assert (convention.horizontal_axis, convention.vertical_axis) == (Axis.X, Axis.Y)
    assert convention.out_of_plane_axis is Axis.Z
    assert convention.gravity_direction == "negative_y"
    assert convention.axial_sign is AxialSign.TENSION_POSITIVE
    assert convention.rotation_sign is RotationSign.COUNTERCLOCKWISE_POSITIVE


def test_tolerance_taxonomy_records_delegated_research_policy(project_root: Path) -> None:
    """Require exact software tolerances and predeclared delegated research thresholds."""

    path = project_root / "configs" / "analysis" / "tolerance_taxonomy.json"
    taxonomy = json.loads(path.read_text(encoding="utf-8"))
    assert taxonomy["exact_identity"]["absolute_tolerance"] == 0.0
    assert taxonomy["unit_conversion_round_trip"]["relative_tolerance"] == 1e-12
    research = taxonomy["research_comparison"]
    assert research["status"] == "agent_decided_pending_ratification_DEC-020"
    assert research["published_objective_relative_tolerance"] == 0.05
    assert research["required_independent_runs"] == 30
    assert taxonomy["engineering_acceptance"]["absolute_tolerance"] is None
