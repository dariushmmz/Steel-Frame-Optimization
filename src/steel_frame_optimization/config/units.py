"""Enforce one explicit SI unit boundary without performing structural calculations.

Values cross the project boundary with a unit symbol and are converted once to the canonical
N-m-kg-s-Pa-rad representation. The registry contains definitions only; it does not supply material,
section, loading, or code-provision values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final


class Dimension(StrEnum):
    """Name each dimensional family supported at Phase 1 input boundaries."""

    DIMENSIONLESS = "dimensionless"
    LENGTH = "length"
    FORCE = "force"
    MASS = "mass"
    TIME = "time"
    STRESS = "stress"
    ANGLE = "angle"


@dataclass(frozen=True, slots=True)
class UnitDefinition:
    """Describe a unit by dimension and multiplicative scale to its canonical unit."""

    symbol: str
    dimension: Dimension
    scale_to_canonical: float


CANONICAL_UNITS: Final = MappingProxyType(
    {
        Dimension.DIMENSIONLESS: "1",
        Dimension.LENGTH: "m",
        Dimension.FORCE: "N",
        Dimension.MASS: "kg",
        Dimension.TIME: "s",
        Dimension.STRESS: "Pa",
        Dimension.ANGLE: "rad",
    }
)

UNIT_DEFINITIONS: Final = MappingProxyType(
    {
        "1": UnitDefinition("1", Dimension.DIMENSIONLESS, 1.0),
        "m": UnitDefinition("m", Dimension.LENGTH, 1.0),
        "mm": UnitDefinition("mm", Dimension.LENGTH, 0.001),
        "in": UnitDefinition("in", Dimension.LENGTH, 0.0254),
        "ft": UnitDefinition("ft", Dimension.LENGTH, 0.3048),
        "N": UnitDefinition("N", Dimension.FORCE, 1.0),
        "kN": UnitDefinition("kN", Dimension.FORCE, 1000.0),
        "lbf": UnitDefinition("lbf", Dimension.FORCE, 4.4482216152605),
        "kip": UnitDefinition("kip", Dimension.FORCE, 4448.2216152605),
        "kg": UnitDefinition("kg", Dimension.MASS, 1.0),
        "t": UnitDefinition("t", Dimension.MASS, 1000.0),
        "lbm": UnitDefinition("lbm", Dimension.MASS, 0.45359237),
        "s": UnitDefinition("s", Dimension.TIME, 1.0),
        "ms": UnitDefinition("ms", Dimension.TIME, 0.001),
        "Pa": UnitDefinition("Pa", Dimension.STRESS, 1.0),
        "kPa": UnitDefinition("kPa", Dimension.STRESS, 1000.0),
        "MPa": UnitDefinition("MPa", Dimension.STRESS, 1_000_000.0),
        "GPa": UnitDefinition("GPa", Dimension.STRESS, 1_000_000_000.0),
        "psi": UnitDefinition("psi", Dimension.STRESS, 6894.757293168),
        "ksi": UnitDefinition("ksi", Dimension.STRESS, 6_894_757.293168),
        "rad": UnitDefinition("rad", Dimension.ANGLE, 1.0),
        "deg": UnitDefinition("deg", Dimension.ANGLE, math.pi / 180.0),
    }
)


class UnitError(ValueError):
    """Indicate an unknown unit, incompatible dimensions, or non-finite value."""


def get_unit(symbol: str) -> UnitDefinition:
    """Return a registered unit or raise a clear boundary-validation error."""

    try:
        return UNIT_DEFINITIONS[symbol]
    except KeyError as error:
        raise UnitError(f"unknown unit: {symbol}") from error


def convert_value(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a finite scalar between units of the same declared dimension."""

    if not math.isfinite(value):
        raise UnitError("unit conversion requires a finite value")
    source = get_unit(from_unit)
    target = get_unit(to_unit)
    if source.dimension is not target.dimension:
        raise UnitError(
            f"incompatible dimensions: {from_unit} is {source.dimension.value}, "
            f"{to_unit} is {target.dimension.value}"
        )
    return value * source.scale_to_canonical / target.scale_to_canonical


def to_canonical(value: float, from_unit: str) -> tuple[float, str]:
    """Convert a value to its dimension's canonical unit and return value plus symbol."""

    source = get_unit(from_unit)
    canonical_unit = CANONICAL_UNITS[source.dimension]
    return convert_value(value, from_unit, canonical_unit), canonical_unit
