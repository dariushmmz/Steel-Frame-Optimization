"""Expose explicit configuration, unit, and convention boundaries."""

from steel_frame_optimization.config.conventions import CANONICAL_COORDINATE_CONVENTION
from steel_frame_optimization.config.schemas import (
    AnalysisCase,
    DecisionVariant,
    FrameSpec,
    OptimizationCampaign,
    SolverPolicy,
)
from steel_frame_optimization.config.units import (
    CANONICAL_UNITS,
    Dimension,
    UnitDefinition,
    UnitError,
    convert_value,
    get_unit,
    to_canonical,
)

__all__ = [
    "CANONICAL_COORDINATE_CONVENTION",
    "CANONICAL_UNITS",
    "AnalysisCase",
    "DecisionVariant",
    "Dimension",
    "FrameSpec",
    "OptimizationCampaign",
    "SolverPolicy",
    "UnitDefinition",
    "UnitError",
    "convert_value",
    "get_unit",
    "to_canonical",
]
