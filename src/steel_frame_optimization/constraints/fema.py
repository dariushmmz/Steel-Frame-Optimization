"""Evaluate FEMA hinge rotations, force-controlled columns, and performance drifts."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field

from steel_frame_optimization.analysis.hinges import FemaHingeLimits
from steel_frame_optimization.config.schemas import StrictConfigModel

PerformanceLevel = Literal["OP", "IO", "LS", "CP"]


class PerformanceConstraint(StrictConfigModel):
    """Persist one source-traceable nonlinear performance constraint."""

    check_id: str = Field(min_length=1)
    check_type: Literal["hinge_rotation", "force_controlled_column", "story_drift", "hierarchy"]
    target_id: str = Field(min_length=1)
    performance_level: PerformanceLevel
    subject_id: str = Field(min_length=1)
    demand: float = Field(ge=0.0, allow_inf_nan=False)
    capacity: float = Field(gt=0.0, allow_inf_nan=False)
    utilization: float = Field(ge=0.0, allow_inf_nan=False)
    passed: bool
    primary_acceptance: bool
    source: str = Field(min_length=1)
    details: dict[str, str | float | bool] = Field(default_factory=dict)


def plastic_rotation_demand(
    *,
    total_spring_rotation_rad: float,
    spring_moment_nm: float,
    initial_stiffness_nm_per_rad: float,
) -> float:
    """Return the nonnegative inelastic part of corrected spring deformation."""

    if initial_stiffness_nm_per_rad <= 0.0 or not math.isfinite(initial_stiffness_nm_per_rad):
        raise ValueError("spring stiffness must be finite and positive")
    if not math.isfinite(total_spring_rotation_rad) or not math.isfinite(spring_moment_nm):
        raise ValueError("spring rotation and moment must be finite")
    elastic_rotation = abs(spring_moment_nm) / initial_stiffness_nm_per_rad
    return max(abs(total_spring_rotation_rad) - elastic_rotation, 0.0)


def rotation_capacity_for_level(
    limits: FemaHingeLimits, performance_level: Literal["IO", "LS", "CP"]
) -> float:
    """Return the FEMA primary-action plastic-rotation capacity for one level."""

    capacity = {
        "IO": limits.io_rotation_rad,
        "LS": limits.ls_rotation_rad,
        "CP": limits.cp_rotation_rad,
    }[performance_level]
    if capacity is None or capacity <= 0.0:
        raise ValueError("deformation-controlled hinge has no positive rotation capacity")
    return capacity


def force_controlled_column_utilization(
    *, compression_force_n: float, pcl_n: float, moment_nm: float, nominal_moment_nm: float
) -> float:
    """Return REF-A Eq. (29), ``P/Pcl + M/Mn``."""

    demands = (compression_force_n, moment_nm)
    capacities = (pcl_n, nominal_moment_nm)
    if any(value < 0.0 or not math.isfinite(value) for value in demands):
        raise ValueError("force-controlled demands must be finite and nonnegative")
    if any(value <= 0.0 or not math.isfinite(value) for value in capacities):
        raise ValueError("force-controlled capacities must be finite and positive")
    return compression_force_n / pcl_n + moment_nm / nominal_moment_nm


def story_drift_ratio(
    *, upper_displacement_m: float, lower_displacement_m: float, height_m: float
) -> float:
    """Return absolute interstory displacement divided by story height."""

    if height_m <= 0.0 or not math.isfinite(height_m):
        raise ValueError("story height must be finite and positive")
    if not math.isfinite(upper_displacement_m) or not math.isfinite(lower_displacement_m):
        raise ValueError("story displacements must be finite")
    return abs(upper_displacement_m - lower_displacement_m) / height_m
