"""Evaluate explicit strong-column/weak-beam capacity hierarchy at frame joints."""

from __future__ import annotations

import math

from pydantic import Field

from steel_frame_optimization.config.schemas import StrictConfigModel


class CapacityHierarchyResult(StrictConfigModel):
    """Persist one REF-A Eq. (31) joint capacity ratio and its strict pass/fail result."""

    joint_id: str = Field(min_length=1)
    column_member_ids: tuple[str, ...] = Field(min_length=1)
    beam_member_ids: tuple[str, ...] = Field(min_length=1)
    column_capacity_sum_nm: float = Field(gt=0.0, allow_inf_nan=False)
    beam_capacity_sum_nm: float = Field(gt=0.0, allow_inf_nan=False)
    ratio: float = Field(gt=0.0, allow_inf_nan=False)
    required_ratio: float = Field(gt=0.0, allow_inf_nan=False)
    passed: bool
    source: str = Field(min_length=1)


def strong_column_weak_beam_check(
    *,
    joint_id: str,
    column_capacities_nm: dict[str, float],
    beam_capacities_nm: dict[str, float],
    required_ratio: float = 1.0,
) -> CapacityHierarchyResult:
    """Return REF-A Eq. (31), requiring the column sum to be strictly greater."""

    if not joint_id:
        raise ValueError("joint ID must not be empty")
    if not column_capacities_nm or not beam_capacities_nm:
        raise ValueError("hierarchy checks require connected columns and beams")
    all_capacities = (*column_capacities_nm.values(), *beam_capacities_nm.values())
    if any(value <= 0.0 or not math.isfinite(value) for value in all_capacities):
        raise ValueError("hierarchy capacities must be finite and positive")
    if required_ratio <= 0.0 or not math.isfinite(required_ratio):
        raise ValueError("required hierarchy ratio must be finite and positive")
    column_sum = sum(column_capacities_nm.values())
    beam_sum = sum(beam_capacities_nm.values())
    ratio = column_sum / beam_sum
    return CapacityHierarchyResult(
        joint_id=joint_id,
        column_member_ids=tuple(sorted(column_capacities_nm)),
        beam_member_ids=tuple(sorted(beam_capacities_nm)),
        column_capacity_sum_nm=column_sum,
        beam_capacity_sum_nm=beam_sum,
        ratio=ratio,
        required_ratio=required_ratio,
        passed=ratio > required_ratio,
        source="REF-A Eq. (31)",
    )
