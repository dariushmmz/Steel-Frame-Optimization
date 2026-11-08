"""Implement REF-A Eq. (20) without fitting the documented Wmax discrepancy."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise

from pydantic import Field

from steel_frame_optimization.config.schemas import StrictConfigModel
from steel_frame_optimization.sections.records import SectionRecord


class ObjectiveComponents(StrictConfigModel):
    """Persist every raw and weighted component of REF-A Eq. (20)."""

    structural_weight_n: float = Field(gt=0.0, allow_inf_nan=False)
    maximum_structural_weight_n: float = Field(gt=0.0, allow_inf_nan=False)
    normalized_weight: float = Field(gt=0.0, allow_inf_nan=False)
    uniformity_term: float = Field(ge=0.0, allow_inf_nan=False)
    weight_coefficient: float = Field(gt=0.0, allow_inf_nan=False)
    uniformity_coefficient: float = Field(gt=0.0, allow_inf_nan=False)
    weight_contribution: float = Field(gt=0.0, allow_inf_nan=False)
    uniformity_contribution: float = Field(ge=0.0, allow_inf_nan=False)
    raw_objective: float = Field(gt=0.0, allow_inf_nan=False)
    source: str = "REF-A Eq. (20), PDF p. 4 / journal p. 80"


def structural_weight_n(
    *,
    total_member_length_m_by_group: Mapping[str, float],
    sections_by_group: Mapping[str, SectionRecord],
) -> float:
    """Return candidate member weight from section line weight and centerline length."""

    if set(total_member_length_m_by_group) != set(sections_by_group):
        raise ValueError("member lengths and section assignments must have identical group IDs")
    total = 0.0
    for group_id, length_m in total_member_length_m_by_group.items():
        if length_m <= 0.0 or not math.isfinite(length_m):
            raise ValueError(f"group {group_id} length must be finite and positive")
        total += length_m * sections_by_group[group_id].weight_n_per_m
    return total


def drift_uniformity_term(
    *,
    floor_displacements_m: Sequence[float],
    floor_elevations_m: Sequence[float],
) -> float:
    """Return the exact CP drift-uniformity term printed in REF-A Eq. (20).

    The roof term is excluded from the sum, while the divisor remains the total
    number of stories ``ns`` as printed in both supplied references.
    """

    if len(floor_displacements_m) != len(floor_elevations_m):
        raise ValueError("floor displacement and elevation counts must match")
    story_count = len(floor_displacements_m)
    if story_count < 2:
        raise ValueError("uniformity requires at least two stories")
    if any(not math.isfinite(value) for value in floor_displacements_m):
        raise ValueError("floor displacements must be finite")
    if any(value <= 0.0 or not math.isfinite(value) for value in floor_elevations_m):
        raise ValueError("floor elevations must be finite and positive")
    if any(upper <= lower for lower, upper in pairwise(floor_elevations_m)):
        raise ValueError("floor elevations must be strictly increasing")
    roof_displacement = floor_displacements_m[-1]
    if roof_displacement <= 0.0:
        raise ValueError("CP roof displacement must be positive")
    roof_height = floor_elevations_m[-1]
    squared_terms = (
        (displacement / roof_displacement * roof_height / elevation - 1.0) ** 2
        for displacement, elevation in zip(
            floor_displacements_m[:-1], floor_elevations_m[:-1], strict=True
        )
    )
    return math.sqrt(sum(squared_terms) / story_count)


def calculate_objective(
    *,
    structural_weight: float,
    maximum_structural_weight: float,
    uniformity_term: float,
    weight_coefficient: float,
    uniformity_coefficient: float,
) -> ObjectiveComponents:
    """Combine weight and CP uniformity into the paper's weighted objective."""

    values = (structural_weight, maximum_structural_weight)
    if any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("structural weight and Wmax must be finite and positive")
    if uniformity_term < 0.0 or not math.isfinite(uniformity_term):
        raise ValueError("uniformity must be finite and nonnegative")
    coefficients = (weight_coefficient, uniformity_coefficient)
    if any(value <= 0.0 or not math.isfinite(value) for value in coefficients):
        raise ValueError("objective coefficients must be finite and positive")
    if not math.isclose(sum(coefficients), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("objective coefficients must sum to one")
    normalized_weight = structural_weight / maximum_structural_weight
    weight_contribution = weight_coefficient * normalized_weight
    uniformity_contribution = uniformity_coefficient * uniformity_term
    return ObjectiveComponents(
        structural_weight_n=structural_weight,
        maximum_structural_weight_n=maximum_structural_weight,
        normalized_weight=normalized_weight,
        uniformity_term=uniformity_term,
        weight_coefficient=weight_coefficient,
        uniformity_coefficient=uniformity_coefficient,
        weight_contribution=weight_contribution,
        uniformity_contribution=uniformity_contribution,
        raw_objective=weight_contribution + uniformity_contribution,
    )
