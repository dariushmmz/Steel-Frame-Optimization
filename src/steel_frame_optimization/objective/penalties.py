"""Implement the documented normalized additive constraint-penalty policy."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from steel_frame_optimization.config.schemas import StrictConfigModel

ViolationCategory = Literal[
    "strength",
    "force_controlled_action",
    "hinge_rotation",
    "drift",
    "capacity_hierarchy",
    "target_state_shape",
]


class PenaltyPolicy(StrictConfigModel):
    """Freeze coefficients and ranking semantics authorized in ASM-016."""

    schema_version: Literal["1.0"] = "1.0"
    policy_id: Literal["normalized-additive-feasibility-offset-v1"]
    feasible_transform: Literal["F/(1+F)"]
    infeasible_offset: float = Field(gt=0.0, allow_inf_nan=False)
    coefficients: dict[ViolationCategory, float]
    stage1_infeasible_objective_basis: Literal[
        "bounded_weight_contribution_only_uniformity_unavailable"
    ]
    failure_handling: Literal["typed_failure_without_finite_ranking_score"]
    client_approval_status: Literal["awaiting_approval_before_main_campaigns"]
    assumption_id: Literal["ASM-016"]

    @classmethod
    def from_json(cls, path: Path) -> PenaltyPolicy:
        """Load and validate the explicit Phase 8 policy."""

        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))

    @model_validator(mode="after")
    def require_complete_coefficients(self) -> PenaltyPolicy:
        """Require exactly the six structural violation categories and frozen coefficients."""

        expected: dict[ViolationCategory, float] = {
            "strength": 100.0,
            "force_controlled_action": 100.0,
            "hinge_rotation": 100.0,
            "drift": 50.0,
            "capacity_hierarchy": 50.0,
            "target_state_shape": 25.0,
        }
        if self.coefficients != expected or self.infeasible_offset != 1.0:
            raise ValueError("penalty coefficients differ from ASM-016")
        return self


class NormalizedViolation(StrictConfigModel):
    """Persist one positive normalized exceedance and its weighted contribution."""

    check_id: str = Field(min_length=1)
    stage: Literal["stage1", "stage2"]
    category: ViolationCategory
    utilization: float = Field(ge=0.0, allow_inf_nan=False)
    normalized_exceedance: float = Field(ge=0.0, allow_inf_nan=False)
    coefficient: float = Field(gt=0.0, allow_inf_nan=False)
    weighted_penalty: float = Field(ge=0.0, allow_inf_nan=False)


class PenaltyScore(StrictConfigModel):
    """Persist a finite feasible or structurally-infeasible ranking score."""

    feasible: bool
    objective_basis: Literal["full_objective", "weight_contribution_only"]
    objective_base: float = Field(ge=0.0, allow_inf_nan=False)
    bounded_objective_base: float = Field(ge=0.0, lt=1.0, allow_inf_nan=False)
    feasibility_offset: float = Field(ge=0.0, allow_inf_nan=False)
    violation_penalty: float = Field(ge=0.0, allow_inf_nan=False)
    ranking_score: float = Field(ge=0.0, allow_inf_nan=False)


def normalized_violation(
    *,
    check_id: str,
    stage: Literal["stage1", "stage2"],
    category: ViolationCategory,
    utilization: float,
    failed: bool,
    policy: PenaltyPolicy,
) -> NormalizedViolation | None:
    """Return ``max(utilization - 1, 0)`` or no violation for a passing ratio."""

    if utilization < 0.0 or not math.isfinite(utilization):
        raise ValueError("constraint utilization must be finite and nonnegative")
    exceedance = max(utilization - 1.0, 0.0)
    if not failed and exceedance == 0.0:
        return None
    coefficient = policy.coefficients[category]
    return NormalizedViolation(
        check_id=check_id,
        stage=stage,
        category=category,
        utilization=utilization,
        normalized_exceedance=exceedance,
        coefficient=coefficient,
        weighted_penalty=coefficient * exceedance,
    )


def calculate_penalty_score(
    *,
    objective_base: float,
    objective_basis: Literal["full_objective", "weight_contribution_only"],
    feasible: bool,
    violations: tuple[NormalizedViolation, ...],
    policy: PenaltyPolicy,
) -> PenaltyScore:
    """Return ASM-016's bounded score with guaranteed feasibility separation."""

    if objective_base < 0.0 or not math.isfinite(objective_base):
        raise ValueError("penalty objective base must be finite and nonnegative")
    if feasible and violations:
        raise ValueError("a feasible score cannot contain positive violations")
    bounded = objective_base / (1.0 + objective_base)
    offset = 0.0 if feasible else policy.infeasible_offset
    violation_penalty = sum(item.weighted_penalty for item in violations)
    return PenaltyScore(
        feasible=feasible,
        objective_basis=objective_basis,
        objective_base=objective_base,
        bounded_objective_base=bounded,
        feasibility_offset=offset,
        violation_penalty=violation_penalty,
        ranking_score=offset + bounded + violation_penalty,
    )
