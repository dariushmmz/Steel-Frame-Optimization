"""Define typed failures so future analysis errors cannot disappear or look feasible."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from steel_frame_optimization.models.manifests import StrictFrozenModel


class FailureCategory(StrEnum):
    """Classify failures without assigning an optimization severity ordering."""

    INPUT_INVALID = "input_invalid"
    PROVENANCE_MISMATCH = "provenance_mismatch"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    NOT_IMPLEMENTED = "not_implemented"
    NUMERICAL_NONCONVERGENCE = "numerical_nonconvergence"
    NUMERICAL_INCONSISTENCY = "numerical_inconsistency"
    STRUCTURAL_INFEASIBILITY = "structural_infeasibility"
    INTERNAL_ERROR = "internal_error"
    USER_CANCELLED = "user_cancelled"


class FailureRecord(StrictFrozenModel):
    """Persist one explicit failure with stable category, stage, message, and details."""

    schema_version: str = Field(default="1.0", pattern=r"^1\.0$")
    category: FailureCategory
    stage: str = Field(min_length=1)
    message: str = Field(min_length=1)
    recoverable: bool
    details: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
