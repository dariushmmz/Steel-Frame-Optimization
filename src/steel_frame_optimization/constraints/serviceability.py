"""Calculate Stage 1 beam-deflection and elastic-drift utilization ratios."""

from __future__ import annotations

import math


def chord_relative_midspan_deflection_m(
    *, midpoint_vertical_m: float, i_end_vertical_m: float, j_end_vertical_m: float
) -> float:
    """Return absolute midspan deflection relative to the straight displaced end chord."""

    values = (midpoint_vertical_m, i_end_vertical_m, j_end_vertical_m)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("beam displacement inputs must be finite")
    chord_midpoint = (i_end_vertical_m + j_end_vertical_m) / 2.0
    return abs(midpoint_vertical_m - chord_midpoint)


def deflection_utilization(*, deflection_m: float, span_m: float, denominator: float) -> float:
    """Return beam deflection divided by the configured ``L/denominator`` limit."""

    if deflection_m < 0.0 or not math.isfinite(deflection_m):
        raise ValueError("deflection must be finite and nonnegative")
    if span_m <= 0.0 or denominator <= 0.0:
        raise ValueError("span and deflection-limit denominator must be positive")
    return deflection_m / (span_m / denominator)


def drift_utilization(*, drift_ratio: float, allowable_ratio: float) -> float:
    """Return absolute story or roof drift ratio divided by its positive limit."""

    if not math.isfinite(drift_ratio):
        raise ValueError("drift ratio must be finite")
    if allowable_ratio <= 0.0 or not math.isfinite(allowable_ratio):
        raise ValueError("allowable drift ratio must be finite and positive")
    return abs(drift_ratio) / allowable_ratio
