"""Provide source-traceable response-spectrum and modal-result utilities for Phase 4."""

from __future__ import annotations

import math
from collections.abc import Sequence


def spectral_acceleration_g(
    period_s: float, *, short_period_g: float, one_second_g: float, fa: float, fv: float
) -> float:
    """Evaluate REF-A Eqs. (21)-(22) and return spectral acceleration in units of g."""

    if period_s < 0.0:
        raise ValueError("period must be nonnegative")
    if min(short_period_g, one_second_g, fa, fv) <= 0.0:
        raise ValueError("spectrum parameters must be positive")
    transition_period_s = fv * one_second_g / (fa * short_period_g)
    if period_s < 0.2 * transition_period_s:
        return fa * short_period_g * (0.4 + 3.0 * period_s / transition_period_s)
    if period_s <= transition_period_s:
        return fa * short_period_g
    return fv * one_second_g / period_s


def elastic_base_shear_n(
    total_seismic_weight_n: float, acceleration_g: float, r_factor: float
) -> float:
    """Evaluate REF-A Eq. (23) using weight and a spectrum ordinate expressed in g units."""

    if min(total_seismic_weight_n, acceleration_g, r_factor) <= 0.0:
        raise ValueError("weight, acceleration, and R must be positive")
    return total_seismic_weight_n * acceleration_g / r_factor


def distribute_story_forces_n(
    base_shear_n: float,
    story_weights_n: Sequence[float],
    elevations_m: Sequence[float],
    *,
    exponent: float = 2.0,
) -> tuple[float, ...]:
    """Evaluate REF-A Eq. (24) while forcing the final force to close floating-point sum drift."""

    if base_shear_n <= 0.0 or exponent <= 0.0:
        raise ValueError("base shear and distribution exponent must be positive")
    if len(story_weights_n) != len(elevations_m) or not story_weights_n:
        raise ValueError("story weights and elevations must have equal nonzero length")
    if min(*story_weights_n, *elevations_m) <= 0.0:
        raise ValueError("story weights and elevations must be positive")
    terms = [
        weight * elevation**exponent
        for weight, elevation in zip(story_weights_n, elevations_m, strict=True)
    ]
    denominator = sum(terms)
    forces = [base_shear_n * term / denominator for term in terms]
    forces[-1] = base_shear_n - sum(forces[:-1])
    return tuple(forces)


def periods_from_eigenvalues(eigenvalues: Sequence[float]) -> tuple[float, ...]:
    """Convert positive angular-frequency-squared eigenvalues to stable periods."""

    if not eigenvalues or any(value <= 0.0 or not math.isfinite(value) for value in eigenvalues):
        raise ValueError("eigenvalues must be finite and positive")
    return tuple(2.0 * math.pi / math.sqrt(value) for value in eigenvalues)
