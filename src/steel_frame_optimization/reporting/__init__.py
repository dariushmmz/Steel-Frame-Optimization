"""Expose deterministic research-report artifacts."""

from steel_frame_optimization.reporting.optimization import (
    convergence_svg,
    write_convergence_svg,
)

__all__ = ["convergence_svg", "write_convergence_svg"]
