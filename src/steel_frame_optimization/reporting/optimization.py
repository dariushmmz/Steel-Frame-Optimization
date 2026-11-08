"""Render dependency-free deterministic optimization convergence evidence."""

from __future__ import annotations

import html
from collections.abc import Sequence
from pathlib import Path

from steel_frame_optimization.optimization.campaign import GenerationRecord


def convergence_svg(history: Sequence[GenerationRecord], *, title: str) -> str:
    """Return a compact SVG of best ranking score by evaluated population epoch."""

    if not history:
        raise ValueError("convergence plot requires at least one generation record")
    scores = tuple(record.best_ranking_score for record in history)
    if any(score is None for score in scores):
        raise ValueError("convergence plot requires finite ranked structural results")
    numeric_scores = tuple(float(score) for score in scores if score is not None)
    width = 720
    height = 420
    left = 80
    right = 30
    top = 55
    bottom = 65
    plot_width = width - left - right
    plot_height = height - top - bottom
    minimum = min(numeric_scores)
    maximum = max(numeric_scores)
    span = maximum - minimum
    if span == 0.0:
        span = max(abs(maximum), 1.0)
    x_denominator = max(len(numeric_scores) - 1, 1)
    points = " ".join(
        f"{left + plot_width * index / x_denominator:.3f},"
        f"{top + plot_height * (maximum - score) / span:.3f}"
        for index, score in enumerate(numeric_scores)
    )
    escaped_title = html.escape(title)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" '
        f'font-family="sans-serif" font-size="18">{escaped_title}</text>\n'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" '
        'stroke="black"/>\n'
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="black"/>\n'
        f'<polyline points="{points}" fill="none" stroke="#005a9c" stroke-width="3"/>\n'
        f'<text x="{width / 2:.1f}" y="{height - 18}" text-anchor="middle" '
        'font-family="sans-serif" font-size="14">Evaluated population epoch</text>\n'
        f'<text x="18" y="{height / 2:.1f}" text-anchor="middle" '
        'font-family="sans-serif" font-size="14" '
        f'transform="rotate(-90 18 {height / 2:.1f})">Best ranking score</text>\n'
        f'<text x="{left - 8}" y="{top + 5}" text-anchor="end" '
        f'font-family="monospace" font-size="12">{maximum:.6g}</text>\n'
        f'<text x="{left - 8}" y="{top + plot_height + 5}" text-anchor="end" '
        f'font-family="monospace" font-size="12">{minimum:.6g}</text>\n'
        "</svg>\n"
    )


def write_convergence_svg(path: Path, history: Sequence[GenerationRecord], *, title: str) -> None:
    """Write one LF-normalized convergence plot to an explicit artifact path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(convergence_svg(history, title=title), encoding="utf-8", newline="\n")
