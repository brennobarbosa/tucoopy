"""
# Internal Matplotlib visualization utilities.

This module contains small helpers shared by `tucoop.viz.mpl2` and
`tucoop.viz.mpl3`, including:

- optional Matplotlib import (`require_pyplot`),
- input normalization for point lists,
- helper geometry routines used by ternary plots.

It is internal to the visualization subpackage but is part of the public
installation extra ``tucoop[viz]``.
"""

from __future__ import annotations

from math import atan2
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Iterable

from ..backends.optional_deps import require_module
from ..base.exceptions import InvalidParameterError


def require_pyplot():
    matplotlib = require_module("matplotlib", extra="viz", context="tucoop.viz")
    pyplot = require_module("matplotlib.pyplot", extra="viz", context="tucoop.viz")
    return matplotlib, pyplot


def as_points(points: Sequence[Sequence[float]] | None, n: int) -> list[list[float]]:
    if points is None:
        return []
    out: list[list[float]] = []
    for p in points:
        if len(p) != n:
            raise InvalidParameterError("points must contain vectors of length n_players")
        out.append([float(x) for x in p])
    return out


def add_ternary_guides(
    ax,
    *,
    bary_to_xy: Callable[[float, float, float], tuple[float, float]],
    lower_bounds: Sequence[float] | None = None,
    r: float | None = None,
    steps: int = 10,
    show_grid: bool = True,
    show_ticks: bool = True,
    tick_fontsize: int = 9,
    grid_alpha: float = 0.12,
    tick_alpha: float = 0.85,
    tick_offset: float = 0.035,
    tick_fmt: Callable[[float], str] | None = None,
):
    A = bary_to_xy(1.0, 0.0, 0.0)
    B = bary_to_xy(0.0, 1.0, 0.0)
    C = bary_to_xy(0.0, 0.0, 1.0)

    use_alloc_units = False
    l1 = l2 = 0.0
    rr = 1.0

    if lower_bounds is not None and r is not None:
        if len(lower_bounds) == 3:
            rr = float(r)
            if rr > 0.0:
                use_alloc_units = True
                l1 = float(lower_bounds[1])
                l2 = float(lower_bounds[2])

    def fmt(v: float, *, is_alloc: bool) -> str:
        if tick_fmt is not None:
            return tick_fmt(v)
        if is_alloc:
            return f"{v:g}"
        return f"{v:.1f}"

    def line(p0: tuple[float, float], p1: tuple[float, float], *, alpha: float):
        x0, y0 = p0
        x1, y1 = p1
        ax.plot([x0, x1], [y0, y1], color="0.2", linewidth=1.0, alpha=alpha, zorder=0)

    if show_grid and steps >= 2:
        for k in range(1, steps):
            t = k / steps
            line(bary_to_xy(t, 0.0, 1.0 - t), bary_to_xy(t, 1.0 - t, 0.0), alpha=grid_alpha)
            line(bary_to_xy(0.0, t, 1.0 - t), bary_to_xy(1.0 - t, t, 0.0), alpha=grid_alpha)
            line(bary_to_xy(0.0, 1.0 - t, t), bary_to_xy(1.0 - t, 0.0, t), alpha=grid_alpha)

    if show_ticks and steps >= 2:
        for k in range(1, steps):
            t = k / steps
            x, y = bary_to_xy(1.0 - t, t, 0.0)
            val = (l1 + t * rr) if use_alloc_units else t
            ax.text(x, y - tick_offset, fmt(val, is_alloc=use_alloc_units), ha="center", va="top", fontsize=tick_fontsize, alpha=tick_alpha, zorder=5)
            x, y = bary_to_xy(1.0 - t, 0.0, t)
            val = (l2 + t * rr) if use_alloc_units else t
            ax.text(x - tick_offset, y, fmt(val, is_alloc=use_alloc_units), ha="right", va="center", fontsize=tick_fontsize, alpha=tick_alpha, zorder=5)
            x, y = bary_to_xy(0.0, 1.0 - t, t)
            val = (l2 + t * rr) if use_alloc_units else t
            ax.text(x + tick_offset, y, fmt(val, is_alloc=use_alloc_units), ha="left", va="center", fontsize=tick_fontsize, alpha=tick_alpha, zorder=5)

    xs = [A[0], B[0], C[0]]
    ys = [A[1], B[1], C[1]]
    pad = 0.08
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)


def polygon_order(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return sorted(points, key=lambda p: atan2(p[1] - cy, p[0] - cx))
