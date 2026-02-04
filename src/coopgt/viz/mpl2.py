"""
# Matplotlib helpers for 2-player games (2D segment plots).

This module provides a small, static visualization for $n=2$ TU games in the
allocation plane $(x_1, x_2)$:

- draws the imputation set and the core as line segments,
- overlays point solutions (e.g. Shapley, Banzhaf, tau) and custom points,
- can read either a `tucoop.base.game.Game` directly or a spec-like dict
  (compatible with the JSON contract used by the JS demo).

Warnings
--------
- This module depends on Matplotlib at runtime.
  Install it with `pip install "tucoop[viz]"`.
- If Matplotlib is not installed, calling any plotting function will raise a
  `MissingOptionalDependencyError`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

from ..base.types import GameProtocol
from ..base.exceptions import InvalidParameterError
from ..geometry.core_set import Core
from ..geometry.core_cover_set import CoreCover
from ..geometry.epsilon_core_set import EpsilonCore
from ..geometry.imputation_set import ImputationSet
from ..geometry.kernel_set import KernelSet, PreKernelSet
from ..geometry.least_core_set import LeastCore
from ..geometry.reasonable_set import ReasonableSet
from ..solutions.banzhaf import banzhaf_value, normalized_banzhaf_value
from ..solutions.shapley import shapley_value
from ..solutions.tau import tau_value
from ..io.game_spec import (
    game_from_animation_spec,
    analysis_sets,
    analysis_solutions,
    get_field,
    select_frame_allocation,
)

from .utils import as_points, require_pyplot


def plot_segment(
    game: GameProtocol,
    *,
    points: Sequence[Sequence[float]] | None = None,
    points_by_label: Mapping[str, Sequence[float]] | None = None,
    point_sets: Mapping[str, Sequence[Sequence[float]]] | None = None,
    sets_vertices: Mapping[str, Sequence[Sequence[float]]] | None = None,
    show_imputation: bool = True,
    show_core: bool = True,
    ax=None,
):
    """
    Plot the 2-player ($n=2$) imputation set and core as line segments in $(x_1,x_2)$.

    Parameters
    ----------
    game
        TU game with n_players=2.
    points
        Optional list of allocations to plot on top.
    points_by_label
        Optional mapping {label: allocation}.
    point_sets
        Optional mapping {label: allocations} to plot as point clouds.
    sets_vertices
        Optional mapping {label: vertices} for sets drawn as segments.
    show_imputation
        If True, draw the imputation segment.
    show_core
        If True, draw the core segment (if non-empty).
    ax
        Optional Matplotlib Axes; if omitted, a new figure/axes is created.

    Warnings
    --------
    This function requires Matplotlib at runtime (install with
    `pip install "tucoop[viz]"`).
    """
    _, plt = require_pyplot()

    n = game.n_players
    if n != 2:
        raise InvalidParameterError("plot_segment expects n_players=2")

    pts = as_points(points, n)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    default_sets: dict[str, list[list[float]]] = {}
    if show_imputation:
        default_sets["imputation"] = ImputationSet(game).vertices()
    if show_core:
        default_sets["core"] = Core(game).vertices()

    all_sets: dict[str, list[list[float]]] = {
        k: [list(map(float, v)) for v in vs] for k, vs in default_sets.items()
    }
    if sets_vertices:
        for k, vs in sets_vertices.items():
            all_sets[str(k)] = [list(map(float, v)) for v in vs]

    colors = {
        "imputation": ("#666666", 3),
        "core": ("#00aaff", 4),
        "epsilon_core": ("#ffaa00", 3),
        "core_cover": ("#66ccff", 2),
        "reasonable": ("#cc66ff", 2),
    }

    for key, vs in all_sets.items():
        if len(vs) < 2:
            continue
        x0, x1 = vs[0], vs[1]
        if len(x0) != 2 or len(x1) != 2:
            continue
        color, lw = colors.get(key, ("#999999", 2))
        ax.plot([x0[0], x1[0]], [x0[1], x1[1]], color=color, linewidth=lw, label=key)

    if point_sets:
        for label, ps in point_sets.items():
            pts2 = as_points(ps, n)
            if not pts2:
                continue
            ax.scatter(
                [p[0] for p in pts2],
                [p[1] for p in pts2],
                s=12,
                alpha=0.35,
                label=str(label),
            )

    if pts:
        ax.scatter(
            [p[0] for p in pts],
            [p[1] for p in pts],
            color="#00ffaa",
            s=35,
            label="points",
        )

    if points_by_label:
        for label, p in points_by_label.items():
            if len(p) != 2:
                continue
            ax.scatter([float(p[0])], [float(p[1])], s=45, label=str(label))

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="best")
    return fig, ax


def plot_spec_segment(
    spec: Any,
    *,
    sets: Sequence[str] = ("imputation", "core"),
    solutions: Sequence[str] = ("shapley",),
    series_id: str | None = None,
    frame_index: int | None = 0,
    t: float | None = None,
    use_analysis: bool = True,
    compute_missing: bool = True,
    epsilon: float | None = None,
    n_samples: int = 200,
    seed: int | None = None,
    ax=None,
):
    """
    Plot an AnimationSpec-like object for $n=2$ as a static segment plot.

    This is a convenience wrapper that:
    
    1. builds a ``Game`` from ``spec.game``,
    2. reads precomputed sets/solutions from ``spec.analysis`` (if enabled),
    and 
    3. optionally computes missing objects for small $n$.

    Parameters
    ----------
    spec
        A spec object (dict-like or attribute-like) with fields:
        ``game`` and optionally ``analysis`` and ``series``.
    sets
        Set labels to plot (e.g. ``("imputation", "core", "epsilon_core")``).
        Some sets may be drawn from vertices; others may be drawn as point clouds.
    solutions
        Solution labels to plot as points (e.g. ``("shapley", "tau")``).
    series_id, frame_index, t
        Frame selection parameters for plotting a highlighted "frame" allocation
        from ``spec.series`` (if present).
    use_analysis
        If True, try using ``spec.analysis.sets`` and ``spec.analysis.solutions``.
    compute_missing
        If True, compute sets/solutions that are not present in analysis.
    epsilon
        Epsilon value used when computing ``epsilon_core`` (if needed).
        If omitted, tries to read it from analysis when available.
    n_samples
        Sample size for sampling-based sets (kernel/prekernel/bargaining).
    seed
        Random seed for sampling-based sets.
    ax
        Optional Matplotlib Axes.

    Returns
    -------
    (fig, ax)
        The Matplotlib Figure and Axes used for the plot.

    Raises
    ------
    InvalidParameterError
        If the game in ``spec`` does not have ``n_players=2``.

    Warnings
    --------
    This function requires Matplotlib at runtime (install with
    `pip install "tucoop[viz]"`).
    """
    game = game_from_animation_spec(spec)
    if game.n_players != 2:
        raise InvalidParameterError("plot_spec_segment expects n_players=2")

    show_imputation = "imputation" in set(sets)

    frame_alloc = select_frame_allocation(
        spec, series_id=series_id, frame_index=frame_index, t=t
    )
    points_by_label: dict[str, list[float]] = {}
    if frame_alloc is not None:
        points_by_label["frame"] = frame_alloc

    sets_vertices: dict[str, list[list[float]]] = {}
    point_sets: dict[str, list[list[float]]] = {}
    sets_obj = analysis_sets(spec) if use_analysis else None
    for key in sets:
        # If the spec provides points (e.g., sampling-based sets), prefer plotting them as a cloud.
        if sets_obj is not None:
            entry = get_field(sets_obj, key, None)
            pts = get_field(entry, "points", None) if entry is not None else None
            if pts is not None:
                point_sets[str(key)] = [list(map(float, p)) for p in pts]
                continue

        verts = None
        if sets_obj is not None:
            entry = get_field(sets_obj, key, None)
            verts = get_field(entry, "vertices", None) if entry is not None else None
        if verts is not None:
            sets_vertices[str(key)] = [list(map(float, v)) for v in verts]
            continue
        if not compute_missing:
            continue
        if key == "imputation":
            sets_vertices[key] = ImputationSet(game).vertices()
        elif key == "core":
            sets_vertices[key] = Core(game).vertices()
        elif key == "epsilon_core":
            eps = epsilon
            if eps is None and sets_obj is not None:
                entry = get_field(sets_obj, "epsilon_core", None)
                eps = get_field(entry, "epsilon", None) if entry is not None else None
            if eps is not None:
                sets_vertices[key] = EpsilonCore(game, float(eps)).vertices()
        elif key == "core_cover":
            sets_vertices[key] = CoreCover(game).vertices()
        elif key == "reasonable":
            sets_vertices[key] = ReasonableSet(game).vertices()
        elif key == "least_core":
            sets_vertices[key] = LeastCore(game).vertices()
        elif key == "prekernel_set":
            point_sets[key] = PreKernelSet(game).sample_points(
                n_samples=int(n_samples), seed=seed
            )
        elif key == "kernel_set":
            point_sets[key] = KernelSet(game).sample_points(
                n_samples=int(n_samples), seed=seed
            )
        elif key == "bargaining":
            from ..geometry.bargaining_set import BargainingSet

            point_sets[key] = BargainingSet(game).sample_points(
                n_samples=int(n_samples), seed=seed
            )

    sols_obj = analysis_solutions(spec) if use_analysis else None
    for sid in solutions:
        alloc = None
        if sols_obj is not None:
            entry = get_field(sols_obj, sid, None)
            alloc = get_field(entry, "allocation", None) if entry is not None else None
        if alloc is not None:
            points_by_label[str(sid)] = [float(x) for x in alloc]
            continue
        if not compute_missing:
            continue
        if sid == "shapley":
            points_by_label[sid] = shapley_value(game)
        elif sid == "banzhaf":
            points_by_label[sid] = banzhaf_value(game)
        elif sid == "normalized_banzhaf":
            points_by_label[sid] = normalized_banzhaf_value(game)
        elif sid == "tau":
            points_by_label[sid] = tau_value(game)
        else:
            from ..solutions.solve import solve as solve_dispatch

            res = solve_dispatch(game, method=cast(Any, sid))
            points_by_label[sid] = [float(v) for v in res.x]

    return plot_segment(
        game,
        show_imputation=show_imputation,
        # We pass show_core=False here because "core" may already be included in sets_vertices.
        show_core=False,
        sets_vertices=sets_vertices,
        point_sets=point_sets,
        points_by_label=points_by_label,
        ax=ax,
    )


__all__ = ["plot_segment", "plot_spec_segment"]
