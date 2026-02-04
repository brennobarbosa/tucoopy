"""
# Matplotlib helpers for 3-player games (ternary plots).

This module provides a static ternary (simplex) plot for $n=3$ TU games:

- draws the imputation simplex as the background triangle,
- overlays core and other set-valued objects as polygons/segments,
- overlays point solutions and custom points.

Warnings
--------
- This module depends on Matplotlib at runtime.
  Install it with `pip install "tucoopy[viz]"`.
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
from ..geometry.imputation_set import ImputationSet, imputation_lower_bounds
from ..geometry.kernel_set import KernelSet, PreKernelSet
from ..geometry.least_core_set import LeastCore
from ..geometry.reasonable_set import ReasonableSet
from ..geometry.weber_set import weber_marginal_vectors
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
from .utils import (
    add_ternary_guides,
    as_points,
    polygon_order,
    require_pyplot,
)


def plot_ternary(
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
    Plot a 3-player TU game on a ternary (simplex) diagram.

    The imputation set is drawn as the background triangle (after the standard
    affine normalization using individual lower bounds). Other sets are drawn
    either as polygons (>=3 vertices), segments (2 vertices), or points.

    Parameters
    ----------
    game
        TU game with n_players=3.
    points
        Optional allocations to overlay as points.
    points_by_label
        Optional mapping {label: allocation} to overlay labeled points.
    point_sets
        Optional mapping {label: allocations} to overlay point clouds.
    sets_vertices
        Optional mapping {label: vertices} for set polygons/segments in R^3.
    show_imputation
        If True, draw the imputation triangle (when feasible).
    show_core
        If True, compute and draw the core polygon (if non-empty).
    ax
        Optional Matplotlib Axes; if omitted, a new figure/axes is created.

    Returns
    -------
    (fig, ax)
        The Matplotlib Figure and Axes used for the plot.

    Raises
    ------
    InvalidParameterError
        If game.n_players != 3.

    Warnings
    --------
    This function requires Matplotlib at runtime (install with
    `pip install "tucoopy[viz]"`).
    """
    _, plt = require_pyplot()

    n = game.n_players
    if n != 3:
        raise InvalidParameterError("plot_ternary expects n_players=3")

    pts = as_points(points, n)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5.5))
    else:
        fig = ax.figure

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # Triangle vertices (equilateral)
    A = (0.0, 0.0)
    B = (1.0, 0.0)
    C = (0.5, 0.8660254037844386)  # sqrt(3)/2

    # Triangle outline
    ax.plot([A[0], B[0]], [A[1], B[1]], color="#666666", linewidth=2)
    ax.plot([B[0], C[0]], [B[1], C[1]], color="#666666", linewidth=2)
    ax.plot([C[0], A[0]], [C[1], A[1]], color="#666666", linewidth=2)

    def bary_to_xy(b0: float, b1: float, b2: float) -> tuple[float, float]:
        x = b0 * A[0] + b1 * B[0] + b2 * C[0]
        y = b0 * A[1] + b1 * B[1] + b2 * C[1]
        return (x, y)

    # --- normalization (imputation simplex) ---
    l = imputation_lower_bounds(game)
    vN = float(game.value(game.grand_coalition))
    r = vN - sum(l)

    if r <= 0:
        ax.text(
            0.02,
            0.98,
            "Imputation set is empty or degenerate.",
            transform=ax.transAxes,
            va="top",
        )
        return fig, ax

    # Now we can add guides with ticks in allocation units
    add_ternary_guides(
        ax,
        bary_to_xy=bary_to_xy,
        lower_bounds=l,
        r=r,
        steps=10,
        show_grid=True,
        show_ticks=True,
    )

    def allocation_to_xy(x: Sequence[float]) -> tuple[float, float] | None:
        b = [(float(x[i]) - l[i]) / r for i in range(3)]
        return bary_to_xy(b[0], b[1], b[2])

    if show_imputation:
        ax.fill(
            [A[0], B[0], C[0]],
            [A[1], B[1], C[1]],
            color="#999999",
            alpha=0.08,
            label="imputation",
        )

    default_sets: dict[str, list[list[float]]] = {}
    if show_core:
        default_sets["core"] = Core(game).vertices()

    all_sets: dict[str, list[list[float]]] = {
        k: [list(map(float, v)) for v in vs] for k, vs in default_sets.items()
    }
    if sets_vertices:
        for k, vs in sets_vertices.items():
            all_sets[str(k)] = [list(map(float, v)) for v in vs]

    colors = {
        "core": ("#00aaff", 0.10),
        "epsilon_core": ("#ffaa00", 0.10),
        "core_cover": ("#66ccff", 0.08),
        "reasonable": ("#cc66ff", 0.08),
    }

    for key, verts in all_sets.items():
        poly_xy: list[tuple[float, float]] = []
        for v in verts:
            p = allocation_to_xy(v)
            if p is not None:
                poly_xy.append(p)
        if not poly_xy:
            continue

        if len(poly_xy) >= 3:
            poly_xy = polygon_order(poly_xy)
            xs = [p[0] for p in poly_xy] + [poly_xy[0][0]]
            ys = [p[1] for p in poly_xy] + [poly_xy[0][1]]
            stroke, alpha = colors.get(key, ("#00aaff", 0.10))
            ax.fill(xs, ys, color=stroke, alpha=alpha)
            ax.plot(xs, ys, color=stroke, linewidth=2.2, label=key)
        elif len(poly_xy) == 2:
            stroke, _ = colors.get(key, ("#00aaff", 0.10))
            ax.plot(
                [poly_xy[0][0], poly_xy[1][0]],
                [poly_xy[0][1], poly_xy[1][1]],
                color=stroke,
                linewidth=2.5,
                label=key,
            )
        else:
            stroke, _ = colors.get(key, ("#00aaff", 0.10))
            ax.scatter([poly_xy[0][0]], [poly_xy[0][1]], color=stroke, s=40, label=key)

    if pts:
        pts_xy: list[tuple[float, float]] = []
        for alloc in pts:
            p = allocation_to_xy(alloc)
            if p is not None:
                pts_xy.append(p)
        if pts_xy:
            ax.scatter(
                [p[0] for p in pts_xy],
                [p[1] for p in pts_xy],
                color="#00ffaa",
                s=30,
                label="points",
            )

    if point_sets:
        for label, ps in point_sets.items():
            pts3 = as_points(ps, n)
            if not pts3:
                continue
            set_pts_xy: list[tuple[float, float]] = []
            for alloc in pts3:
                p = allocation_to_xy(alloc)
                if p is not None:
                    set_pts_xy.append(p)
            if set_pts_xy:
                ax.scatter(
                    [p[0] for p in set_pts_xy],
                    [p[1] for p in set_pts_xy],
                    s=14,
                    alpha=0.30,
                    label=str(label),
                )

    if points_by_label:
        for label, alloc_seq in points_by_label.items():
            if len(alloc_seq) != 3:
                continue
            p = allocation_to_xy([float(xi) for xi in alloc_seq])
            if p is None:
                continue
            ax.scatter([p[0]], [p[1]], s=45, label=str(label))

    # Corner labels (player names)
    game_labels = getattr(game, "player_labels", None)
    labels = list(game_labels) if game_labels else ["P1", "P2", "P3"]
    while len(labels) < 3:
        labels.append(f"P{len(labels)+1}")

    ax.text(A[0] - 0.02, A[1] - 0.04, labels[0])
    ax.text(B[0] + 0.01, B[1] - 0.04, labels[1])
    ax.text(C[0], C[1] + 0.03, labels[2], ha="center")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=4,
        frameon=False,
    )
    fig.subplots_adjust(bottom=0.20)
    return fig, ax


def plot_spec_ternary(
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
    Plot an AnimationSpec-like object for $n=3$ as a static ternary (simplex) plot.

    Reads sets/solutions from ``spec.analysis`` when available and enabled, and can
    compute missing objects for small $n$. The imputation simplex is rendered as the
    background triangle.

    Parameters
    ----------
    spec
        A spec object (dict-like or attribute-like) with fields:
        ``game`` and optionally ``analysis`` and ``series``.
    sets
        Set labels to plot. ``"imputation"`` controls whether the background
        simplex is drawn.
    solutions
        Solution labels to plot as points.
    series_id, frame_index, t
        Frame selection parameters for plotting a highlighted "frame" allocation.
    use_analysis
        If True, try using ``spec.analysis`` entries for sets/solutions.
    compute_missing
        If True, compute sets/solutions that are not present in analysis.
    epsilon
        Epsilon value used when computing ``epsilon_core`` (if needed).
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
        If the game in ``spec`` does not have ``n_players=3``.

    Warnings
    --------
    This function requires Matplotlib at runtime (install with
    `pip install "tucoopy[viz]"`).
    """
    game = game_from_animation_spec(spec)
    if game.n_players != 3:
        raise InvalidParameterError("plot_spec_ternary expects n_players=3")

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
        if key == "imputation":
            # In ternary plots the imputation set is always the full simplex after normalization,
            # so we render it as the background triangle only (no duplicated polygon outline).
            continue
        if key == "weber":
            pts = None
            if sets_obj is not None:
                entry = get_field(sets_obj, "weber", None)
                pts = get_field(entry, "points", None) if entry is not None else None
            if pts is not None:
                point_sets["weber"] = [list(map(float, p)) for p in pts]
            elif compute_missing:
                point_sets["weber"] = weber_marginal_vectors(game)
            continue

        # Generic "points" support from analysis (e.g., bargaining/kernel sampling).
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

    return plot_ternary(
        game,
        show_imputation=show_imputation,
        # We pass show_core=False here because "core" may already be included in sets_vertices.
        show_core=False,
        sets_vertices=sets_vertices,
        point_sets=point_sets,
        points_by_label=points_by_label,
        ax=ax,
    )


__all__ = ["plot_ternary", "plot_spec_ternary"]
