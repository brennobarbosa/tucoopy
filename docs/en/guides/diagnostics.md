# Debugging core/epsilon-core via `analysis.diagnostics`

The goal of `analysis.diagnostics` is to let the frontend explain "why" a point fails (or passes) without backend calls.

## Why is a point not in the core?

For a point $x$, the main diagnostic is:

- $\text{maxexcess} = \max_S (v(S) - x(S))$
- If $\text{maxexcess} > 0$, there exists a blocking coalition.

In JSON, this appears as:

- `analysis.diagnostics.solutions.<id>.core.max_excess`
- `analysis.diagnostics.solutions.<id>.core.tight_coalitions`
- `analysis.diagnostics.solutions.<id>.core.violations` (top-k with `vS`, `xS`, `excess`)

## Per-frame diagnostics (tooltip)

The examples also include a small payload in each frame:

- `series[].frames[].highlights.diagnostics.core.blocking_coalition_mask`
- `series[].frames[].highlights.diagnostics.core.blocking_players`

This is useful for a tooltip that follows the mouse: when hovering a point/segment, show the blocking coalition and coordinates.

