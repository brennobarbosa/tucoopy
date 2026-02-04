# Performance, limits, and computational costs

This page summarizes the main **asymptotic costs** in `tucoopy` and how to choose practical limits for larger games.

## Rule of thumb

- Almost everything that "scans coalitions" is at least $O(2^n)$.
- Almost everything that "scans permutations" is $O(n!)$.
- Almost everything that "enumerates polytope vertices" blows up with the number of constraints and the dimension.

As $n$ grows, the recommended strategy is:

- prefer **sampling** (points and/or permutations),
- generate `analysis.bundle` (tables and summaries) instead of trying to "draw the whole simplex",
- keep exact geometry to small $n$.

## Quick table (order of magnitude)

| Object / routine | Typical cost | Notes |
|---|---:|---|
| Coalition scans (e.g. excesses) | $O(2^n)$ | depends on having `v(S)` accessible / cached |
| Shapley (exact, tabular) | $O(n 2^n)$ | sum over subcoalitions / DP |
| Banzhaf (exact, tabular) | $O(n 2^n)$ | similar cost to Shapley |
| Nucleolus / least-core / balancedness | many LPs | each LP may have many constraints (coalitions) |
| Weber set (exact) | $O(n!)$ | only feasible for small $n$ |
| Polyhedra (vertices) | exponential | vertex enumeration does not scale to large $n$ |
| Hit-and-run (sampling) | many steps | requires a **bounded** set and an initial point (LP) |

## Practical recommendations by family

### Point solutions

- For small $n$ (~10-12): `shapley_value`, `normalized_banzhaf_value`, and similar can be used exactly, as long as the game is "complete" (tabular).
- For larger $n$:
  - prefer approximating Shapley by sampling permutations (when available),
  - avoid routines with many LPs (nucleolus/modiclus) without clear limits.

### Geometry (sets / polytopes)

- `PolyhedralSet.extreme_points(...)` is for visualization in low dimensions.
- For projections as $n$ grows, prefer `project(..., approx_n_points=...)` (sampling + point projection).

### Simple games / power indices

- If you can represent the simple game compactly (e.g. weighted voting), indices like Banzhaf/SSI tend to scale better than blindly scanning all subsets.
- For tabular simple games, there is still an $O(2^n)$ cost for many operations.

### Weber set

The Weber set is the convex hull of marginal vectors. The exact generator has size $n!$, so:

- for small $n$: use `WeberSet.points()` and `WeberSet.poly` (when `n in {2,3}`),
- for larger $n$: use `WeberSet.sample_points(...)` and treat the result as a point cloud (not an exact polytope).

## Backends and dependencies

- LP routines depend on a backend (recommended: SciPy/HiGHS). See `guides/lp_backends.md`.
- Some performance-sensitive routines use NumPy when available (extra `tucoopy[fast]`).

## Checklist: what to do when things get slow

1. Check whether the game is complete/tabular (when the routine assumes it).
2. Reduce `max_players` / `max_dim` / `max_points`.
3. Replace vertices with sampling (`sample_points_*`) and approximate projection.
4. If there is LP, confirm SciPy is installed and being used.

