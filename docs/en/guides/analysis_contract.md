# `analysis` contract (Python -> JSON)

This page documents the meaning of the fields in `AnimationSpec.analysis` and the limits/guarantees of each one.

## Principles

- `analysis` is **optional** and intended for caching/explainability on the frontend.
- For **n<=4**, Python can export geometries (vertices) for drawing.
- For **n>4**, we prefer exporting **tables/lists** (a bundle) rather than geometric objects.

## `analysis.meta`

Provenance and parameters used to generate `analysis`:

- `analysis.meta.computed_by`
- `analysis.meta.build_analysis` (flags + `max_players`, `tol`, `diagnostics_top_k`)
- `analysis.meta.computed` (which sections were included)
- `analysis.meta.skipped` (reasons for skipping sections)
- `analysis.meta.limits` (applied limits/truncations, e.g. `diagnostics_max_list`)
- `analysis.meta.contract_version` (`analysis` contract version)

## `analysis.solutions`

Named points (allocations):

- E.g. `shapley`, `normalized_banzhaf`
- Each entry contains `{ "allocation": [x1, ..., xn] }`

Limits:

- May be **exponential** in `n`. By default, `build_analysis` only computes this for `n<=max_players`.
- `analysis.solutions.*.meta` records `computed_by` and `method`.

## `analysis.sets`

Representations for visualization (small `n`):

- `imputation.vertices`: vertices of the imputation set
- `core.vertices`: vertices of the core
- `reasonable.vertices`: vertices of the reasonable set (imputation + upper bounds `M`)
- `core_cover.vertices`: vertices of the core cover (`m <= x <= M`)
- `weber.points` (optional)

Limits:

- Generated only for `n<=max_players`.
- Each entry may include `meta` with provenance.
- Large lists (`vertices` / `points`) may be truncated by `max_points` and flagged in `meta.truncated`.

## `analysis.blocking_regions`

Cache of blocking regions (currently for `n=3`):

- `coordinate_system`
- `regions[]`: polygons with an associated `coalition_mask`

## `analysis.diagnostics`

Diagnostics for the UI (tooltips/tables):

- `analysis.diagnostics.input`: game checks (e.g. `vN`, `sum_singletons`, `essential`)
- `analysis.diagnostics.solutions.<id>.core`: why a point is/is not in the core

Truncations:

- Potentially large lists in `analysis.diagnostics.input` may be truncated (e.g. `missing_coalition_masks`).
  The field `missing_coalition_masks_truncated` indicates whether truncation happened and `missing_coalition_mask_count` reports the total.

Consistency checks (when applicable, small `n`):

- `simple_game`: whether all values are in `{0,1}`
- `monotone_simple_game`: whether the simple game is monotone (`v(S) <= v(S U {i})`)
- `monotone_counterexample`: a counterexample when `monotone_simple_game=false`

## LP-based explanations (optional)

When `include_lp_explanations=true` and `n<=lp_explanations_max_players`, `build_analysis` may include:

- `analysis.diagnostics.lp.balancedness_check` (Bondareva-Shapley: certificate for an empty core)
- `analysis.diagnostics.lp.least_core` (least-core $\epsilon$, tight coalitions, and solver diagnostics)

## `analysis.bundle` (n>4)

For `n>max_players`, `build_analysis` includes a lightweight bundle with summaries:

- `analysis.bundle.game_summary` (e.g. `n_players`, `vN`, `essential`, `provided_coalitions`)
- `analysis.bundle.notes` (how to interpret limits)
- `analysis.bundle.meta` records provenance

When `n<=bundle_max_players` and the game is **complete** (has all `v(S)`), the bundle may also include tables:

- `analysis.bundle.tables.players` (per-player scalars)
- `analysis.bundle.tables.power_indices` (only for complete simple games)
- `analysis.bundle.tables.tau_vectors` (utopia payoff and minimal rights; requires a complete game)
- `analysis.bundle.tables.approx_solutions.shapley` (approximate Shapley via sampling; includes `stderr`)
