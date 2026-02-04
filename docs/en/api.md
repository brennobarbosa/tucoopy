# API map

The public API is intentionally constrained:

- `import tucoop` is a small, stable convenience surface.
- Most functionality lives in the canonical subpackages:
  - `tucoop.base`
  - `tucoop.games`
  - `tucoop.solutions`
  - `tucoop.geometry`
  - `tucoop.transforms`
  - `tucoop.properties`
  - `tucoop.power`
  - `tucoop.io`
  - `tucoop.backends`

## Stable top-level (`tucoop`)

```py
from tucoop import Game, glove_game, mask_from_players, nucleolus, shapley_value, weighted_voting_game
```

Notes:
- `nucleolus()` is an optional method (requires SciPy at runtime when called).

## Canonical subpackages

### `tucoop.base`

Primitives (bitmask coalitions + game interface):

- Coalition helpers: `all_coalitions`, `subcoalitions`, `players`, `size`, `grand_coalition`, `mask_from_players`
- Games: `Game`, `TabularGame`, `ValueFunctionGame`

### `tucoop.properties`

Game properties / recognizers:

- `is_convex`, `is_concave`
- `is_essential`, `is_monotone`, `is_normalized`, `is_superadditive`
- `is_simple_game`, `validate_simple_game`, `is_weighted_voting_game`
- `balancedness_check` (LP-backed)

### `tucoop.io`

JSON + animation spec helpers:

- Dataclasses: `AnimationSpec`, `GameSpec`, `SeriesSpec`, `FrameSpec`
- Helpers: `game_to_spec`, `series_from_allocations`, `build_animation_spec`
- JSON (games): `game_to_dict`, `game_from_dict`

### `tucoop.backends`

Optional dependency adapters:

- LP adapter: `tucoop.backends.lp.linprog_solve`
- NumPy helper: `tucoop.backends.numpy_fast.require_numpy`

### `tucoop.games`

Classic games / generators:

- `glove_game`
- `weighted_voting_game`
- `airport_game`
- `bankruptcy_game`
- `savings_game`
- `unanimity_game`
- `apex_game`

### `tucoop.solutions`

Solution concepts:

- Values: `shapley_value`, `banzhaf_value`, `normalized_banzhaf_value`
- Voting indices: `shapley_shubik_index`, `banzhaf_index`
- DP voting indices (integer weights): `shapley_shubik_index_weighted_voting`, `banzhaf_index_weighted_voting`
- Nucleolus family (SciPy): `least_core`, `nucleolus`, `prenucleolus` (+ result dataclasses)
- Kernel family (NumPy): `kernel`, `prekernel` (+ result dataclasses)
- Tau value helpers: `tau_value`, `utopia_payoff`, `minimal_rights`

### `tucoop.geometry`

Geometry for visualization:

- Core: `Core(game).vertices()` (small `n`)
- Excess / checks: `excesses`, `max_excess`, `tight_coalitions`, `is_in_core`, `is_in_epsilon_core`, `is_imputation`, `is_efficient`
- Imputation set: `imputation_lower_bounds`, `is_in_imputation_set`, `project_to_imputation`, `ImputationSet(game).vertices()`
- Epsilon-core: `EpsilonCore(game, eps).poly`, `EpsilonCore(game, eps).vertices()`, `least_core_polytope`
- Weber set: `marginal_vector`, `weber_marginal_vectors`, `weber_sample`
- Bargaining set (SciPy): `bargaining_set_check`, `bargaining_set_sample`, `is_in_bargaining_set`
- Balancedness (SciPy): `balancedness_check` (+ result dataclass)

### `tucoop.transforms`

Transforms / representations:

- `to_dense_values`
- `mobius_transform`, `inverse_mobius_transform`
- `harsanyi_dividends`
