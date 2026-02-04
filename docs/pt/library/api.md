# Mapa da API

A API pública é intencionalmente restrita:

- `import tucoop` é uma superfície pequena e estável de conveniência.
- A maior parte da funcionalidade vive nos subpacotes canônicos:
  - `tucoop.base`
  - `tucoop.games`
  - `tucoop.solutions`
  - `tucoop.geometry`
  - `tucoop.transforms`
  - `tucoop.properties`
  - `tucoop.power`
  - `tucoop.io`
  - `tucoop.backends`
  - `tucoop.viz`

## Top-level estável (`tucoop`)

```py
from tucoop import Game, glove_game, mask_from_players, nucleolus, shapley_value, weighted_voting_game
```

Notas:
- `nucleolus()` é um método opcional (exige SciPy em runtime quando chamado).
- Vértices do núcleo são pensados para `n` pequeno (uso em visualização).

## Subpacotes canônicos

### `tucoop.base`

Representação de jogos + helpers “amigáveis” para IO:

- Helpers de coalizÃ£o: `all_coalitions`, `subcoalitions`, `players`, `size`, `grand_coalition`, `mask_from_players`
- Jogos: `Game`, `TabularGame`, `ValueFunctionGame`

### `tucoop.properties`

Propriedades / reconhecedores:

- `is_convex`, `is_concave`
- `is_essential`, `is_monotone`, `is_normalized`, `is_superadditive`
- `is_simple_game`, `validate_simple_game`, `is_weighted_voting_game`
- `balancedness_check` (LP)

### `tucoop.io`

Helpers de JSON + animation spec:

- Dataclasses: `AnimationSpec`, `GameSpec`, `SeriesSpec`, `FrameSpec`
- Helpers: `game_to_spec`, `series_from_allocations`, `build_animation_spec`
- JSON (jogos): `game_to_dict`, `game_from_dict`

### `tucoop.backends`

Adapters para dependÃªncias opcionais:

- Adapter LP: `tucoop.backends.lp.linprog_solve`
- Helper NumPy: `tucoop.backends.numpy_fast.require_numpy`

### `tucoop.games`

Jogos clássicos / geradores:

- `glove_game`
- `weighted_voting_game`
- `airport_game`
- `bankruptcy_game`
- `savings_game`
- `unanimity_game`
- `apex_game`

### `tucoop.solutions`

Conceitos de solução:

- Valores: `shapley_value`, `banzhaf_value`, `normalized_banzhaf_value`
- Família do nucleolus (SciPy): `least_core`, `nucleolus`, `prenucleolus` (+ dataclasses de resultado)
- Família kernel (NumPy): `kernel`, `prekernel` (+ dataclasses de resultado)
- Helpers do valor τ: `tau_value`, `utopia_payoff`, `minimal_rights`

### `tucoop.geometry`

Geometria para visualização:

- Núcleo: `Core(game).vertices()` (n pequeno)
- Excesso / checks: `excesses`, `max_excess`, `tight_coalitions`, `is_in_core`, `is_in_epsilon_core`, `is_imputation`, `is_efficient`
- Conjunto de imputações: `imputation_lower_bounds`, `is_in_imputation_set`, `project_to_imputation`, `ImputationSet(game).vertices()`
- ε-núcleo: `EpsilonCore(game, eps).poly`, `EpsilonCore(game, eps).vertices()`, `least_core_polytope`
- Conjunto de Weber: `marginal_vector`, `weber_marginal_vectors`, `weber_sample`
- Conjunto de barganha (SciPy): `bargaining_set_check`, `bargaining_set_sample`, `is_in_bargaining_set`
- Balanceamento (SciPy): `balancedness_check` (+ dataclass de resultado)

### `tucoop.power`

- Índices de votação: `shapley_shubik_index`, `banzhaf_index`
- Índices via PD (pesos inteiros): `shapley_shubik_index_weighted_voting`, `banzhaf_index_weighted_voting`

### `tucoop.transforms`

Transformações / representações:

- `to_dense_values`
- `mobius_transform`, `inverse_mobius_transform`
- `harsanyi_dividends`
