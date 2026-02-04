# Animation spec (Python -> JS contract)

`tucoopy` can emit a JSON "animation spec" that a renderer can consume to draw allocations over time.

Schema files (in this repo):

- `src/tucoopy/io/schemas/tucoop-animation.schema.json`
- `src/tucoopy/io/schemas/tucoop-game.schema.json`

## Dataclasses

The Python-side data model lives in `tucoopy.io.animation_spec`:

- `AnimationSpec`
- `GameSpec` / `CharacteristicEntry`
- `SeriesSpec` / `FrameSpec`

Helper functions:

- `game_to_spec(game)` converts a `Game` into a `GameSpec` (JSON-friendly).
- `series_from_allocations(...)` builds a `SeriesSpec` from a sequence of allocations.
- `build_animation_spec(...)` builds a "full" `AnimationSpec` (game + analysis + series), with optional highlights.

## Minimal example

```py
from tucoopy import Game, shapley_value
from tucoopy.io import build_animation_spec

g = Game.from_coalitions(
    n_players=3,
    values={
        (): 0.0,
        (0,): 1.0,
        (1,): 1.2,
        (2,): 0.8,
        (0, 1): 2.8,
        (0, 2): 2.2,
        (1, 2): 2.0,
        (0, 1, 2): 4.0,
    },
)

phi = shapley_value(g)
spec = build_animation_spec(
    g,
    schema_version="0.1.0",
    series_id="shapley",
    allocations=[phi] * 60,
    dt=1 / 30,
)
print(spec.to_json())
```

Notes:

- `analysis` is intentionally flexible, but it is worth keeping it aligned with the JSON schema.
- For visualization, the JS package can only render up to 4 players (simplex up to the 3-simplex).

## Per-frame highlights (`series[].frames[].highlights`)

Each frame can carry a `highlights` object with extra UI information (e.g. a tooltip that follows the mouse).

Current (optional) convention used by the examples:

- `frame.highlights.diagnostics.core`: contains a small payload with `max_excess` and `blocking_coalition_mask`.

## Provenance (`analysis.meta`)

`tucoopy.io.build_analysis(...)` fills an `analysis.meta` block to record:

- `analysis.meta.computed_by`: who generated it (e.g. `tucoopy`)
- `analysis.meta.build_analysis`: flags and parameters (e.g. `max_players`, `tol`, `diagnostics_top_k`)
- `analysis.meta.computed`: which sections were actually included (`solutions`, `sets`, `diagnostics`, `blocking_regions`)

## Diagnostics (`analysis.diagnostics`)

To support the UI (tooltips/tables) without a backend, Python can attach compact diagnostics in `analysis.diagnostics`.

Example: for each point in `analysis.solutions`, `tucoopy.io.build_analysis(...)` can include a summary of core membership:

- `analysis.diagnostics.solutions.<id>.core.in_core`
- `analysis.diagnostics.solutions.<id>.core.max_excess`
- `analysis.diagnostics.solutions.<id>.core.tight_coalitions` (coalitions attaining `max_excess`)
- `analysis.diagnostics.solutions.<id>.core.violations` (top-k blocking coalitions)

There is also `analysis.diagnostics.input` for game-level checks (e.g. `vN`, `sum_singletons`, `essential`, and whether the characteristic function is complete for small `n`).
