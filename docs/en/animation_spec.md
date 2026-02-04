# Animation spec (Python -> JS contract)

`tucoopy` can emit a JSON "animation spec" that the JS renderer can consume to draw allocations over time.

Schema file (in this monorepo):
- `schema/tucoopy-animation.schema.json`

## Dataclasses

The Python-side data model lives in `tucoopy.io.animation_spec`:

- `AnimationSpec`
- `GameSpec` / `CharacteristicEntry`
- `SeriesSpec` / `FrameSpec`

Helper functions:

- `game_to_spec(game)` converts a `Game` to the JSON-friendly `GameSpec`.
- `series_from_allocations(...)` builds a `SeriesSpec` from a sequence of allocations.

## Minimal example

```py
from tucoopy import Game, shapley_value
from tucoopy.geometry import Core
from tucoopy.io import AnimationSpec, game_to_spec, series_from_allocations

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
spec = AnimationSpec(
    schema_version="0.1.0",
    game=game_to_spec(g),
    analysis={"sets": {"core": {"vertices": Core(g).vertices()}}},
    series=[series_from_allocations(series_id="shapley", allocations=[phi] * 60, dt=1 / 30)],
)
print(spec.to_json())
```

Notes:
- `analysis` is intentionally flexible, but you should keep it aligned with the JSON schema.
- For visualization, the JS package may only render up to 4 players (simplex up to 3-simplex).
