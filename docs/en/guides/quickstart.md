# Quickstart

## Installation

Basic install (without optional heavy dependencies):

```bash
pip install tucoopy
```

Optional extras:

- LP-based methods (least-core / nucleolus / balancedness / bargaining set):
  ```bash
  pip install "tucoopy[lp]"
  ```
- Alternative LP backend (PuLP):
  ```bash
  pip install "tucoopy[lp_alt]"
  ```
- NumPy-based speedups (kernel / prekernel and some utilities):
  ```bash
  pip install "tucoopy[fast]"
  ```
- Simple 2-3 player Matplotlib visualizations:
  ```bash
  pip install "tucoopy[viz]"
  ```

## Building a TU game

Coalitions are stored internally as bitmasks, but you can define games using "Pythonic" coalition keys:

```py
from tucoopy import Game

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
```

## Computing a solution

```py
from tucoopy.solutions import shapley_value

phi = shapley_value(g)
print(phi)
```

## Generating an animation spec (Python -> JS contract)

```py
from tucoopy.io.animation_spec import build_animation_spec

spec = build_animation_spec(
    g,
    series_id="shapley",
    allocations=[phi] * 60,
    dt=1 / 30,
    series_description="Shapley value (static).",
    include_analysis=True,
)
print(spec.to_json())
```

More runnable scripts live in `packages/tucoopy/examples/`.

