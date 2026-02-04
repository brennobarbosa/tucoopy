# Quickstart

## Install

Base install (no optional heavy deps):

```bash
pip install tucoop
```

Optional extras:

- LP-based methods (least-core / nucleolus / balancedness / bargaining set):
  ```bash
  pip install "tucoop[lp]"
  ```
- NumPy-based speedups (kernel / prekernel and some helpers):
  ```bash
  pip install "tucoop[fast]"
  ```

## Build a TU game

Coalitions are stored as bitmasks, but you can author games with Pythonic coalition keys:

```py
from tucoop import Game

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

## Compute a solution

```py
from tucoop.solutions import shapley_value

phi = shapley_value(g)
print(phi)
```

## Produce an animation spec (Python -> JS contract)

```py
from tucoop.io.animation_spec import build_animation_spec

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

More runnable scripts live in `packages/tucoop-py/examples/`.
