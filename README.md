# tucoopy-py

Python package for cooperative game theory (TU) algorithms and for generating
animation specs (JSON) consumed by the JS renderer.

Optional speedups:
- `pip install "tucoopy[fast]"` (uses NumPy for small linear solves in geometry helpers)
- `pip install "tucoopy[lp]"` (enables LP-based methods like least-core / nucleolus via SciPy)

Optional visualization:
- `pip install "tucoopy[viz]"` for Matplotlib visualization (2 or 3 players only)

## Optional extras (feature -> extra)

| Feature | Extra | Notes |
| --- | --- | --- |
| LP-backed methods | `lp` | SciPy backend (recommended) |
| LP-backed methods (fallback) | `lp_alt` | PuLP backend |
| Speedups | `fast` | NumPy helper routines |
| Static visualization | `viz` | Matplotlib (2 or 3 players only) |
| Dev tools | `dev` | pytest + mypy + ruff |
| Docs build | `docs` | mkdocs + mkdocstrings |

## Package layout

- `tucoopy.base`: game/coalition primitives (bitmask-based)
- `tucoopy.properties`: game properties / recognizers
- `tucoopy.games`: classic games (glove, weighted voting, airport, bankruptcy, savings, unanimity, apex, ...)
- `tucoopy.geometry`: geometry for visualization (core vertices, ...)
- `tucoopy.solutions`: solution concepts (Shapley, Banzhaf, ...)
- `tucoopy.power`: voting/simple-game power indices
- `tucoopy.transforms`: transforms/representations (Harsanyi dividends, ...)
- `tucoopy.viz`: optional visualization with Matplotlib (games with 2 or 3 players only)
- `tucoopy.io`: JSON + animation spec helpers
- `tucoopy.backends`: adapters for optional dependencies (LP, NumPy, ...)

Docs (in this repo): see `packages/tucoopy-py/docs/en/index.md` (EN) and `packages/tucoopy-py/docs/pt/index.md` (PT).
Examples (runnable scripts): see `packages/tucoopy-py/examples/README.md`.

## Install

```bash
pip install tucoopy
```

Optional extras:

- `pip install "tucoopy[lp]"` for LP-based methods (least-core / nucleolus / modiclus / balancedness) using SciPy (recommended)
- `pip install "tucoopy[lp_alt]"` for LP-based methods (least-core / nucleolus / modiclus / balancedness) using PuLP (fallback)
- `pip install "tucoopy[fast]"` for NumPy speedups (kernel / prekernel and helpers)
- `pip install "tucoopy[viz]"` for Matplotlib visualization (2 or 3 players only)

## Quick example (generate an animation spec)

```py
from tucoopy import Game
from tucoopy.solutions import shapley_value
from tucoopy.io.animation_spec import build_animation_spec

game = Game.from_coalitions(
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
    player_labels=["P1", "P2", "P3"],
)

phi = shapley_value(game)
spec = build_animation_spec(
    game,
    series_id="shapley",
    allocations=[phi] * 60,
    dt=1 / 30,
    series_description="Shapley value (static).",
    include_analysis=True,
)
print(spec.to_json())
```

## Nucleolus / least-core (LP)

LP-based methods are behind the optional `lp` extra:

```py
from tucoopy import Game
from tucoopy.solutions import least_core, nucleolus

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

lc = least_core(g)
nu = nucleolus(g)
print(lc.epsilon, lc.x)
print(nu.levels, nu.x)
```

## Core non-emptiness certificate (LP)

Bondareva–Shapley balancedness check (behind `lp`):

```py
from tucoopy import Game
from tucoopy.properties.balancedness import balancedness_check

g = Game.from_coalitions(
    n_players=3,
    values={
        (): 0.0,
        (0,): 0.0,
        (1,): 0.0,
        (2,): 0.0,
        (0, 1): 1.0,
        (0, 2): 1.0,
        (1, 2): 1.0,
        (0, 1, 2): 1.0,
    },
)

res = balancedness_check(g)
print(res.core_nonempty, res.objective, res.weights)
```

## Schema (Python <-> JS contract)

- Canonical schema: `schema/tucoopy-animation.schema.json`
- Bundled schema (package data): `packages/tucoopy-py/src/tucoopy/io/schemas/tucoopy-animation.schema.json`
