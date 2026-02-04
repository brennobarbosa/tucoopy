# How-to: n>4 (bundle instead of geometry)

For $n>4$, the UI does not draw a full simplex; the recommended strategy is:

- Python exports an `analysis.bundle` with tables/lists/summaries.
- The frontend only presents the data (tables, lists, tooltips), without needing a backend to "draw".

## Example (Python)

```py
from tucoopy import Game
from tucoopy.io import build_analysis

g = Game.from_coalitions(
    n_players=6,
    values={(): 0.0, (0,1,2,3,4,5): 10.0},
    player_labels=[f"P{i+1}" for i in range(6)],
)

analysis = build_analysis(g, max_players=4, include_bundle=True)
print(analysis["bundle"]["game_summary"])
```

## What to expect in JSON

- `analysis.sets` / `analysis.solutions` may be omitted when `n>max_players`.
- `analysis.meta.skipped` explains why (e.g. `n=6 > max_players=4`).
- `analysis.bundle` contains a lightweight summary and notes to guide the UI.

When `n<=bundle_max_players` and the game is complete, the bundle may include extra tables:

- `analysis.bundle.tables.players` (per-player scalars)
- `analysis.bundle.tables.power_indices` (if it is a complete simple game)
- `analysis.bundle.tables.tau_vectors` (auxiliary vectors for the $\tau$ value)
- `analysis.bundle.tables.approx_solutions.shapley` (approximate Shapley via sampling, with `stderr`)

