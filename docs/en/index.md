# tucoopy (Python)

<p class="centered-logo">
  <img src="assets/logo.png" width="320" alt="tucoopy">
</p>

`tucoopy` is a Python library for **TU (transferable utility) cooperative game theory**.
It provides:

- Classic game generators (glove, weighted voting, airport, bankruptcy, unanimity, ...)
- Classic solution concepts (Shapley, Banzhaf, least-core / nucleolus, kernel / prekernel, tau value, ...)
- Geometric helpers intended for **visualization** (core, epsilon-core, imputation set, Weber set, bargaining set)
- A JSON-friendly **animation spec** generator compatible with the `tucoopyjs` package

## Scope and design goals

- Focus on TU cooperative games (characteristic function games).
- Keep a clean, well-structured API: a small top-level surface, with most functionality organized in subpackages.
- Prefer correctness and clear diagnostics over maximum performance; many routines are exponential and intended for small `n`.
- Optional heavy dependencies:
  - `tucoopy[lp]` enables LP-based methods via `SciPy` (least-core, nucleolus, balancedness, bargaining set, ...)
  - `tucoopy[lp_alt]` enables an alternative LP backend via `PuLP`
  - `tucoopy[fast]` enables `NumPy`-based helpers (kernel / prekernel)
  - `tucoopy[viz]` enables simple 2-3 player visualization via `Matplotlib`

## Quick links

If you are new to the package, start here:

- [Quickstart](guides/quickstart.md): installation + minimal examples
- [API reference](reference/index.md): public API map (stable top-level vs. subpackages)
- [Theory overview](theory/index.md): core concepts and solution ideas
- [Geometry](library/geometry.md): geometric objects for visualization (core, epsilon-core, imputation, Weber, bargaining, ...)
- [Animation spec](guides/animation_spec.md): generating JSON specs consumed by the JS renderer
- [Roadmap](project/roadmap.md): implementation checklist / next steps
- [Contributing](project/contributions.md): how to contribute

## Minimal example

```py
from tucoopy import Game
from tucoopy.solutions import shapley_value

g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
print(shapley_value(g))  # [1.0, 1.0, 1.0]
```

