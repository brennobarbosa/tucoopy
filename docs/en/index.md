# tucoopy (Python)

<p align="center">
  <img src="assets/tucoopy-logo.png" width="320" alt="tucoopy">
</p>

`tucoopy` is a Python library for **TU (transferable utility) cooperative game theory**.
It provides:

- Classic game generators (glove, weighted voting, airport, bankruptcy, unanimity, ...)
- Classic solution concepts (Shapley, Banzhaf, least-core / nucleolus, kernel / prekernel, tau value, ...)
- Geometry helpers intended for **visualization** (core, epsilon-core, imputation set, Weber set, bargaining set)
- A JSON-friendly **animation spec** generator that can be consumed by the JS renderer in this monorepo

## Scope and design goals

- Focus on TU cooperative games (characteristic function games).
- Keep a clean, well-structured API: a small top-level surface, with most functionality in subpackages.
- Prefer correctness and clear diagnostics over maximum performance; many routines are exponential and intended for small `n`.
- Optional heavy dependencies:
  - `tucoopy[lp]` enables LP-based methods via SciPy (least-core, nucleolus, balancedness, bargaining set, ...)
  - `tucoopy[fast]` enables NumPy-based helpers (kernel / prekernel)

## Quick links

If you are new to the package, start here:

- `api.md`: public API map (stable top-level vs subpackages)
- `quickstart.md`: install + minimal examples
- `solutions.md`: solution concepts (Shapley, Banzhaf, nucleolus, kernel, ...)
- `geometry.md`: geometric objects for visualization (core, epsilon-core, imputation, Weber, bargaining, ...)
- `animation_spec.md`: generating JSON specs consumed by the JS renderer
- `roadmap.md`: implementation checklist / next steps

## Minimal example

```py
from tucoopy import Game
from tucoopy.solutions import shapley_value

g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
print(shapley_value(g))  # [1.0, 1.0, 1.0]
```
