# tucoopy structure

This package is organized to keep the public API explicit, small, and easy to navigate.

## Canonical modules

- `tucoopy.base`
  - fundamental structures (`Coalition` bitmask, `Game`)

- `tucoopy.properties`
  - properties / recognizers

- `tucoopy.io`
  - JSON IO + animation spec helpers (shared contract with the JS package)

- `tucoopy.backends`
  - adapters for optional dependencies (LP, NumPy, ...)

- `tucoopy.power`
  - power indices (voting / simple games)

- `tucoopy.solutions`
  - solution concepts (Shapley, Banzhaf, ...)

- `tucoopy.transforms`
  - representations / transforms (e.g. Harsanyi dividends)

- `tucoopy.geometry`
  - geometric objects/operations used for visualization (e.g. core vertices)

## Import rules (restricted)

- Internal code should import from the canonical modules above.
- Public re-exports live in:
  - each subpackage's `__init__.py` (e.g. `tucoopy.geometry.__init__`)
  - the top-level `tucoopy/__init__.py`, for convenience

Avoid introducing extra module "shims"/shortcuts like `tucoopy/core.py`: that makes the package harder to navigate.

