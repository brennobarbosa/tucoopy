# tucoopy-py Structure

This package is organized to keep the public API explicit, small, and easy to read.

## Canonical modules

- `tucoopy.base`
  - foundational data structures (`Coalition` bitmasks, `Game`)

- `tucoopy.properties`
  - game properties / recognizers

- `tucoopy.io`
  - JSON IO + animation spec helpers (shared contract with JS)

- `tucoopy.backends`
  - adapters for optional dependencies (LP, NumPy, ...)

- `tucoopy.power`
  - voting/simple-game power indices

- `tucoopy.solutions`
  - solution concepts (Shapley, Banzhaf, ...)

- `tucoopy.transforms`
  - representations / transforms (e.g. Harsanyi dividends)

- `tucoopy.geometry`
  - geometric objects/operations used for visualization (e.g. core vertices)

## Import rules (constrained)

- Internal code should import from the canonical modules above.
- Public re-exports live in:
  - each subpackage `__init__.py` (e.g. `tucoopy.geometry.__init__`)
  - the top-level `tucoopy/__init__.py` for convenience

Avoid introducing extra module shims like `tucoopy/core.py`. They make the package harder to navigate.
