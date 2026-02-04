# tucoop-py Structure

This package is organized to keep the public API explicit, small, and easy to read.

## Canonical modules

- `tucoop.base`
  - foundational data structures (`Coalition` bitmasks, `Game`)

- `tucoop.properties`
  - game properties / recognizers

- `tucoop.io`
  - JSON IO + animation spec helpers (shared contract with JS)

- `tucoop.backends`
  - adapters for optional dependencies (LP, NumPy, ...)

- `tucoop.power`
  - voting/simple-game power indices

- `tucoop.solutions`
  - solution concepts (Shapley, Banzhaf, ...)

- `tucoop.transforms`
  - representations / transforms (e.g. Harsanyi dividends)

- `tucoop.geometry`
  - geometric objects/operations used for visualization (e.g. core vertices)

## Import rules (constrained)

- Internal code should import from the canonical modules above.
- Public re-exports live in:
  - each subpackage `__init__.py` (e.g. `tucoop.geometry.__init__`)
  - the top-level `tucoop/__init__.py` for convenience

Avoid introducing extra module shims like `tucoop/core.py`. They make the package harder to navigate.
