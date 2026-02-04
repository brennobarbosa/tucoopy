# JS vs Python (renderer vs library)

Historically, this project used a "Python produces JSON, JS renders it" split.

In this repository you will find **the Python library** (`tucoopy`) and the JSON schemas used by `tucoopy.io`:

- `src/tucoopy/io/schemas/tucoop-animation.schema.json`
- `src/tucoopy/io/schemas/tucoop-game.schema.json`

The main compatibility boundary is the **JSON contract** (schemas), not internal architecture.

## What the browser-side renderer typically does

A (separate) JS/TS renderer can:

- validate specs against the schema;
- render allocations/sets/diagnostics (Canvas/SVG);
- optionally derive "cheap" analysis for very small `n` (when feasible).

## What should stay in Python

Rule of thumb: anything that depends on **LP** or is numerically delicate should be computed in Python.

Typical examples:

- least-core, nucleolus, modiclus (LP)
- balancedness (Bondareva-Shapley via LP)
- bargaining set (expensive; sampling/LP)
- kernel/pre-kernel (iterative; numerically sensitive)
- vertex enumeration and higher-dimensional projections

A reasonable pipeline is:

1. Run Python to compute `analysis` (solutions, sets, diagnostics).
2. Export the JSON spec.
3. The renderer only **renders** and optionally fills in missing cheap pieces for very small `n`.

## See also

- `../guides/animation_spec.md` (how to generate specs in Python)
- `../guides/analysis_contract.md` (what goes into `analysis`)

