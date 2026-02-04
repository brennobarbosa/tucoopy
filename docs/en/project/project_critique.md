# A direct critique of `tucoopy`

This document is intentionally critical. The idea is to record:

- real weaknesses (technical and product),
- maintenance risks,
- theory / state-of-the-art gaps,
- opportunities for performance/robustness,

and suggest the smallest set of changes that significantly increases project predictability.

## 0) Executive summary

`tucoopy` is already in a good place as an **MVP** (TU games, core solution concepts and sets, JSON contract + examples).
The biggest risk today is not "missing features", but:

1) **numerical robustness + LP** (degeneracy, tolerances, solver differences),
2) **public contract** (avoid drift and repeated refactors),
3) **predictable performance** (clear limits and approximate paths).

## 1) Strengths (where the project is above average)

- A reasonably clear **layered architecture**:
  - base (game/coalition/config),
  - solutions (single-valued),
  - geometry (set-valued / polyhedra),
  - diagnostics (explanations),
  - io/schema (contract).
- Using the **JSON contract** as the interface boundary: a correct and scalable decision.
- Well-scoped **optional dependencies**: `lp`, `fast`, `viz` (no forced SciPy/NumPy/Matplotlib in the core).
- Existing **tests** that cover real flows (rare for small math libraries).

## 2) Where it is fragile (important weaknesses)

### 2.1 Numerical robustness (LP and polyhedra)

Problem: cooperative game theory quickly becomes LP/polyhedra, and this is an area where:

- degeneracy is common,
- tolerances change results (especially "tight" constraints),
- different solvers (SciPy/HiGHS vs PuLP/CBC) may diverge in details,
- vertex enumeration can blow up or fail silently.

High-value improvements:

- standardize *always* using `tol` (in one place) and document what it means (feasibility vs dedup),
- ensure diagnostics return "why it got weird" (slacks, residuals, affine dimension, boundedness),
- make the LP backend "contract" more explicit (what is expected, what can be `None`).

### 2.2 Performance: risks of naive usage

The package has intrinsically explosive routines:

- coalitions: $2^n$,
- Weber set: $n!$,
- vertices/projections: combinatorial,
- kernel/prekernel: iterative.

Risk: a user calls a function at `n=20` and thinks it "hung".

Improvements:

- keep conservative default limits,
- error messages should always explain the cost and the alternative (sampling/approx),
- provide "approx via sampling" paths where appropriate (especially geometry/projection).

### 2.3 API/contract: structural churn is expensive

If users don't know which imports are stable, they become hostage to the repo.

The project already started improving this with a "minimal public API", but it still needs:

- actually applying a deprecation policy (even in 0.x),
- avoiding "ghost files" (especially on Windows/OneDrive),
- treating docs as part of the contract (CI helps a lot).

## 3) Missing features / gaps (content and state of the art)

### 3.1 Solutions/sets (cooperative game theory)

Even ignoring "public goods" and rare indices, there are natural gaps:

- **nucleolus**: constraint generation and Kohlberg criteria are often used to scale better than enumerating everything at once (depends on desired scope).
- **kernel/prekernel**: usually require care with numerical stability, stopping criteria, and diagnostics (in practice, "converged" and "quality of point" matter).
- **bargaining set** and variants: exact computation is heavy; sampling is fine but needs good diagnostics (false positives/negatives).
- **vertex enumeration**: to go beyond `n=3`, the "right" path usually involves tools like `cddlib`/`pycddlib` or `polymake` (likely optional, but worth documenting).

### 3.2 Properties and recognizers

Compared to mature toolboxes (MATLAB TuGames, R CoopGame), you typically see:

- a larger catalog of sanity checks and canonical examples,
- helper functions for normalization and standard transforms,
- more "theory tests" (not only software tests).

This project has already progressed, but can still improve:

- convexity/balancedness checks with better diagnostics,
- small examples for each property (including counterexamples).

### 3.3 Power indices

The package has a good set of classic indices, but there will always be "one more index".
The bigger risk is not missing an index, but:

- inconsistent definitions (normalization, domain: simple vs TU),
- confusing documentation of what is supported.

## 4) Comparison with other systems (where `tucoopy` fits)

### MATLAB (TuGames toolbox, etc.)

Typically: lots of content, ready-to-use functions and examples; but:

- depends on MATLAB,
- web integration/rendering is not a focus,
- architecture is not always modular.

`tucoopy` can compete well if it prioritizes:

- a clear contract,
- good docs,
- simple installation + optional extras.

### R (CoopGame, etc.)

R is strong for analysis and quick visualization, but:

- performance/engineering can be uneven,
- integration with a custom JS renderer is rare.

### Polyhedron libraries (cddlib/polymake)

These are state of the art for V/H representations and vertices.
`tucoopy` doesn't need to compete directly: it can integrate optionally, or use them as references.

## 5) Concrete performance opportunities

Things that often yield big performance gains without rewriting everything:

- **avoid Python loops** where repeated coalition sums happen: utilities (e.g. `coalition_sum`) help.
- **cache** where cost is high but input repeats (value function + coalition iterators).
- optional **NumPy** for small linear algebra routines (the `fast` extra exists, but can be expanded carefully).
- transforms in $O(n 2^n)$ (Mobius) are already a good step; apply similar ideas where there is hidden $3^n$ work.

Not worth it for the MVP (high cost):

- implementing generic vertex enumeration for `n>3` without a specialized backend,
- optimizing nucleolus/kernels for large `n` without clearly defining the target audience.

## 6) Priority recommendation (suggested order)

1) **Stabilize the contract and docs** (already improved a lot with CI + public_api + deprecation).
2) **Improve numerical diagnostics** (explain "why", not just "error").
3) **More theory tests** (canonical cases and counterexamples).
4) **Targeted performance** (real hotspots measured, not "optimization by intuition").

## 7) Conclusion

The project is on a good track and has real potential as a lightweight reference (educational + practical).
To get there, the focus must be predictability:

- what is stable,
- what is expensive,
- what is approximate,
- and how to debug when solvers/numbers "don't match".
