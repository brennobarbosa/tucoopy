# The core and related polyhedra

Many solution objects are defined as polyhedra.

## Core

Allocations in the core satisfy:

- Efficiency: $\sum_i x_i = v(N)$
- Coalitional rationality: $x(S) \ge v(S)$ for every non-empty proper coalition $S$

!!! note "Definition (core)"
    The core of a TU game $(N,v)$ is the set of allocations $x \in \mathbb{R}^n$ such that:

    $$\sum_{i \in N} x_i = v(N),$$
    and for every non-empty proper coalition $S$,
    $$x(S) = \sum_{i \in S} x_i \ge v(S).$$

!!! tip "Intuition"
    No coalition can profitably deviate: every coalition receives at least what it can guarantee on its own.

## $\epsilon$-core / least-core

The $\epsilon$-core relaxes coalitional rationality by $\epsilon \ge 0$:

- $x(S) \ge v(S) - \epsilon$

The least-core chooses the smallest possible $\epsilon$ (computed via LP when SciPy is available).

## Imputation set

- Efficiency + individual rationality ($x_i \ge v(\{i\})$)

!!! note "Definition (imputation set)"
    The imputation set is:

    $$I(v) = \left\{x \in \mathbb{R}^n : \sum_{i \in N} x_i = v(N),\; x_i \ge v(\{i\})\;\forall i\right\}.$$

For implementation-oriented details, see `../library/geometry.md`.

