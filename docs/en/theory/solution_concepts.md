# Solution concepts (overview)

This library implements multiple classic solution concepts. At a high level:

- Values (single-point selections): Shapley value, Banzhaf value, tau value.
- LP-based selections (optional SciPy): least-core, nucleolus, pre-nucleolus.
- Complementarity-based selections (optional NumPy): kernel, prekernel.

For the implementation-facing summary, see `solutions.md`.

## Shapley value

!!! note "Definition"
    The Shapley value assigns to each player $i$ the average marginal contribution over all permutations:

    $$\varphi_i(v) = \sum_{S \subseteq N\setminus\{i\}} \frac{|S|!(n-|S|-1)!}{n!}\,\bigl(v(S\cup\{i\})-v(S)\bigr).$$

!!! tip "Intuition"
    Each ordering of player arrivals is equally likely; a player's payoff is its expected marginal contribution.
