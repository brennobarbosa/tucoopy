# Solution concepts (overview)

This library implements several classic solution concepts. At a high level:

- Values (single-point selections): Shapley value, Banzhaf value, $\tau$ value.
- LP-based selections (SciPy optional): least-core, nucleolus, pre-nucleolus.
- Complementarity-based selections (NumPy optional): kernel, prekernel.

## Shapley value

!!! note "Definition"
    The Shapley value assigns to each player $i$ the average marginal contribution over all permutations:

    $$\varphi_i(v) = \sum_{S \subseteq N\setminus\{i\}} \frac{|S|!(n-|S|-1)!}{n!}\,\bigl(v(S\cup\{i\})-v(S)\bigr).$$

!!! tip "Intuition"
    Each arrival order is equally likely; a player's payoff is their expected marginal contribution.

