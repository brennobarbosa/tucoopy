"""
# Tau value helpers.

This module provides the utopia payoff and minimal rights vectors, and a tau value implementation.
"""

from __future__ import annotations

from ..base.types import GameProtocol
from ..base.exceptions import InvalidGameError
from ..base.coalition import all_coalitions


def utopia_payoff(game: GameProtocol) -> list[float]:
    """
    Compute the **utopia payoff** vector $M$ of a TU game.

    The utopia payoff of player $i$ is defined as:

    $$
    M_i = v(N) - v(N \\setminus \\{i\\}),
    $$

    i.e. the maximum amount player $i$ could hope to obtain if the rest of the
    players formed the grand coalition without them.

    Parameters
    ----------
    game : GameProtocol
        TU game.

    Returns
    -------
    list[float]
        Utopia payoff vector of length `n_players`.

    Notes
    -----
    - $M$ is sometimes called the *marginal vector at the grand coalition*.
    - It provides an upper bound for reasonable imputations.
    - The utopia payoff plays a central role in the definition of the
      **minimal rights** and the **tau value**.

    Examples
    --------
    >>> M = utopia_payoff(g)
    >>> len(M) == g.n_players
    True
    """
    n = game.n_players
    N = game.grand_coalition
    vN = game.value(N)
    M = [0.0] * n
    for i in range(n):
        M[i] = float(vN - game.value(N & ~(1 << i)))
    return M


def minimal_rights(game: GameProtocol, M: list[float] | None = None) -> list[float]:
    """
    Compute the **minimal rights** vector $m$ of a TU game.

    Given the utopia payoff vector $M$, the minimal right of player $i$ is:

    $$
    m_i = \\max_{S \\ni i}
          \\left[ v(S) - \\sum_{j \\in S \\setminus \\{i\\}} M_j \\right].
    $$

    Intuitively, this represents the minimum payoff player $i$ can claim
    without being blocked by any coalition.

    Parameters
    ----------
    game : GameProtocol
        TU game.
    M : list[float] | None, optional
        Precomputed utopia payoff vector. If None, it is computed internally.

    Returns
    -------
    list[float]
        Minimal rights vector of length `n_players`.

    Notes
    -----
    - The minimal rights vector depends on the utopia payoff.
    - Together, $(m, M)$ define a line segment used to construct the
      **tau value**.
    - This vector provides a lower bound for reasonable imputations.

    Examples
    --------
    >>> m = minimal_rights(g)
    >>> len(m) == g.n_players
    True
    """
    n = game.n_players
    if M is None:
        M = utopia_payoff(game)

    m = [-float("inf")] * n
    for S in all_coalitions(n):
        if S == 0:
            continue
        vS = float(game.value(S))
        for i in range(n):
            if not (S & (1 << i)):
                continue
            rhs = 0.0
            for j in range(n):
                if j == i:
                    continue
                if S & (1 << j):
                    rhs += float(M[j])
            m[i] = max(m[i], vS - rhs)

    # If some player never appears (should not happen), clamp to 0.
    return [0.0 if v == -float("inf") else float(v) for v in m]


def tau_value(game: GameProtocol, *, tol: float = 1e-9) -> list[float]:
    """
    Compute the **tau value** of a TU game.

    The tau value is defined (when applicable) as the intersection of the
    imputation hyperplane with the line segment connecting the minimal
    rights vector $m$ and the utopia payoff vector $M$:

    $$
    \\tau = m + \\alpha (M - m),
    $$

    where the scalar $\\alpha$ is chosen such that:

    $$
    \\sum_i \\tau_i = v(N).
    $$

    Parameters
    ----------
    game : GameProtocol
        TU game.
    tol : float, default=1e-9
        Numerical tolerance for detecting degeneracy.

    Returns
    -------
    list[float]
        Tau value allocation.

    Raises
    ------
    InvalidGameError
        If the interpolation between $m$ and $M$ is ill-defined
        (degenerate case where the line cannot intersect the imputation plane).

    Notes
    -----
    - The tau value is particularly well-behaved for **quasi-balanced games**.
    - It lies on the line segment between minimal rights and utopia payoff,
      balancing lower and upper bounds on player claims.
    - If $\\sum M_i = \\sum m_i = v(N)$, the tau value coincides with $m$.

    Examples
    --------
    >>> tau = tau_value(g)
    >>> len(tau) == g.n_players
    True
    """
    n = game.n_players
    vN = float(game.value(game.grand_coalition))
    M = utopia_payoff(game)
    m = minimal_rights(game, M=M)

    sum_m = sum(m)
    sum_M = sum(M)
    denom = sum_M - sum_m
    if abs(denom) <= tol:
        # Line is degenerate: m and M have same sum.
        if abs(vN - sum_m) <= tol:
            return [float(v) for v in m]
        raise InvalidGameError("tau_value undefined: sum(M) == sum(m) but != v(N)")

    alpha = (vN - sum_m) / denom
    return [float(m[i] + alpha * (M[i] - m[i])) for i in range(n)]
