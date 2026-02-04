"""
# Möbius transform over the subset lattice.

This module provides a fast Möbius transform (and its inverse) for dense
functions defined on all coalitions. In cooperative game theory, applying the
transform to a characteristic function yields Harsanyi dividends.
"""

from __future__ import annotations

from typing import Sequence

from ..base.exceptions import InvalidParameterError
from ._mobius_impl import (
    _mobius_py,
    _inv_mobius_py,
    _mobius_numpy,
    _inv_mobius_numpy,
    _validate,
)


def mobius_transform(
    values: Sequence[float],
    *,
    n_players: int | None = None,
    backend: str = "auto",
) -> list[float]:
    r"""
    Compute the **Möbius transform** over the subset lattice (bitmask coalitions).

    Given a function $f(S)$ defined for all $S \subseteq N$, its Möbius transform
    $g$ is defined by:

    $$
    g(S) = \sum_{T \subseteq S} (-1)^{|S|-|T|} f(T).
    $$

    In cooperative game theory, applying this transform to the characteristic
    function $v(S)$ yields the **Harsanyi dividends** $d(S)$.

    Parameters
    ----------
    values : Sequence[float]
        Dense sequence of length $2^n$, indexed by coalition bitmask.
        The entry at index `mask` must correspond to $f(S)$ for that coalition.
    n_players : int | None, optional
        Number of players $n$. If omitted, it is inferred from ``len(values)``.
    backend : {"auto", "numpy", "python"}, default="auto"
        Backend used for the computation:
        - ``"auto"``: try NumPy implementation, fall back to pure Python,
        - ``"numpy"``: force NumPy (raises ImportError if unavailable),
        - ``"python"``: use the pure-Python implementation.

    Returns
    -------
    list[float]
        A new list of length $2^n$ containing the Möbius-transformed values.

    Notes
    -----
    - Complexity is $O(n \cdot 2^n)$ using the standard in-place fast transform.
    - This function operates purely on dense numeric sequences and is agnostic
      to the :class:`~tucoopy.base.game.Game` abstraction.
    - The Möbius transform is the algebraic foundation for:
        * Harsanyi dividends,
        * unanimity game decomposition,
        * several fast algorithms for solution concepts.

    Examples
    --------
    >>> # values indexed by bitmask for n=2 players
    >>> f = [0.0, 1.0, 2.0, 4.0]
    >>> g = mobius_transform(f, n_players=2)
    """
    _, n = _validate(values, n_players)

    if backend == "python":
        return _mobius_py(values, n)

    if backend == "numpy":
        return _mobius_numpy(values, n)

    if backend != "auto":
        raise InvalidParameterError("backend must be one of: 'auto', 'numpy', 'python'")

    # auto: try numpy, fall back to python
    try:
        return _mobius_numpy(values, n)
    except ImportError:
        return _mobius_py(values, n)


def inverse_mobius_transform(
    values: Sequence[float],
    *,
    n_players: int | None = None,
    backend: str = "auto",
) -> list[float]:
    r"""
    Compute the **inverse Möbius transform** (zeta transform over subsets).

    If $g$ is the Möbius transform of $f$, then the original function can be
    recovered by:

    $$
    f(S) = \sum_{T \subseteq S} g(T).
    $$

    Parameters
    ----------
    values : Sequence[float]
        Dense sequence of length $2^n$, indexed by coalition bitmask.
    n_players : int | None, optional
        Number of players $n$. If omitted, inferred from ``len(values)``.
    backend : {"auto", "numpy", "python"}, default="auto"
        Backend used for the computation (same semantics as
        :func:`mobius_transform`).

    Returns
    -------
    list[float]
        A new list of length $2^n$ containing the reconstructed values.

    Notes
    -----
    - Complexity is $O(n \cdot 2^n)$.
    - This operation is also known as the **subset zeta transform**.
    - Applying this to Harsanyi dividends reconstructs the original
      characteristic function.

    Examples
    --------
    >>> g = mobius_transform(f, n_players=2)
    >>> f_rec = inverse_mobius_transform(g, n_players=2)
    >>> f_rec == f
    True
    """
    _, n = _validate(values, n_players)

    if backend == "python":
        return _inv_mobius_py(values, n)

    if backend == "numpy":
        return _inv_mobius_numpy(values, n)

    if backend != "auto":
        raise InvalidParameterError("backend must be one of: 'auto', 'numpy', 'python'")

    try:
        return _inv_mobius_numpy(values, n)
    except ImportError:
        return _inv_mobius_py(values, n)


__all__ = ["mobius_transform", "inverse_mobius_transform"]
