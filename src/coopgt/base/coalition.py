"""
## Coalition (bitmask) utilities.

In `tucoop`, a coalition is represented as a non-negative integer bitmask:

- Player $i$ corresponds to bit ``(1 << i)``.
- The empty coalition $\\varnothing$ is ``0``.
- The grand coalition $N$ is ``(1 << n_players) - 1``.

This convention is used across the package (games, solutions, geometry, and
diagnostics) because it is compact and fast.

The helpers in this module implement the basic iteration patterns needed by
cooperative game algorithms:

- Iterate all coalitions: `all_coalitions`
- Iterate subcoalitions (submasks): `subcoalitions`
- Convert between bitmasks and player lists: `players`, `mask_from_players`
- Efficiently compute coalition sums: `coalition_sum`, `coalition_sums`

Notes
-----
All functions assume **0-indexed** players.

Examples
--------
Basic coalition encoding:

>>> from tucoop.base.coalition import mask_from_players, players
>>> S = mask_from_players([0, 2])  # players {0,2}
>>> S
5
>>> players(S)
[0, 2]
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator

from .exceptions import InvalidCoalitionError

Coalition = int  # bitmask


def all_coalitions(n_players: int) -> Iterator[Coalition]:
    """
    Iterate over all coalitions of ``n_players`` as integer bitmasks.

    Coalitions are represented as bitmasks in the range

    $$
    0, 1, \\dots, 2^n - 1,
    $$

    where bit ``i`` indicates whether player $i$ is in the coalition.

    Parameters
    ----------
    n_players
        Number of players ``n`` (must be >= 0).

    Yields
    ------
    Coalition
        Coalition bitmask (an ``int``).

    Raises
    ------
    InvalidCoalitionError
        If ``n_players < 0``.

    Examples
    --------
    All coalitions for ``n=2``::

        >>> list(all_coalitions(2))
        [0, 1, 2, 3]

    Notes
    -----
    The number of yielded masks is ``2**n_players``.
    """
    if n_players < 0:
        raise InvalidCoalitionError("n_players must be >= 0")
    for mask in range(1 << int(n_players)):
        yield int(mask)


def subcoalitions(coalition: Coalition) -> Iterator[Coalition]:
    """
    Iterate over all subcoalitions $T \\subseteq S$ of a coalition $S$ (as submasks).

    This yields all submasks of $S$ (including $S$ itself and the empty
    coalition $\\varnothing$) using the classic bit-trick:

    ``` python
    T = S
    while True:
        yield T
        if T == 0: break
        T = (T - 1) & S
    ```

    Parameters
    ----------
    coalition
        Coalition bitmask ``S`` (must be >= 0).

    Yields
    ------
    Coalition
        Subcoalition bitmask ``T``.

    Raises
    ------
    InvalidCoalitionError
        If ``coalition < 0``.

    Examples
    --------
    Subcoalitions of ``S = 0b101`` (players {0,2})::

        >>> list(subcoalitions(0b101))
        [5, 4, 1, 0]

    Notes
    -----
    - The order is descending by construction (starting at ``S`` down to ``0``).
    - Number of yielded masks is ``2**k`` where ``k = popcount(S)``.
    """
    S = int(coalition)
    if S < 0:
        raise InvalidCoalitionError("coalition mask must be >= 0")
    T = S
    while True:
        yield int(T)
        if T == 0:
            break
        T = (T - 1) & S


def size(coalition: Coalition) -> int:
    """
    Return the number of players in a coalition (popcount).

    Parameters
    ----------
    coalition
        Coalition bitmask (must be >= 0).

    Returns
    -------
    int
        The coalition cardinality ``|S|``.

    Raises
    ------
    InvalidCoalitionError
        If ``coalition < 0``.

    Examples
    --------
        >>> size(0b101)
        2
        >>> size(0)
        0
    """
    S = int(coalition)
    if S < 0:
        raise InvalidCoalitionError("coalition mask must be >= 0")
    return S.bit_count()


def players(coalition: Coalition, *, n_players: int) -> list[int]:
    """
    Convert a coalition mask to the sorted list of player indices.

    Parameters
    ----------
    coalition
        Coalition bitmask (must be >= 0).
    n_players
        Number of players ``n`` (must be >= 0). Bits above ``n-1`` are ignored.

    Returns
    -------
    list[int]
        Sorted list of player indices included in the coalition.

    Raises
    ------
    InvalidCoalitionError
        If ``coalition < 0`` or ``n_players < 0``.

    Examples
    --------
        >>> players(0b101, n_players=3)
        [0, 2]
        >>> players(0b101, n_players=2)  # ignores bit 2 because n_players=2
        [0]
    """
    S = int(coalition)
    if S < 0:
        raise InvalidCoalitionError("coalition mask must be >= 0")
    if n_players < 0:
        raise InvalidCoalitionError("n_players must be >= 0")
    out: list[int] = []
    for i in range(int(n_players)):
        if S & (1 << i):
            out.append(i)
    return out


def mask_from_players(ps: Iterable[int]) -> Coalition:
    """
    Build a coalition mask from an iterable of player indices.

    Parameters
    ----------
    ps
        Iterable of player indices (each must be >= 0).

    Returns
    -------
    Coalition
        Coalition bitmask.

    Raises
    ------
    InvalidCoalitionError
        If any player index is negative.

    Examples
    --------
        >>> mask_from_players([0, 2])
        5
        >>> mask_from_players([])
        0

    Notes
    -----
    Duplicate indices are harmless (bitwise OR).
    """
    mask = 0
    for p in ps:
        ip = int(p)
        if ip < 0:
            raise InvalidCoalitionError(f"player index must be >= 0, got {p}")
        mask |= 1 << ip
    return int(mask)


def grand_coalition(n_players: int) -> Coalition:
    """
    Return the grand coalition mask ``(1<<n) - 1``.

    Parameters
    ----------
    n_players
        Number of players ``n`` (must be >= 0).

    Returns
    -------
    Coalition
        Bitmask for the grand coalition (all players included).

    Raises
    ------
    InvalidCoalitionError
        If ``n_players < 0``.

    Examples
    --------
        >>> grand_coalition(3)
        7
        >>> bin(grand_coalition(4))
        '0b1111'
    """
    if n_players < 0:
        raise InvalidCoalitionError("n_players must be >= 0")
    return int((1 << int(n_players)) - 1)


def coalition_sum(coalition: Coalition, x: Iterable[float], *, n_players: int) -> float:
    """
    Sum an allocation vector over a coalition.

    Computes:

    $$
    x(S) = \\sum_{i \\in S} x_i.
    $$

    Parameters
    ----------
    coalition
        Coalition bitmask ``S``.
    x
        Allocation vector (iterable of length ``n_players``).
    n_players
        Number of players ``n``. Used to validate the length of ``x`` and to
        decide which bits are considered.

    Returns
    -------
    float
        The coalition sum ``x(S)``.

    Raises
    ------
    InvalidCoalitionError
        If ``x`` has length different from ``n_players``.

    Examples
    --------
        >>> coalition_sum(0b101, [1.0, 2.0, 3.0], n_players=3)
        4.0
        >>> coalition_sum(0, [1.0, 2.0], n_players=2)
        0.0
    """
    S = int(coalition)
    xs = list(x)
    n = int(n_players)
    if n < 0:
        raise InvalidCoalitionError("n_players must be >= 0")
    if len(xs) != n:
        raise InvalidCoalitionError("x must have length n_players")
    s = 0.0
    for i in range(n):
        if S & (1 << i):
            s += float(xs[i])
    return float(s)


def coalition_sums(x: Iterable[float], *, n_players: int) -> list[float]:
    """
    Precompute coalition sums $x(S)$ for all coalitions $S$.

    Returns an array ``out`` of length ``2**n_players`` such that:

    $$
    \\text{out}[S] = x(S) = \\sum_{i \\in S} x_i.
    $$

    Implementation
    --------------
    Uses an $O(2^n)$ dynamic program based on the least-significant bit:

    $$
    x(S) = x(S \\setminus \\{i\\}) + x_i,
    $$

    where $i$ is the index of the least-significant set bit of $S$.

    Parameters
    ----------
    x
        Allocation vector (iterable of length ``n_players``).
    n_players
        Number of players ``n``.

    Returns
    -------
    list[float]
        List ``out`` with ``out[mask] = x(mask)`` for all masks in
        ``0..(1<<n_players)-1``.

    Raises
    ------
    InvalidCoalitionError
        If ``x`` has length different from ``n_players``.

    Examples
    --------
        >>> out = coalition_sums([1.0, 2.0, 3.0], n_players=3)
        >>> out[0]          # empty coalition
        0.0
        >>> out[0b101]      # players {0,2}
        4.0
        >>> out[0b111]      # grand coalition
        6.0

    Notes
    -----
    This is a building block for excess/surplus computations, where many
    coalition sums are needed repeatedly.
    """
    xs = list(x)
    n = int(n_players)
    if n < 0:
        raise InvalidCoalitionError("n_players must be >= 0")
    if len(xs) != n:
        raise InvalidCoalitionError("x must have length n_players")

    out = [0.0] * (1 << n)
    for mask in range(1, 1 << n):
        lsb = mask & -mask
        i = (lsb.bit_length() - 1)
        out[mask] = out[mask ^ lsb] + float(xs[i])
    return out
