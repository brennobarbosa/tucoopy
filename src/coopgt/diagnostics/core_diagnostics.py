"""
# Core-family diagnostics (core and epsilon-core membership).

This module provides computations around the coalition excess function
$e(S, x) = v(S) - x(S)$ and uses it to:

- test whether an allocation is in the core (``max_excess <= tol``),
- compute max-excess coalitions (ties),
- list the most violated coalitions for explanation/debug.

Examples
--------
>>> from tucoop import Game
>>> from tucoop.diagnostics.core_diagnostics import core_diagnostics, explain_core_membership
>>> g = Game.from_coalitions(n_players=2, values={0: 0, 1: 0, 2: 0, 3: 1})
>>> d = core_diagnostics(g, [0.5, 0.5])
>>> d.in_core
True
>>> explain_core_membership(g, [0.25, 0.75])[0].startswith("In the core")
True
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..base.config import DEFAULT_GEOMETRY_TOL
from ..base.types import GameProtocol
from ..base.exceptions import InvalidParameterError
from ..base.coalition import all_coalitions, coalition_sums, players
from ._excess_scan import scan_excesses


def _scan_core_constraints(
    game: GameProtocol, x: list[float], *, tol: float
) -> tuple[float, list[int], list[CoreViolation]]:
    rows: list[CoreViolation] = []
    mx, tight, raw = scan_excesses(game, x, tie_tol=float(tol), violation_threshold=float(tol))
    n = game.n_players
    for r in raw:
        rows.append(
            CoreViolation(
                coalition_mask=int(r.coalition_mask),
                players=list(players(int(r.coalition_mask), n_players=n)),
                vS=float(r.vS),
                xS=float(r.xS),
                excess=float(r.excess),
            )
        )
    rows.sort(key=lambda rr: (-rr.excess, rr.coalition_mask))
    return float(mx), list(tight), rows


@dataclass(frozen=True)
class CoreViolation:
    """
    One coalition constraint evaluation at allocation x.

    Excess: $e(S, x) = v(S) - x(S)$.
    A positive excess means coalition $S$ can improve by deviating ($x$ violates the core constraint).

    Parameters
    ----------
    coalition_mask : int
        Bitmask representing the coalition.
    players : list of int
        List of player indices in the coalition.
    vS : float
        Value of the coalition.
    xS : float
        Sum of allocation to coalition members.
    excess : float
        Excess value for the coalition.

    Examples
    --------
    >>> cv = CoreViolation(3, [0, 1], 1.0, 0.8, 0.2)
    >>> cv.excess
    0.2
    """

    coalition_mask: int
    players: list[int]
    vS: float
    xS: float
    excess: float


@dataclass(frozen=True)
class CoreDiagnostics:
    """
    Diagnostics for core membership at an allocation $x$.

    - ``max_excess`` is the maximum excess over all non-empty proper coalitions.
    - ``violations`` is the list of the most violated coalitions (sorted by excess desc).
    - ``tight_coalitions`` are coalitions achieving `max_excess` (within tol used in computation).

    Parameters
    ----------
    n_players : int
        Number of players in the game.
    vN : float
        Value of the grand coalition.
    sum_x : float
        Sum of the allocation vector.
    efficient : bool
        Whether the allocation is efficient.
    in_core : bool
        Whether the allocation is in the core.
    max_excess : float
        Maximum excess over all coalitions.
    tight_coalitions : list of int
        Coalitions achieving the maximum excess.
    violations : list of CoreViolation
        List of most violated coalitions.

    Examples
    --------
    >>> from tucoop.base.game import Game
    >>> from tucoop.diagnostics import core_diagnostics
    >>> g = Game([0, 0, 0, 1, 1, 1, 2])
    >>> x = [1, 1]
    >>> cd = core_diagnostics(g, x)
    >>> cd.in_core
    True
    >>> cd.max_excess
    0.0
    """

    n_players: int
    vN: float
    sum_x: float
    efficient: bool
    in_core: bool
    max_excess: float
    tight_coalitions: list[int]
    # Coalition(s) attaining max_excess (ties). Same as tight_coalitions, but named explicitly
    # for plotting/consumers that care about "who blocks" at x.
    blocking_coalitions: list[int]
    violations: list[CoreViolation]

    def to_dict(self) -> dict[str, object]:
        """
        Convert diagnostics to a dictionary for serialization.

        Returns
        -------
        dict
            Dictionary representation of diagnostics.

        Examples
        --------
        >>> cd.to_dict()
        {...}
        """
        return asdict(self)


def core_frame_highlight(game: GameProtocol, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL) -> dict[str, object]:
    """
    Small per-frame highlight payload for UI display.

    Intended for embedding into FrameSpec.highlights.

    Parameters
    ----------
    game : Game
        The cooperative game instance.
    x : list of float
        The candidate allocation vector.
    tol : float, optional
        Tolerance for numerical checks (default 1e-9).

    Returns
    -------
    dict
        Dictionary with highlight information for UI.

    Examples
    --------
    >>> from tucoop.base.game import Game
    >>> from tucoop.diagnostics import core_frame_highlight
    >>> g = Game([0, 0, 0, 1, 1, 1, 2])
    >>> x = [1, 1]
    >>> core_frame_highlight(g, x)
    {'in_core': True, 'efficient': True, ...}
    """
    d = core_diagnostics(game, x, tol=tol, top_k=0)
    blocking = int(d.blocking_coalitions[0]) if d.blocking_coalitions else None
    blocking_players = players(blocking, n_players=game.n_players) if blocking is not None else []
    return {
        "in_core": d.in_core,
        "efficient": d.efficient,
        "max_excess": d.max_excess,
        "tight_coalitions": d.tight_coalitions,
        "blocking_coalitions": d.blocking_coalitions,
        "blocking_coalition_mask": blocking,
        "blocking_players": blocking_players,
    }


def core_diagnostics(
    game: GameProtocol, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL, top_k: int = 8
) -> CoreDiagnostics:
    """
    Compute a compact explanation of why $x$ is (not) in the core.

    Parameters
    ----------
    game
        TU game.
    x
        Allocation vector (length n_players).
    tol
        Numerical tolerance. `x` is considered in the core if max_excess <= tol and efficient.
    top_k
        Return up to this many most violated coalitions.

    Returns
    -------
    CoreDiagnostics
        Diagnostics object for core membership.

    Examples
    --------
    >>> from tucoop import Game
    >>> from tucoop.diagnostics.core_diagnostics import core_diagnostics
    >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
    >>> core_diagnostics(g, [0.5, 0.5]).max_excess
    -0.5
    """
    n = game.n_players
    if len(x) != n:
        raise InvalidParameterError("x must have length n_players")

    grand = game.grand_coalition
    vN = float(game.value(grand))
    sum_x = float(sum(x))
    efficient = abs(sum_x - vN) <= tol

    mx, tight, rows = _scan_core_constraints(game, x, tol=tol)
    violations = rows[: max(0, int(top_k))]
    in_core = efficient and mx <= tol
    return CoreDiagnostics(
        n_players=n,
        vN=vN,
        sum_x=sum_x,
        efficient=efficient,
        in_core=in_core,
        max_excess=mx,
        tight_coalitions=tight,
        blocking_coalitions=tight,
        violations=violations,
    )


def explain_core_membership(
    game: GameProtocol, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL, top_k: int = 3
) -> list[str]:
    """
    Generate a short textual explanation about core membership of an allocation.

    This is a human-friendly wrapper around `core_diagnostics`. It reports
    (i) efficiency, (ii) whether the allocation is in the core, and (iii) the
    most violated coalitions when it is not.

    Parameters
    ----------
    game
        TU game.
    x
        Allocation vector.
    tol
        Numerical tolerance for efficiency and core constraints.
    top_k
        Maximum number of violations to compute (default 3). The explanation
        itself reports at most the most violated coalition plus the tight set.

    Returns
    -------
    list[str]
        One sentence per line, ready for UI tooltips/logs.

    Examples
    --------
    >>> from tucoop import Game
    >>> from tucoop.diagnostics.core_diagnostics import explain_core_membership
    >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
    >>> explain_core_membership(g, [0.5, 0.5])[0]
    'In the core (max excess=-0.5 <= tol=1e-09).'
    """
    d = core_diagnostics(game, x, tol=tol, top_k=top_k)
    lines: list[str] = []
    if not d.efficient:
        lines.append(f"Not efficient: sum(x)={d.sum_x:.6g} but v(N)={d.vN:.6g}.")
    if d.in_core:
        lines.append(f"In the core (max excess={d.max_excess:.6g} <= tol={tol:.6g}).")
        if d.blocking_coalitions:
            lines.append(f"Max-excess coalition(s) (mask): {d.blocking_coalitions}.")
        return lines
    lines.append(f"Not in the core (max excess={d.max_excess:.6g} > tol={tol:.6g}).")
    if d.violations:
        v0 = d.violations[0]
        lines.append(
            "Most violated coalition: "
            f"S={v0.coalition_mask} players={v0.players} excess={v0.excess:.6g} (v(S)={v0.vS:.6g}, x(S)={v0.xS:.6g})."
        )
    if d.tight_coalitions:
        lines.append(f"Tight coalitions (mask): {d.tight_coalitions}.")
    if d.blocking_coalitions and d.blocking_coalitions != d.tight_coalitions:
        lines.append(f"Max-excess coalition(s) (mask): {d.blocking_coalitions}.")
    return lines


def excesses(
    game: GameProtocol,
    x: list[float],
    *,
    include_empty: bool = False,
    include_grand: bool = False,
) -> dict[int, float]:
    """
    Excess function ``e(S, x) = v(S) - x(S)`` over coalitions.

    This is a small helper to export the **full excess vector** as a dict keyed by
    coalition mask.

    Parameters
    ----------
    game
        TU game.
    x
        Allocation vector (length ``n_players``).
    include_empty
        Whether to include the empty coalition (mask 0). Default False.
    include_grand
        Whether to include the grand coalition (mask ``2**n - 1``). Default False.

    Returns
    -------
    dict[int, float]
        Dict mapping coalition mask to excess ``v(S) - x(S)``.

    Raises
    ------
    InvalidParameterError
        If ``x`` does not have length ``n_players``.

    Examples
    --------
    >>> from tucoop import Game
    >>> from tucoop.diagnostics.core_diagnostics import excesses
    >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
    >>> ex = excesses(g, [0.5, 0.5])
    >>> ex[0b01]  # v({0})-x0
    -0.5
    """
    n = int(game.n_players)
    if len(x) != n:
        raise InvalidParameterError("x must have length n_players")

    x_sum = coalition_sums(x, n_players=n)
    out: dict[int, float] = {}
    for S in all_coalitions(n):
        if S == 0 and not include_empty:
            continue
        if S == int(game.grand_coalition) and not include_grand:
            continue
        out[int(S)] = float(game.value(int(S)) - float(x_sum[int(S)]))
    return out


def max_excess(game: GameProtocol, x: list[float]) -> float:
    """
    Maximum excess over all proper coalitions (non-empty, excluding the grand coalition).

    Examples
    --------
    >>> from tucoop import Game
    >>> from tucoop.diagnostics.core_diagnostics import max_excess
    >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
    >>> max_excess(g, [0.5, 0.5])
    -0.5
    """
    return float(core_diagnostics(game, x, tol=0.0, top_k=0).max_excess)


def is_in_core(game: GameProtocol, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL) -> bool:
    """
    Check if allocation is in the core (efficient and max_excess <= tol).

    Examples
    --------
    >>> from tucoop import Game
    >>> from tucoop.diagnostics.core_diagnostics import is_in_core
    >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
    >>> is_in_core(g, [0.5, 0.5])
    True
    """
    return bool(core_diagnostics(game, x, tol=tol, top_k=0).in_core)


def is_in_epsilon_core(
    game: GameProtocol, x: list[float], epsilon: float, *, tol: float = DEFAULT_GEOMETRY_TOL
) -> bool:
    """
    Check if allocation is in the epsilon-core (efficient and max_excess <= epsilon + tol).

    Examples
    --------
    >>> from tucoop import Game
    >>> from tucoop.diagnostics.core_diagnostics import is_in_epsilon_core
    >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
    >>> is_in_epsilon_core(g, [0.5, 0.5], epsilon=0.0)
    True
    """
    d = core_diagnostics(game, x, tol=tol, top_k=0)
    if not d.efficient:
        return False
    return float(d.max_excess) <= float(epsilon) + float(tol)


@dataclass(frozen=True)
class TightCoalitions:
    """
    Coalitions achieving the maximum excess (within tolerance).

    Examples
    --------
    >>> TightCoalitions(max_excess=0.0, coalitions=[1, 2]).coalitions
    [1, 2]
    """

    max_excess: float
    coalitions: list[int]


def tight_coalitions(game: GameProtocol, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL) -> TightCoalitions:
    """
    Find coalitions achieving the maximum excess (within tolerance).

    Examples
    --------
    >>> from tucoop import Game
    >>> from tucoop.diagnostics.core_diagnostics import tight_coalitions
    >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
    >>> tight_coalitions(g, [0.5, 0.5]).coalitions
    [1, 2]
    """
    d = core_diagnostics(game, x, tol=float(tol), top_k=0)
    return TightCoalitions(max_excess=float(d.max_excess), coalitions=list(d.blocking_coalitions))


def core_violations(game: GameProtocol, x: list[float], *, tol: float = DEFAULT_GEOMETRY_TOL) -> list[CoreViolation]:
    """
    List all core constraints violated by allocation x (excess > tol).

    Returns a list ordered by descending excess and coalition mask.

    Examples
    --------
    >>> from tucoop import Game
    >>> from tucoop.diagnostics.core_diagnostics import core_violations
    >>> g = Game.from_coalitions(n_players=2, values={0: 0, 1: 1, 2: 0, 3: 1})
    >>> core_violations(g, [0.0, 1.0])  # v({0})=1 violates x0>=1
    [CoreViolation(coalition_mask=1, players=[0], vS=1.0, xS=0.0, excess=1.0)]
    """
    # Deduplicates conversion/sorting logic with `core_diagnostics`.
    _mx, _tight, rows = _scan_core_constraints(game, x, tol=float(tol))
    return rows


__all__ = [
    "CoreDiagnostics",
    "CoreViolation",
    "core_diagnostics",
    "core_frame_highlight",
    "explain_core_membership",
    "excesses",
    "max_excess",
    "is_in_core",
    "is_in_epsilon_core",
    "TightCoalitions",
    "tight_coalitions",
    "core_violations",
]
