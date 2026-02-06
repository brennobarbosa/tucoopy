"""
# Least-core (set-valued object + LP helpers).

The least-core is the epsilon-core with the smallest feasible ``epsilon*``. This
module defines `LeastCore` as a thin wrapper that:

- computes/stores the optimal ``epsilon*`` (when an LP backend is available),
- exposes a corresponding `tucoopy.geometry.EpsilonCore` / polyhedron, and
- provides convenience methods for membership and small-n geometry operations.

In addition, this module provides LP-based helpers to compute ``epsilon*`` and
select representative points from the least-core polytope.

Examples
--------
>>> from tucoopy import Game
>>> from tucoopy.geometry import LeastCore
>>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
>>> lc = LeastCore(g)
>>> lc.contains([0.5, 0.5])  # doctest: +SKIP
True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

from ..base.config import DEFAULT_GEOMETRY_MAX_DIM, DEFAULT_GEOMETRY_MAX_PLAYERS, DEFAULT_GEOMETRY_TOL
from ..base.coalition import all_coalitions
from ..base.exceptions import InvalidGameError, InvalidParameterError
from ..base.types import GameProtocol
from .epsilon_core_set import EpsilonCore
from .polyhedron import PolyhedralSet


SelectionMethod: TypeAlias = Literal["chebyshev_center", "any_feasible"]


if TYPE_CHECKING:  # pragma: no cover
    from ..diagnostics.linprog_diagnostics import LinprogDiagnostics


@dataclass(frozen=True)
class LeastCoreResult:
    x: list[float]
    epsilon: float
    tight: list[int] | None = None
    dual_weights: dict[int, float] | None = None
    lp: "LinprogDiagnostics | None" = None


@dataclass(frozen=True)
class LeastCorePointResult:
    x: list[float]
    epsilon: float
    method: str


@dataclass
class LeastCore:
    """
    Least-core as a **set-valued** object.

    Background
    ----------
    For a TU game, the (standard) core is the set of efficient allocations that
    satisfy all coalitional constraints:

    $$
    \\sum_i x_i = v(N), \\qquad x(S) \\ge v(S) \\; \\text{for all } \\varnothing \\ne S \\subsetneq N.
    $$

    When the core is empty, a common relaxation is the **epsilon-core**:

    $$
    \\sum_i x_i = v(N), \\qquad x(S) \\ge v(S) - \\varepsilon \\; \\text{for all } \\varnothing \\ne S \\subsetneq N.
    $$

    The **least-core** is the epsilon-core at the *smallest* relaxation level
    $\\varepsilon^*$ for which the epsilon-core becomes non-empty:

    $$
    \\varepsilon^* = \\min\\{\\varepsilon \\ge 0 : \\text{epsilon-core}(\\varepsilon) \\ne \\varnothing\\}.
    $$

    This class is a thin wrapper around ``EpsilonCore(game, epsilon*)``, computed
    lazily on first access.

    Lazy evaluation / dependencies
    ------------------------------
    The value ``epsilon`` is computed on demand via `least_core_epsilon_star`
    in this module (which requires an LP backend at runtime, typically via
    ``pip install "tucoopy[lp]"``). Once computed, it is cached in ``_epsilon``.

    Parameters
    ----------
    game
        TU game.
    restrict_to_imputation
        If True, additionally enforce individual rationality ``x_i >= v({i})``
        (handled as bounds) when forming the underlying polyhedron.
        This selects the least-core *within* the imputation set.
    tol
        Numerical tolerance forwarded to the LP solver and set-membership checks.
    _epsilon
        Optional cached value for ``epsilon`` (internal). If provided, the LP solve
        is skipped.

    Attributes
    ----------
    epsilon
        The least-core value ``epsilon*`` (computed lazily and cached).
    epsilon_core
        The epsilon-core object at ``epsilon*``.
    poly
        The underlying polyhedral representation (``PolyhedralSet``) of the least-core.

    Notes
    -----
    - ``LeastCore`` is set-valued. If you need a *single-valued selector*, use
      a routine like ``least_core_point`` (e.g., Chebyshev center) on top of this set.
    - The method ``vertices`` enumerates vertices using a brute-force routine intended
      for small n (visualization).

    Examples
    --------
    Minimal example (may have empty core; least-core always exists for TU games
    under this LP formulation):

    >>> from tucoopy import Game
    >>> from tucoopy.geometry import LeastCore
    >>> g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
    >>> lc = LeastCore(g)
    >>> eps = lc.epsilon
    >>> eps
    0.0
    >>> lc.sample_point()  # one least-core point (here also core point)
    [1.0, 1.0, 1.0]
    """

    game: GameProtocol
    restrict_to_imputation: bool = False
    tol: float = DEFAULT_GEOMETRY_TOL
    _epsilon: float | None = None

    @property
    def epsilon(self) -> float:
        """
        Least-core value ``epsilon*`` (computed lazily).

        Notes
        -----
        This requires an LP backend (typically SciPy/HiGHS via ``tucoopy[lp]``).

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import LeastCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> LeastCore(g).epsilon  # doctest: +SKIP
        0.0
        """
        if self._epsilon is None:
            self._epsilon = float(least_core_epsilon_star(self.game, tol=float(self.tol)))
        return float(self._epsilon)

    @property
    def epsilon_core(self) -> EpsilonCore:
        """
        Epsilon-core object at ``epsilon*``.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import LeastCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> lc = LeastCore(g)
        >>> lc.epsilon_core.epsilon  # doctest: +SKIP
        0.0
        """
        return EpsilonCore(self.game, self.epsilon, restrict_to_imputation=self.restrict_to_imputation)

    @property
    def poly(self) -> PolyhedralSet:
        """
        Underlying polyhedral representation of the least-core.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import LeastCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> LeastCore(g).poly.n_vars  # doctest: +SKIP
        2
        """
        return self.epsilon_core.poly

    def contains(self, x: list[float], *, tol: float | None = None) -> bool:
        """
        Check whether x lies in the least-core (within tolerance).

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import LeastCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> LeastCore(g).contains([0.5, 0.5])  # doctest: +SKIP
        True
        """
        t = float(self.tol if tol is None else tol)
        return self.epsilon_core.contains(x, tol=t)

    def check(self, x: list[float], *, tol: float | None = None, top_k: int = 8):
        """
        Return least-core membership diagnostics for $x$.

        Notes
        -----
        This delegates to `tucoopy.diagnostics.least_core_diagnostics.least_core_diagnostics`.
        If the LP backend is unavailable, the returned diagnostics will have
        ``available=False``.
        """
        from ..diagnostics.least_core_diagnostics import least_core_diagnostics

        t = float(self.tol if tol is None else tol)
        return least_core_diagnostics(self.game, x, tol=t, top_k=top_k)

    def explain(self, x: list[float], *, tol: float | None = None, top_k: int = 3) -> list[str]:
        """
        Return a short human-readable explanation of least-core membership.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import LeastCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> LeastCore(g).explain([0.5, 0.5])[0]  # doctest: +SKIP
        'In the least-core (epsilon*=0).'
        """
        d = self.check(x, tol=tol, top_k=top_k)
        if not d.available:
            return [f"Least-core diagnostics unavailable: {d.reason}"]
        if d.epsilon_star is None or d.epsilon_core is None:
            return ["Least-core diagnostics unavailable (missing epsilon*)."]
        if d.epsilon_core.in_set:
            return [f"In the least-core (epsilon*={d.epsilon_star:.6g})."]
        return [f"Not in the least-core (epsilon*={d.epsilon_star:.6g}, max_excess={d.epsilon_core.max_excess:.6g})."]

    def is_empty(self) -> bool:
        """
        Return True if the least-core polytope is empty.

        Notes
        -----
        This delegates to the underlying epsilon-core polyhedron. Requires an LP backend.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import LeastCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> LeastCore(g).is_empty()  # doctest: +SKIP
        False
        """
        return self.poly.is_empty()

    def sample_point(self) -> list[float] | None:
        """
        Attempt to sample one point from the least-core.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import LeastCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> LeastCore(g).sample_point()  # doctest: +SKIP
        [0.5, 0.5]
        """
        return self.poly.sample_point()

    def chebyshev_center(self) -> tuple[list[float], float] | None:
        """
        Compute the Chebyshev center of the least-core polytope.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import LeastCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> LeastCore(g).chebyshev_center()  # doctest: +SKIP
        ([0.5, 0.5], 0.0)
        """
        return self.poly.chebyshev_center()

    def extreme_points(
        self, *, tol: float = DEFAULT_GEOMETRY_TOL, max_dim: int = DEFAULT_GEOMETRY_MAX_DIM
    ) -> list[list[float]]:
        """
        Enumerate extreme points of the least-core (small dimension).

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import LeastCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> LeastCore(g).extreme_points(max_dim=2)  # doctest: +SKIP
        [[0.0, 1.0], [1.0, 0.0]]
        """
        return self.poly.extreme_points(tol=tol, max_dim=max_dim)

    def project(
        self,
        dims: tuple[int, ...] | list[int],
        *,
        tol: float = DEFAULT_GEOMETRY_TOL,
        max_dim: int = DEFAULT_GEOMETRY_MAX_DIM,
    ) -> list[list[float]]:
        """
        Project the least-core to selected coordinates.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import LeastCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> LeastCore(g).project((0,), max_dim=2)  # doctest: +SKIP
        [[0.0], [1.0]]
        """
        return self.poly.project(dims, tol=tol, max_dim=max_dim)

    def vertices(
        self,
        *,
        tol: float = DEFAULT_GEOMETRY_TOL,
        max_players: int = DEFAULT_GEOMETRY_MAX_PLAYERS,
        max_dim: int = DEFAULT_GEOMETRY_MAX_DIM,
    ) -> list[list[float]]:
        # Use the epsilon-core vertex enumerator (brute-force, small n) for consistency.
        """
        Enumerate vertices of the least-core polytope (small dimension).

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import LeastCore
        >>> g = Game.from_coalitions(n_players=2, values={0: 0, 3: 1})
        >>> LeastCore(g).vertices(max_dim=2)  # doctest: +SKIP
        [[0.0, 1.0], [1.0, 0.0]]
        """
        return self.epsilon_core.vertices(tol=tol, max_players=max_players, max_dim=max_dim)


def least_core(game: GameProtocol, *, tol: float = 1e-9) -> LeastCoreResult:
    """
    Compute a least-core allocation and the least-core epsilon.

    Notes
    -----
    Requires an LP backend at runtime (`pip install "tucoopy[lp]"`).
    """
    from ..backends.optional_deps import require_module

    np = require_module("numpy", extra="lp", context="least_core")  # type: ignore

    n = game.n_players
    grand = game.grand_coalition
    vN = float(game.value(grand))

    # Variables: [x0..x_{n-1}, eps]
    c = np.zeros(n + 1, dtype=float)
    c[n] = 1.0

    A_eq = np.zeros((1, n + 1), dtype=float)
    A_eq[0, :n] = 1.0
    b_eq = np.array([vN], dtype=float)

    rows: list[list[float]] = []
    rhs: list[float] = []
    for S in all_coalitions(n):
        if S == 0 or S == grand:
            continue
        row = [0.0] * (n + 1)
        for i in range(n):
            if S & (1 << i):
                row[i] = -1.0
        row[n] = -1.0  # -eps
        rows.append(row)
        rhs.append(-float(game.value(S)))

    A_ub = np.asarray(rows, dtype=float) if rows else None
    b_ub = np.asarray(rhs, dtype=float) if rhs else None

    bounds = [(None, None)] * n + [(None, None)]

    from ..backends.lp import linprog_solve

    res = linprog_solve(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        context="least_core",
    )

    z = res.x.tolist()
    x = [float(v) for v in z[:n]]
    eps = float(z[n])
    if abs(eps) <= float(tol):
        eps = 0.0

    tight: list[int] = []
    duals: dict[int, float] = {}
    if A_ub is not None and b_ub is not None:
        z_np = np.asarray(z, dtype=float)
        slack = b_ub - (A_ub @ z_np)
        for idx, S in enumerate([S for S in all_coalitions(n) if S not in (0, grand)]):
            if idx >= slack.shape[0]:
                break
            if float(slack[idx]) <= float(tol):
                tight.append(S)
        tight.sort()

        marg = getattr(getattr(res, "ineqlin", None), "marginals", None)
        if marg is not None:
            for idx, S in enumerate([S for S in all_coalitions(n) if S not in (0, grand)]):
                if idx >= len(marg):
                    break
                w = float(marg[idx])
                if abs(w) > float(tol):
                    duals[S] = w

    diag = None
    try:
        from ..diagnostics.linprog_diagnostics import linprog_diagnostics

        diag = linprog_diagnostics(res)
    except Exception:
        diag = None

    return LeastCoreResult(x=x, epsilon=eps, tight=tight or None, dual_weights=duals or None, lp=diag)


def least_core_epsilon_star(game: GameProtocol, *, tol: float = 1e-9) -> float:
    """
    Compute the least-core value epsilon*.
    """

    return float(least_core(game, tol=float(tol)).epsilon)


def least_core_point(
    game: GameProtocol,
    *,
    restrict_to_imputation: bool = False,
    tol: float = 1e-9,
    method: SelectionMethod = "chebyshev_center",
) -> LeastCorePointResult:
    """
    Select a single allocation from the least-core set.
    """

    lc = LeastCore(game, restrict_to_imputation=restrict_to_imputation, tol=float(tol))

    if method == "chebyshev_center":
        cc = lc.chebyshev_center()
        if cc is None:
            raise InvalidGameError("least_core_point: least-core set is empty")
        x_cc, _r = cc
        return LeastCorePointResult(x=[float(v) for v in x_cc], epsilon=float(lc.epsilon), method=str(method))

    if method == "any_feasible":
        x_any = lc.sample_point()
        if x_any is None:
            raise InvalidGameError("least_core_point: least-core set is empty")
        return LeastCorePointResult(x=[float(v) for v in x_any], epsilon=float(lc.epsilon), method=str(method))

    raise InvalidParameterError("method must be 'chebyshev_center' or 'any_feasible'")


__all__ = [
    "LeastCore",
    "LeastCoreResult",
    "LeastCorePointResult",
    "least_core",
    "least_core_epsilon_star",
    "least_core_point",
]
