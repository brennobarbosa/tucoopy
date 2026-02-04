"""
# Nucleolus and pre-nucleolus.

This module implements lexicographic LP refinement for the nucleolus (and pre-nucleolus).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from ..base.types import GameProtocol
from ..base.coalition import all_coalitions
from ..base.exceptions import ConvergenceError, InvalidGameError
from .tau import utopia_payoff

if TYPE_CHECKING:  # pragma: no cover
    from ..diagnostics.linprog_diagnostics import LinprogDiagnostics


def _default_excess(S: int, xS: float, game: GameProtocol) -> float:
    """
    Default coalition excess used in the nucleolus procedures.

    The coalition excess is defined as

    $$
    e(S, x) = v(S) - x(S),
    $$

    where:

    - $v(S)$ is the worth of coalition $S$,
    - $x(S) = \\sum_{i \\in S} x_i$ is the payoff allocated to $S$.

    This is the classical notion of excess used in the definitions of the
    pre-nucleolus and nucleolus.

    Parameters
    ----------
    S
        Coalition mask.
    xS
        Current allocation sum over the coalition.
    game
        TU game.

    Returns
    -------
    float
        The excess value $e(S, x)$.
    """
    return float(game.value(S) - xS)


@dataclass(frozen=True)
class NucleolusResult:
    """
    Result container for the nucleolus / pre-nucleolus computation.

    Attributes
    ----------
    x : list[float]
        Allocation vector (length `n_players`).
    levels : list[float]
        Sequence of epsilon levels fixed during the lexicographic minimization.
        Each entry corresponds to the maximum excess minimized at that round.
    tight_sets : list[list[int]] | None
        For each round, the list of coalitions that were tight (achieved the
        maximum excess) at that level.
    lp_rounds : list[LinprogDiagnostics] | None
        Optional diagnostics from each LP solve (HiGHS / SciPy).
    """

    x: list[float]
    # Sequence of epsilon levels fixed during the lexicographic minimization.
    levels: list[float]
    # Optional per-round diagnostics: tight coalitions achieving the max excess.
    tight_sets: list[list[int]] | None = None
    # Optional per-round solver diagnostics (SciPy/HiGHS).
    lp_rounds: list[LinprogDiagnostics] | None = None


def prenucleolus(
    game: GameProtocol,
    *,
    tol: float = 1e-9,
    max_rounds: int = 200,
    excess: Callable[[int, float, GameProtocol], float] | None = None,
) -> NucleolusResult:
    """
    Compute the **pre-nucleolus** of a TU cooperative game.

    The pre-nucleolus is defined as the allocation that lexicographically
    minimizes the vector of coalition excesses:

    $$
    e(S, x) = v(S) - x(S),
    $$

    ordered from largest to smallest.

    Generalized excess
    ------------------
    This implementation supports a **generalized notion of excess** through
    the ``excess`` parameter. Instead of the classical excess, one may supply
    a custom function

    ``excess(S, xS, game)``

    allowing the algorithm to minimize lexicographically any transformed
    excess such as:

    - per-capita excess: $\\fra{v(S)-x(S)}{|S|}$,
    - proportional excess: $\\frac{v(S)-x(S)}{v(S)}$,
    - disruption-based excess, etc.

    This turns the routine into a generic **lexicographic excess minimization
    engine**.

    Algorithm
    ---------
    The method follows the standard lexicographic LP refinement:

    1. Solve an LP minimizing the maximum excess over all coalitions.
    2. Identify coalitions that achieve this maximum excess (tight sets).
    3. Fix their excess as equality constraints.
    4. Repeat on the reduced problem.

    Parameters
    ----------
    game : GameProtocol
        TU game.
    tol : float, default=1e-9
        Numerical tolerance for tightness detection.
    max_rounds : int, default=200
        Maximum number of lexicographic refinement rounds.
    excess : callable, optional
        Custom excess function ``excess(S, xS, game)``.

    Returns
    -------
    NucleolusResult
        Allocation, epsilon levels, and optional diagnostics.

    Notes
    -----
    - Only **efficiency** is enforced (no individual rationality).
    - The result may lie outside the imputation set.
    - Requires SciPy/HiGHS at runtime (`pip install "tucoop[lp]"`).
    - The algorithm terminates when the solution is uniquely determined
      by the accumulated tight constraints.

    Examples
    --------
    >>> prenucleolus(g).x
    """
    from ..backends.optional_deps import require_module

    np = require_module("numpy", extra="lp", context="prenucleolus")  # type: ignore

    n = game.n_players
    excess_fn = excess or _default_excess
    grand = game.grand_coalition
    vN = float(game.value(grand))

    # Remaining coalitions to constrain (proper, non-empty).
    remaining: list[int] = []
    for S in all_coalitions(n):
        if S == 0 or S == grand:
            continue
        remaining.append(S)

    # Fixed equalities: coalition mask -> fixed excess value (epsilon level).
    fixed: dict[int, float] = {}
    levels: list[float] = []
    tight_sets: list[list[int]] = []
    lp_rounds: list[LinprogDiagnostics] = []

    # Base equality: efficiency.
    A_eq_base = np.zeros((1, n), dtype=float)
    A_eq_base[0, :] = 1.0
    b_eq_base = np.array([vN], dtype=float)

    def mask_row(S: int) -> Any:
        row = np.zeros(n, dtype=float)
        for i in range(n):
            if S & (1 << i):
                row[i] = 1.0
        return row

    def current_rank(A_eq: Any) -> int:
        return int(np.linalg.matrix_rank(A_eq, tol=1e-10))

    for _round in range(max_rounds):
        # Variables: [x0..x_{n-1}, t]
        c = np.zeros(n + 1, dtype=float)
        c[n] = 1.0

        # Equalities: efficiency + fixed coalitions (excess fixed to their levels).
        Aeq_rows = [A_eq_base[0, :].tolist()]
        beq_vals = [vN]
        for S, eps_S in fixed.items():
            Aeq_rows.append(mask_row(S).tolist())
            beq_vals.append(float(game.value(S)) - float(eps_S))

        A_x = np.asarray(Aeq_rows, dtype=float)
        A_eq = np.hstack([A_x, np.zeros((A_x.shape[0], 1), dtype=float)])
        b_eq = np.asarray(beq_vals, dtype=float)

        # Inequalities: for remaining coalitions, v(S) - x(S) <= t  <=>  -x(S) - t <= -v(S)
        Aub_rows: list[list[float]] = []
        bub_vals: list[float] = []
        for S in remaining:
            row = [0.0] * (n + 1)
            for i in range(n):
                if S & (1 << i):
                    row[i] = -1.0
            row[n] = -1.0
            Aub_rows.append(row)
            bub_vals.append(-float(game.value(S)))

        A_ub = np.asarray(Aub_rows, dtype=float) if Aub_rows else None
        b_ub = np.asarray(bub_vals, dtype=float) if bub_vals else None

        # Pre-nucleolus only enforces efficiency; x is otherwise unrestricted.
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
            context="prenucleolus",
        )

        z = res.x.tolist()
        x = np.asarray(z[:n], dtype=float)
        t = float(z[n])
        if abs(t) <= tol:
            t = 0.0

        # Compute excesses for remaining coalitions and find tight ones.
        tight: list[int] = []
        for S in remaining:
            xs = float((mask_row(S) @ x).item())
            exc = float(excess_fn(S, xs, game))
            if exc >= t - tol:
                tight.append(S)
        tight.sort()
        tight_sets.append(tight[:])

        # Fix t level for this round.
        levels.append(t)
        try:
            from ..diagnostics.linprog_diagnostics import linprog_diagnostics

            lp_rounds.append(linprog_diagnostics(res))
        except Exception:
            pass

        # Add tight coalitions as equalities only if they increase rank.
        base_rank = current_rank(A_eq[:, :n])
        added_any = False
        newly_fixed: list[int] = []

        for S in tight:
            if S in fixed:
                continue
            cand_A = np.vstack([A_eq[:, :n], mask_row(S)[None, :]])
            cand_rank = current_rank(cand_A)
            if cand_rank > base_rank:
                fixed[S] = t
                newly_fixed.append(S)
                base_rank = cand_rank
                added_any = True
                if base_rank >= n:
                    break

        # Remove fixed coalitions from remaining constraints.
        if newly_fixed:
            fixed_set = set(newly_fixed)
            remaining = [S for S in remaining if S not in fixed_set]

        # Stop if x is uniquely determined (rank n) or no constraints left.
        if base_rank >= n or not remaining:
            return NucleolusResult(
                x=[float(v) for v in x.tolist()],
                levels=levels,
                tight_sets=tight_sets,
                lp_rounds=lp_rounds or None,
            )

        if not added_any:
            # Degeneracy: we did not find an independent tight coalition.
            # As a fallback, fix all tight coalitions at this level (may be redundant),
            # then continue; HiGHS can handle redundant equalities.
            for S in tight:
                fixed.setdefault(S, t)
            remaining = [S for S in remaining if S not in fixed]

    raise ConvergenceError("prenucleolus did not converge: max_rounds exceeded")


def nucleolus(
    game: GameProtocol,
    *,
    tol: float = 1e-9,
    max_rounds: int = 200,
    excess: Callable[[int, float, GameProtocol], float] | None = None,
) -> NucleolusResult:
    """
    Compute the **nucleolus** of a TU cooperative game.

    The nucleolus is the allocation in the **imputation set** that
    lexicographically minimizes coalition excesses:

    $$
    e(S, x) = v(S) - x(S),
    $$

    from largest to smallest.

    Generalized excess
    ------------------
    As in ``prenucleolus``, this implementation supports a **generalized
    excess function** through the ``excess`` parameter, allowing the computation
    of several nucleolus variants (per-capita, proportional, disruption-based, etc.)
    without modifying the core algorithm.

    Individual rationality
    ----------------------
    In addition to efficiency, this version enforces

    $$
    x_i \\ge v(\\{i\\}) \\quad \\text{for all players } i,
    $$

    restricting the solution to the imputation set.

    Parameters
    ----------
    game : GameProtocol
        TU game.
    tol : float, default=1e-9
        Numerical tolerance for tightness detection.
    max_rounds : int, default=200
        Maximum number of lexicographic refinement rounds.
    excess : callable, optional
        Custom excess function ``excess(S, xS, game)``.

    Returns
    -------
    NucleolusResult
        Allocation, epsilon levels, and optional diagnostics.

    Raises
    ------
    InvalidGameError
        If the imputation set is empty, i.e.
        $\\sum_i v(\\{i\\}) > v(N)$.

    Notes
    -----
    - The nucleolus always lies in the imputation set.
    - It is one of the most important solution concepts in cooperative
      game theory due to its strong stability properties.
    - Requires SciPy/HiGHS at runtime (`pip install "tucoop[lp]"`).
    - The algorithm terminates when enough tight coalitions determine
      the allocation uniquely.

    Examples
    --------
    >>> nucleolus(g).x
    """
    from ..backends.optional_deps import require_module

    np = require_module("numpy", extra="lp", context="nucleolus")  # type: ignore

    n = game.n_players
    excess_fn = excess or _default_excess
    grand = game.grand_coalition
    vN = float(game.value(grand))

    # Imputation feasibility check.
    lb = [float(game.value(1 << i)) for i in range(n)]
    if sum(lb) > vN + tol:
        raise InvalidGameError("nucleolus undefined: imputation set is empty (sum v({i}) > v(N))")

    remaining: list[int] = []
    for S in all_coalitions(n):
        if S == 0 or S == grand:
            continue
        remaining.append(S)

    fixed: dict[int, float] = {}
    levels: list[float] = []
    tight_sets: list[list[int]] = []
    lp_rounds: list[LinprogDiagnostics] = []

    A_eq_base = np.zeros((1, n), dtype=float)
    A_eq_base[0, :] = 1.0
    b_eq_base = np.array([vN], dtype=float)

    def mask_row(S: int) -> Any:
        row = np.zeros(n, dtype=float)
        for i in range(n):
            if S & (1 << i):
                row[i] = 1.0
        return row

    def current_rank(A_eq: Any) -> int:
        return int(np.linalg.matrix_rank(A_eq, tol=1e-10))

    for _round in range(max_rounds):
        # Variables: [x0..x_{n-1}, t]
        c = np.zeros(n + 1, dtype=float)
        c[n] = 1.0

        Aeq_rows = [A_eq_base[0, :].tolist()]
        beq_vals = [vN]
        for S, eps_S in fixed.items():
            Aeq_rows.append(mask_row(S).tolist())
            beq_vals.append(float(game.value(S)) - float(eps_S))

        A_x = np.asarray(Aeq_rows, dtype=float)
        A_eq = np.hstack([A_x, np.zeros((A_x.shape[0], 1), dtype=float)])
        b_eq = np.asarray(beq_vals, dtype=float)

        Aub_rows: list[list[float]] = []
        bub_vals: list[float] = []
        for S in remaining:
            row = [0.0] * (n + 1)
            for i in range(n):
                if S & (1 << i):
                    row[i] = -1.0
            row[n] = -1.0
            Aub_rows.append(row)
            bub_vals.append(-float(game.value(S)))

        A_ub = np.asarray(Aub_rows, dtype=float) if Aub_rows else None
        b_ub = np.asarray(bub_vals, dtype=float) if bub_vals else None

        # Nucleolus is computed over the imputation set.
        bounds = [(lb[i], None) for i in range(n)] + [(None, None)]

        from ..backends.lp import linprog_solve

        res = linprog_solve(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            context="nucleolus",
        )

        z = res.x.tolist()
        x = np.asarray(z[:n], dtype=float)
        t = float(z[n])
        if abs(t) <= tol:
            t = 0.0

        tight: list[int] = []
        for S in remaining:
            xs = float((mask_row(S) @ x).item())
            exc = float(excess_fn(S, xs, game))
            if exc >= t - tol:
                tight.append(S)
        tight.sort()
        tight_sets.append(tight[:])

        levels.append(t)
        try:
            from ..diagnostics.linprog_diagnostics import linprog_diagnostics

            lp_rounds.append(linprog_diagnostics(res))
        except Exception:
            pass

        base_rank = current_rank(A_eq[:, :n])
        newly_fixed: list[int] = []
        for S in tight:
            if S in fixed:
                continue
            cand_A = np.vstack([A_eq[:, :n], mask_row(S)[None, :]])
            cand_rank = current_rank(cand_A)
            if cand_rank > base_rank:
                fixed[S] = t
                newly_fixed.append(S)
                base_rank = cand_rank
                if base_rank >= n:
                    break

        if newly_fixed:
            fixed_set = set(newly_fixed)
            remaining = [S for S in remaining if S not in fixed_set]

        if base_rank >= n or not remaining:
            return NucleolusResult(
                x=[float(v) for v in x.tolist()],
                levels=levels,
                tight_sets=tight_sets,
                lp_rounds=lp_rounds or None,
            )

        if not newly_fixed:
            for S in tight:
                fixed.setdefault(S, t)
            remaining = [S for S in remaining if S not in fixed]

    raise ConvergenceError("nucleolus did not converge: max_rounds exceeded")


def per_capita_nucleolus(game: GameProtocol) -> NucleolusResult:
    """
    Per-capita nucleolus.

    Minimizes the lexicographic vector of

    $$e_{pc}(S, x) = \\frac{v(S) - x(S)}{|S|}.$$

    """

    def excess(S: int, xS: float, game: GameProtocol) -> float:
        k = int(S).bit_count()
        return (game.value(S) - xS) / float(k)

    return prenucleolus(game, excess=excess)


def proportional_nucleolus(game: GameProtocol) -> NucleolusResult:
    """
    Proportional nucleolus.

    This solution minimizes the lexicographic vector of
    **relative (proportional) excesses**:

    $$
    e_{prop}(S, x) = \\frac{v(S) - x(S)}{v(S)}.
    $$

    Coalitions are evaluated by how large their dissatisfaction is
    *relative to their own worth*.

    Notes
    -----
    - Coalitions with larger worth are normalized accordingly.
    - Common in bankruptcy and allocation problems where proportional
      dissatisfaction is more meaningful than absolute dissatisfaction.
    - Coalitions with $v(S)=0$ are ignored (excess defined as 0).

    Returns
    -------
    NucleolusResult
        Allocation and diagnostic information.
    """
    def excess(S: int, xS: float, game: GameProtocol) -> float:
        vS = game.value(S)
        if vS == 0.0:
            return 0.0
        return (vS - xS) / vS

    return prenucleolus(game, excess=excess)


def disruption_nucleolus(game: GameProtocol) -> NucleolusResult:
    """
    Disruption nucleolus.

    This variant minimizes the lexicographic vector of **disruption ratios**:

    $$
    e_{dis}(S, x) = \\frac{v(S) - x(S)}{M(S)},
    $$

    where $M$ is the utopia payoff vector and

    $$
    M(S) = \\sum_{i \\in S} M_i.
    $$

    Interpretation
    --------------
    The excess of a coalition is scaled by its *utopia potential*.
    Coalitions that could potentially demand more (high utopia value)
    are weighted accordingly.

    Notes
    -----
    - Closely related to the concept of *propensity to disrupt*.
    - Used in bargaining and stability analyses.
    - Reduces to the standard nucleolus when utopia payoffs are uniform.

    Returns
    -------
    NucleolusResult
        Allocation and diagnostic information.
    """
    M = utopia_payoff(game)

    def excess(S: int, xS: float, game: GameProtocol) -> float:
        MS = 0.0
        for i in range(game.n_players):
            if S & (1 << i):
                MS += M[i]
        if MS == 0.0:
            return 0.0
        return (game.value(S) - xS) / MS

    return prenucleolus(game, excess=excess)


__all__ = [
    "NucleolusResult",
    "prenucleolus",
    "nucleolus",
    "per_capita_nucleolus",
    "proportional_nucleolus",
    "disruption_nucleolus",
]
