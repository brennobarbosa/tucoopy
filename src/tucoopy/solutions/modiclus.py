"""
# Modiclus.

The modiclus is a nucleolus-type solution concept defined by lexicographic minimization of pairwise excess differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..base.types import GameProtocol, require_tabular_game
from ..base.coalition import coalition_sums
from ..base.exceptions import ConvergenceError, InvalidGameError, NotSupportedError

if TYPE_CHECKING:  # pragma: no cover
    from ..diagnostics.linprog_diagnostics import LinprogDiagnostics


@dataclass(frozen=True)
class ModiclusResult:
    """
    Result container for the modiclus computation.

    Attributes
    ----------
    x : list[float]
        Allocation vector (length `n_players`).
    levels : list[float]
        Sequence of lexicographic levels fixed during refinement. Each level is
        the optimal value of the auxiliary variable ``t`` in that round, i.e.
        the minimized maximum envy.
    tight_pairs : list[list[tuple[int, int]]] | None
        For each round, the list of ordered coalition pairs (S, T) that were
        tight at the optimum, meaning they achieved the maximum envy level
        (within tolerance).
    lp_rounds : list[LinprogDiagnostics] | None
        Optional solver diagnostics captured after each LP solve.

    Notes
    -----
    The modiclus is a nucleolus-type selection concept defined on **pairwise
    differences of excesses** (often interpreted as "maximum envy" across
    coalitions). This implementation mirrors the standard lexicographic LP
    refinement approach used for the nucleolus, but the constraints are indexed
    by ordered pairs (S, T) rather than coalitions S.
    """
    x: list[float]
    levels: list[float]
    tight_pairs: list[list[tuple[int, int]]] | None = None
    lp_rounds: list[LinprogDiagnostics] | None = None


def modiclus(
    game: GameProtocol,
    *,
    tol: float = 1e-9,
    max_rounds: int = 200,
    max_players: int = 12,
    require_complete: bool = True,
) -> ModiclusResult:
    """
    Compute the **modiclus** of a TU cooperative game.

    Definition
    ----------
    Let the excess of a coalition S under an allocation x be

    $$
    e(S, x) = v(S) - x(S),
    $$

    where $x(S) = \\sum_{i \\in S} x_i$.

    The modiclus is defined via the **envy** (difference of excesses) between two
    coalitions S and T:

    $$
    \\operatorname{envy}(S, T; x) = e(S, x) - e(T, x)
                                 = (v(S) - x(S)) - (v(T) - x(T)).
    $$

    The modiclus is the allocation x (typically over the **pre-imputation set**)
    that lexicographically minimizes the vector of envies over all ordered pairs
    $(S, T)$ with $S \\ne T$, sorted from largest to smallest.

    Algorithm
    ---------
    This implementation follows a nucleolus-style **lexicographic LP refinement**:

    1. Solve an LP minimizing the maximum envy over all remaining pairs (S, T).
    2. Identify *tight* pairs that achieve the maximum envy (within tolerance).
    3. Promote a subset of tight pairs to equality constraints (chosen so as to
       increase the rank of the equality system over x).
    4. Repeat until x is uniquely determined or no pairs remain.

    Parameters
    ----------
    game
        TU game.
    tol
        Numerical tolerance for tightness detection and for snapping tiny
        optimal levels to 0.
    max_rounds
        Maximum number of lexicographic refinement rounds.
    max_players
        Safety cap. The number of envy constraints scales as
        $O((2^n)^2)$, so the problem becomes large very quickly.
    require_complete
        If True, require a complete characteristic function (all ``2^n`` coalition
        values explicitly present in ``game.v``). If False, missing coalitions are
        treated as value 0 via ``game.value``.

    Returns
    -------
    ModiclusResult
        Allocation, lexicographic envy levels, and optional diagnostics.

    Notes
    -----
    - This implementation enforces **efficiency** only (pre-imputation set):
      $\\sum_i x_i = v(N)$. It does *not* enforce individual rationality.
    - The modiclus is sensitive to the set of coalitions considered; here we
      include all proper, non-empty coalitions and all ordered pairs (S, T),
      S != T.
    - Requires SciPy/HiGHS at runtime (`pip install "tucoopy[lp]"`).

    References
    ----------
    The modiclus is introduced in the TU cooperative game theory literature as a
    nucleolus-type solution defined on pairwise differences of excesses. For a
    detailed treatment, see standard references on nucleolus-like solution
    concepts and their variants.
    """
    from ..backends.optional_deps import require_module

    np = require_module("numpy", extra="lp", context="modiclus")  # type: ignore

    n = game.n_players
    if n > int(max_players):
        raise NotSupportedError(f"modiclus is exponential; requires n<={max_players} (got n={n})")

    grand = game.grand_coalition
    vN = float(game.value(grand))

    if require_complete:
        expected = 1 << n
        tabular = require_tabular_game(game, context="modiclus")
        if len(tabular.v) < expected or any(int(m) not in tabular.v for m in range(expected)):
            raise InvalidGameError("modiclus requires a complete characteristic function (2^n coalition values)")

    coalitions = [S for S in range(1 << n) if S not in (0, grand)]
    m = len(coalitions)
    if m <= 1:
        # n=1 has no proper coalitions; any efficient allocation is the modiclus.
        x = [vN] if n == 1 else [vN] + [0.0 for _ in range(n - 1)]
        return ModiclusResult(x=[float(v) for v in x], levels=[0.0])

    # Dense v (for speed).
    v = [0.0] * (1 << n)
    for S in range(1 << n):
        v[S] = float(game.value(S))

    # All ordered pairs (S,T), S!=T.
    remaining: list[tuple[int, int]] = []
    for S in coalitions:
        for T in coalitions:
            if S != T:
                remaining.append((int(S), int(T)))

    fixed: dict[tuple[int, int], float] = {}
    levels: list[float] = []
    tight_pairs: list[list[tuple[int, int]]] = []
    lp_rounds: list[LinprogDiagnostics] = []

    A_eq_base = np.zeros((1, n), dtype=float)
    A_eq_base[0, :] = 1.0
    #b_eq_base = np.array([vN], dtype=float) Not in use.

    def diff_row(S: int, T: int) -> Any:
        row = np.zeros(n, dtype=float)
        for i in range(n):
            if S & (1 << i):
                row[i] += 1.0
            if T & (1 << i):
                row[i] -= 1.0
        return row

    def current_rank(A_eq_x: Any) -> int:
        return int(np.linalg.matrix_rank(A_eq_x, tol=1e-10))

    for _round in range(int(max_rounds)):
        # Variables: [x0..x_{n-1}, t]
        c = np.zeros(n + 1, dtype=float)
        c[n] = 1.0

        # Equality constraints: efficiency + fixed envy equalities (over x only).
        Aeq_rows = [A_eq_base[0, :].tolist()]
        beq_vals = [vN]
        for (S, T), t_ST in fixed.items():
            Aeq_rows.append(diff_row(S, T).tolist())
            beq_vals.append(float(v[S] - v[T] - t_ST))

        A_x = np.asarray(Aeq_rows, dtype=float)
        A_eq = np.hstack([A_x, np.zeros((A_x.shape[0], 1), dtype=float)])
        b_eq = np.asarray(beq_vals, dtype=float)

        # Inequalities: envy(S,T;x) <= t  <=>  -x(S)+x(T)-t <= -(v(S)-v(T))
        Aub_rows: list[list[float]] = []
        bub_vals: list[float] = []
        for (S, T) in remaining:
            row = [0.0] * (n + 1)
            for i in range(n):
                if S & (1 << i):
                    row[i] = -1.0
                if T & (1 << i):
                    row[i] += 1.0
            row[n] = -1.0
            Aub_rows.append(row)
            bub_vals.append(-float(v[S] - v[T]))

        A_ub = np.asarray(Aub_rows, dtype=float) if Aub_rows else None
        b_ub = np.asarray(bub_vals, dtype=float) if bub_vals else None

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
            context="modiclus",
        )

        z = res.x.tolist()
        x = [float(vv) for vv in z[:n]]
        t = float(z[n])
        if abs(t) <= tol:
            t = 0.0

        # Compute all coalition sums once.
        x_sum = coalition_sums(x, n_players=n)

        # Tight pairs: envy(S,T;x) >= t - tol.
        tight: list[tuple[int, int]] = []
        for (S, T) in remaining:
            envy = float((v[S] - x_sum[S]) - (v[T] - x_sum[T]))
            if envy >= t - tol:
                tight.append((int(S), int(T)))
        tight.sort()
        tight_pairs.append(tight[:])

        levels.append(float(t))
        try:
            from ..diagnostics.linprog_diagnostics import linprog_diagnostics

            lp_rounds.append(linprog_diagnostics(res))
        except Exception:
            pass

        # Promote some tight constraints to equalities if they increase rank.
        base_rank = current_rank(A_eq[:, :n])
        newly_fixed: list[tuple[int, int]] = []
        for (S, T) in tight:
            if (S, T) in fixed:
                continue
            cand_A = np.vstack([A_eq[:, :n], diff_row(S, T)[None, :]])
            cand_rank = current_rank(cand_A)
            if cand_rank > base_rank:
                fixed[(S, T)] = t
                newly_fixed.append((S, T))
                base_rank = cand_rank
                if base_rank >= n:
                    break

        if newly_fixed:
            fixed_set = set(newly_fixed)
            remaining = [p for p in remaining if p not in fixed_set]

        if base_rank >= n or not remaining:
            return ModiclusResult(
                x=[float(vv) for vv in x],
                levels=levels,
                tight_pairs=tight_pairs,
                lp_rounds=lp_rounds or None,
            )

        if not newly_fixed:
            # Degenerate case: fix all tight at this level to make progress.
            for p in tight:
                fixed.setdefault(p, t)
            remaining = [p for p in remaining if p not in fixed]

    raise ConvergenceError("modiclus: reached max_rounds without convergence")


__all__ = ["ModiclusResult", "modiclus"]
