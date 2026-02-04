"""
# Kernel and pre-kernel (iterative methods).

This module implements pre-kernel and kernel iterations based on surplus equalization,
with optional approximations for larger games.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..base.config import (
    DEFAULT_KERNEL_APPROX_MAX_COALITIONS_PER_PAIR,
    DEFAULT_KERNEL_APPROX_SEED,
    DEFAULT_KERNEL_MAX_ITER,
    DEFAULT_KERNEL_TOL,
)
from ..base.types import GameProtocol
from ..base.exceptions import InvalidGameError, InvalidParameterError
from ..backends.numpy_fast import require_numpy
from ..geometry.imputation_set import imputation_lower_bounds, project_to_imputation
from ._surplus import SurplusEvaluator, _all_values_list, _coalition_sums_dp


def _indicator(mask: int, n: int) -> list[float]:
    row = [0.0] * n
    for i in range(n):
        if mask & (1 << i):
            row[i] = 1.0
    return row


@dataclass(frozen=True)
class PreKernelResult:
    """
    Result container for the pre-kernel iteration.

    Attributes
    ----------
    x : list[float]
        Candidate pre-kernel allocation (length `n_players`).
    iterations : int
        Number of outer iterations performed.
    residual : float
        Final maximum surplus imbalance:

        $$
        \\max_{i<j} |s_{ij}(x) - s_{ji}(x)|.
        $$

    delta : float
        Maximum absolute coordinate update in the last iteration.
    argmax : dict[tuple[int, int], int]
        Dictionary mapping ordered pairs (i, j) to an argmax coalition mask
        achieving $s_{ij}(x)$ at the returned point.
    """
    x: list[float]
    iterations: int
    residual: float
    delta: float
    # Coalition argmax selections used at the fixed point: (i,j) -> mask
    argmax: dict[tuple[int, int], int]


@dataclass(frozen=True)
class KernelResult:
    """
    Result container for the kernel iteration (imputation-constrained).

    Attributes
    ----------
    x : list[float]
        Candidate kernel allocation (length `n_players`), intended to lie in the
        imputation set (efficiency + individual rationality).
    iterations : int
        Number of outer iterations performed.
    residual : float
        A measure of kernel complementarity violation used as stopping criterion.
    delta : float
        Maximum absolute coordinate update in the last iteration.
    argmax : dict[tuple[int, int], int]
        Dictionary mapping ordered pairs (i, j) to an argmax coalition mask
        achieving $s_{ij}(x)$ at the returned point.
    active_bounds : set[int]
        Active-set of players clamped at individual rationality bounds
        ($x_i = v(\\{i\\})$) at the returned point.
    """
    x: list[float]
    iterations: int
    residual: float
    delta: float
    argmax: dict[tuple[int, int], int]
    active_bounds: set[int]


def prekernel(
    game: GameProtocol,
    *,
    x0: list[float] | None = None,
    tol: float = DEFAULT_KERNEL_TOL,
    max_iter: int = DEFAULT_KERNEL_MAX_ITER,
    relax: float = 1.0,
    approx_max_coalitions_per_pair: int | None = DEFAULT_KERNEL_APPROX_MAX_COALITIONS_PER_PAIR,
    approx_seed: int | None = DEFAULT_KERNEL_APPROX_SEED,
) -> PreKernelResult:
    """
    Compute a (candidate) **pre-kernel** element.

    Background
    ----------
    For an allocation $x \\in \\mathbb{R}^n$, define the pairwise surplus:

    $$
    s_{ij}(x) = \\max_{S: i \\in S,\\ j \\notin S} \\left[v(S) - x(S)\\right].
    $$

    The **pre-kernel** is the set of allocations $x$ such that

    $$
    s_{ij}(x) = s_{ji}(x) \\quad \\text{for all } i \\ne j.
    $$

    Method
    ------
    This routine implements a practical fixed-point style iteration:

    1. Given the current allocation $x$, compute argmax coalitions $S_{ij}$ for each
       ordered pair $(i,j)$ (so that $S_{ij}$ achieves $s_{ij}(x)$).
    2. Freeze these argmax coalitions and enforce the equalities

    $$
    s_{ij}(x) = s_{ji}(x)
    \\iff
    x(S_{ji}) - x(S_{ij}) = v(S_{ji}) - v(S_{ij})
    $$

       for all $i<j$.
      
    3. Solve the resulting (typically overdetermined) linear system in a least-squares
       sense, adding the efficiency constraint $\\sum_i x_i = v(N)$.
    4. Optionally apply a damped update controlled by ``relax``.
    5. Repeat until argmax selections stabilize and the maximum surplus imbalance is
       below tolerance.

    Parameters
    ----------
    game
        TU game.
    x0
        Optional initial allocation. If omitted, the Shapley value is used as a
        starting point.
    tol
        Stopping tolerance for both surplus imbalance and coordinate changes.
    max_iter
        Maximum number of iterations.
    relax
        Relaxation parameter in (0, 1]. Values below 1 damp updates and can improve
        stability in degenerate cases.
    approx_max_coalitions_per_pair
        If provided, approximate each surplus ``s_ij(x)`` by evaluating at most this
        many candidate coalitions for each ordered pair (i, j), sampled uniformly
        without replacement from the admissible set ``{S: i in S, j not in S}``.
        This reduces per-iteration cost substantially for n > ~10.
    approx_seed
        Random seed used when ``approx_max_coalitions_per_pair`` is set.

    Returns
    -------
    PreKernelResult
        Candidate pre-kernel point and iteration diagnostics.

    Notes
    -----
    - This is a **numerical heuristic** intended for small games and visualization.
      Convergence is not guaranteed for all games.
    - Requires NumPy (`pip install "tucoop[fast]"`) for least-squares solves.

    Examples
    --------
    >>> from tucoop import Game
    >>> from tucoop.solutions.kernel import prekernel
    >>>
    >>> # Additive game: v(S) = |S|
    >>> g = Game.from_value_function(
    ...     n_players=3,
    ...     value_fn=lambda ps: float(len(ps)),
    ... )
    >>> res = prekernel(g, tol=1e-10, max_iter=200)
    >>> [round(v, 8) for v in res.x]
    [1.0, 1.0, 1.0]
    >>> res.residual <= 1e-8
    True
    """
    np = require_numpy(context="prekernel")

    n = game.n_players
    vN = float(game.value(game.grand_coalition))
    evaluator = SurplusEvaluator.for_n_players(n)
    values = None if approx_max_coalitions_per_pair is not None else _all_values_list(game)
    rng = None
    if approx_max_coalitions_per_pair is not None:
        from random import Random

        rng = Random(approx_seed)

    if x0 is None:
        # Start from Shapley if available without circular imports.
        from .shapley import shapley_value

        x = np.asarray(shapley_value(game), dtype=float)
    else:
        if len(x0) != n:
            raise InvalidParameterError("x0 must have length n_players")
        x = np.asarray([float(v) for v in x0], dtype=float)

    last_argmax: dict[tuple[int, int], int] = {}
    imbalance = float("inf")

    for it in range(1, max_iter + 1):
        argmax: dict[tuple[int, int], int] = {}
        imbalance = 0.0
        x_sums = _coalition_sums_dp(x.tolist(), n_players=n)
        surpluses: list[list[float]] = [[0.0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                s, S = evaluator.surplus_and_argmax(
                    game,
                    x.tolist(),
                    i,
                    j,
                    values=values,
                    x_sums=x_sums,
                    approx_max_coalitions=approx_max_coalitions_per_pair,
                    rng=rng,
                )
                argmax[(i, j)] = S
                surpluses[i][j] = float(s)

        # Compute maximum surplus imbalance.
        for i in range(n):
            for j in range(i + 1, n):
                imbalance = max(
                    imbalance, abs(float(surpluses[i][j]) - float(surpluses[j][i]))
                )

        # Build least squares system: for each i<j,
        #   x(S_ji) - x(S_ij) = v(S_ji) - v(S_ij)
        rows = []
        rhs = []
        for i in range(n):
            for j in range(i + 1, n):
                Sij = argmax[(i, j)]
                Sji = argmax[(j, i)]
                a = np.asarray(_indicator(Sji, n), dtype=float) - np.asarray(_indicator(Sij, n), dtype=float)
                b = float(game.value(Sji) - game.value(Sij))
                rows.append(a)
                rhs.append(b)

        # Efficiency constraint.
        rows.append(np.ones(n, dtype=float))
        rhs.append(vN)

        A = np.vstack(rows)
        b = np.asarray(rhs, dtype=float)

        # Solve min ||Ax-b||_2
        x_new, *_ = np.linalg.lstsq(A, b, rcond=None)
        # Damped update to improve stability in degenerate cases.
        if relax <= 0.0 or relax > 1.0:
            raise InvalidParameterError("relax must be in (0,1]")
        x_upd = (1.0 - relax) * x + relax * x_new
        delta = float(np.max(np.abs(x_upd - x)))

        # Stop if selections stable and imbalance small.
        if argmax == last_argmax and imbalance <= tol and delta <= tol:
            return PreKernelResult(
                x=[float(v) for v in x_upd.tolist()],
                iterations=it,
                residual=float(imbalance),
                delta=delta,
                argmax=argmax,
            )

        last_argmax = argmax
        x = x_upd

    # Return best effort.
    return PreKernelResult(
        x=[float(v) for v in x.tolist()],
        iterations=max_iter,
        residual=float(imbalance),
        delta=float(0.0),
        argmax=last_argmax,
    )


def kernel(
    game: GameProtocol,
    *,
    x0: list[float] | None = None,
    tol: float = DEFAULT_KERNEL_TOL,
    max_iter: int = DEFAULT_KERNEL_MAX_ITER,
    approx_max_coalitions_per_pair: int | None = DEFAULT_KERNEL_APPROX_MAX_COALITIONS_PER_PAIR,
    approx_seed: int | None = DEFAULT_KERNEL_APPROX_SEED,
) -> KernelResult:
    """
    Compute a (candidate) **kernel** element.

    Background
    ----------
    The kernel refines the pre-kernel by imposing a complementarity condition
    with respect to the imputation set. Using the same surplus definition
    $s_{ij}(x)$ as in the pre-kernel, the kernel condition can be stated as:

    - For each pair $(i, j)$, if player $i$ has strictly larger surplus against $j$,
      i.e. $s_{ij}(x) > s_{ji}(x)$, then player $j$ must be at its individual
      rationality bound: $x_j = v(\\{j\\})$.

    This prevents a player with "weaker bargaining position" from receiving more
    than their minimal guaranteed payoff.

    Method (active-set heuristic)
    -----------------------------
    This implementation is a practical small-n solver:

    1. Maintain $x$ in the imputation set (efficiency + individual rationality),
       using projection.
    2. Maintain an active-set ``active_bounds`` of players clamped at their
       individual rationality bounds.
    3. From the current $x$, derive which bounds must be active based on surplus
       dominance relations.
    4. With a fixed active-set, solve pre-kernel equalities (in least-squares form)
       for the remaining free players, then project back to the imputation set.
    5. Repeat until:
       - argmax selections stabilize,
       - the active-set stabilizes,
       - kernel violations are below tolerance.

    Parameters
    ----------
    game
        TU game.
    x0
        Optional initial allocation. If omitted, the Shapley value is used as a
        starting point, then projected into the imputation set.
    tol
        Tolerance used for dominance comparisons and stopping conditions.
    max_iter
        Maximum number of iterations.
    approx_max_coalitions_per_pair
        If provided, approximate each surplus ``s_ij(x)`` by evaluating at most this
        many candidate coalitions for each ordered pair (i, j).
    approx_seed
        Random seed used when ``approx_max_coalitions_per_pair`` is set.

    Returns
    -------
    KernelResult
        Candidate kernel point and iteration diagnostics.

    Raises
    ------
    InvalidGameError
        If the imputation set is empty (i.e. $\\sum_i v(\\{i\\}) > v(N)$).

    Notes
    -----
    - This is a **heuristic** implementation intended primarily for small games
      (e.g. $n \\leq 4$) and visualization use-cases.
    - Requires NumPy (`pip install "tucoop[fast]"`) for least-squares solves.

    Examples
    --------
    >>> from tucoop import Game
    >>> from tucoop.solutions.kernel import kernel
    >>>
    >>> # Additive game: kernel is the equal split
    >>> g = Game.from_value_function(
    ...     n_players=3,
    ...     value_fn=lambda ps: float(len(ps)),
    ... )
    >>> res = kernel(g, tol=1e-10, max_iter=200)
    >>> [round(v, 8) for v in res.x]
    [1.0, 1.0, 1.0]
    >>> res.residual <= 1e-8
    True
    >>>
    >>> # If the imputation set is empty, kernel() raises InvalidGameError:
    >>> bad = Game.from_coalitions(
    ...     n_players=2,
    ...     values={
    ...         (): 0.0,
    ...         (0,): 2.0,
    ...         (1,): 2.0,
    ...         (0, 1): 1.0,  # v(N) < v({0}) + v({1})
    ...     },
    ... )
    >>> kernel(bad)
    Traceback (most recent call last):
        ...
    InvalidGameError: kernel undefined: imputation set is empty (sum v({i}) > v(N))
    """
    np = require_numpy(context="kernel")

    n = game.n_players
    vN = float(game.value(game.grand_coalition))
    l = imputation_lower_bounds(game)
    evaluator = SurplusEvaluator.for_n_players(n)
    values = None if approx_max_coalitions_per_pair is not None else _all_values_list(game)
    rng = None
    if approx_max_coalitions_per_pair is not None:
        from random import Random

        rng = Random(approx_seed)

    if x0 is None:
        from .shapley import shapley_value

        x = np.asarray(shapley_value(game), dtype=float)
    else:
        if len(x0) != n:
            raise InvalidParameterError("x0 must have length n_players")
        x = np.asarray([float(v) for v in x0], dtype=float)

    proj = project_to_imputation(game, x.tolist())
    if not proj.feasible:
        raise InvalidGameError("kernel undefined: imputation set is empty (sum v({i}) > v(N))")
    x = np.asarray(proj.x, dtype=float)

    # Active-set of players fixed at individual rationality bounds.
    active_bounds: set[int] = set(i for i in range(n) if x[i] <= l[i] + tol)
    last_argmax: dict[tuple[int, int], int] = {}

    def kernel_violation(xv: list[float]) -> float:
        # s_ij > s_ji implies x_j == v({j}) in the kernel.
        viol = 0.0
        x_sums = _coalition_sums_dp(xv, n_players=n)
        for i in range(n):
            for j in range(i + 1, n):
                sij, _ = evaluator.surplus_and_argmax(
                    game,
                    xv,
                    i,
                    j,
                    values=values,
                    x_sums=x_sums,
                    approx_max_coalitions=approx_max_coalitions_per_pair,
                    rng=rng,
                )
                sji, _ = evaluator.surplus_and_argmax(
                    game,
                    xv,
                    j,
                    i,
                    values=values,
                    x_sums=x_sums,
                    approx_max_coalitions=approx_max_coalitions_per_pair,
                    rng=rng,
                )
                if sij > sji + tol and xv[j] > l[j] + tol:
                    viol = max(viol, sij - sji)
                if sji > sij + tol and xv[i] > l[i] + tol:
                    viol = max(viol, sji - sij)
        return float(viol)

    def compute_required_bounds(xv: list[float]) -> set[int]:
        req: set[int] = set(i for i in range(n) if xv[i] <= l[i] + tol)
        x_sums = _coalition_sums_dp(xv, n_players=n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                sij, _ = evaluator.surplus_and_argmax(
                    game,
                    xv,
                    i,
                    j,
                    values=values,
                    x_sums=x_sums,
                    approx_max_coalitions=approx_max_coalitions_per_pair,
                    rng=rng,
                )
                sji, _ = evaluator.surplus_and_argmax(
                    game,
                    xv,
                    j,
                    i,
                    values=values,
                    x_sums=x_sums,
                    approx_max_coalitions=approx_max_coalitions_per_pair,
                    rng=rng,
                )
                if sij > sji + tol:
                    req.add(j)
        return req

    for it in range(1, max_iter + 1):
        argmax: dict[tuple[int, int], int] = {}
        x_sums = _coalition_sums_dp(x.tolist(), n_players=n)
        surpluses: list[list[float]] = [[0.0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                s, S = evaluator.surplus_and_argmax(
                    game,
                    x.tolist(),
                    i,
                    j,
                    values=values,
                    x_sums=x_sums,
                    approx_max_coalitions=approx_max_coalitions_per_pair,
                    rng=rng,
                )
                argmax[(i, j)] = S
                surpluses[i][j] = float(s)

        # Active-set update (allow add/remove) based on current kernel dominance relations.
        req = compute_required_bounds([float(v) for v in x.tolist()])
        if req != active_bounds:
            active_bounds = req
            # Clamp active players to bounds and reproject.
            x_list = [float(v) for v in x.tolist()]
            for j in active_bounds:
                x_list[j] = float(l[j])
            proj = project_to_imputation(game, x_list)
            x = np.asarray(proj.x, dtype=float)
            last_argmax = {}  # invalidate
            continue

        # Build least squares system for interior pairs where both players are not active.
        rows = []
        rhs = []
        for i in range(n):
            for j in range(i + 1, n):
                if i in active_bounds or j in active_bounds:
                    continue
                Sij = argmax[(i, j)]
                Sji = argmax[(j, i)]
                a = np.asarray(_indicator(Sji, n), dtype=float) - np.asarray(_indicator(Sij, n), dtype=float)
                b = float(game.value(Sji) - game.value(Sij))
                rows.append(a)
                rhs.append(b)

        # Fix active bound players as equalities x_j = l_j by adding rows.
        for j in sorted(active_bounds):
            row = np.zeros(n, dtype=float)
            row[j] = 1.0
            rows.append(row)
            rhs.append(float(l[j]))

        rows.append(np.ones(n, dtype=float))
        rhs.append(vN)

        if len(rows) == 1:
            # Only efficiency constraint => just keep the current imputation.
            viol = kernel_violation([float(v) for v in x.tolist()])
            if argmax == last_argmax and viol <= tol:
                return KernelResult(
                    x=[float(v) for v in x.tolist()],
                    iterations=it,
                    residual=float(viol),
                    delta=0.0,
                    argmax=argmax,
                    active_bounds=set(active_bounds),
                )
            last_argmax = argmax
            continue

        A = np.vstack(rows)
        bvec = np.asarray(rhs, dtype=float)
        x_hat, *_ = np.linalg.lstsq(A, bvec, rcond=None)

        proj = project_to_imputation(game, [float(v) for v in x_hat.tolist()])
        x_new = np.asarray(proj.x, dtype=float)

        viol = kernel_violation([float(v) for v in x_new.tolist()])

        # Stop if:
        # - argmax selections stable
        # - active set stable
        # - kernel violation small
        if argmax == last_argmax and viol <= tol and compute_required_bounds([float(v) for v in x_new.tolist()]) == active_bounds:
            return KernelResult(
                x=[float(v) for v in x_new.tolist()],
                iterations=it,
                residual=float(viol),
                delta=float(np.max(np.abs(x_new - x))),
                argmax=argmax,
                active_bounds=set(active_bounds),
            )

        last_argmax = argmax
        x = x_new

    return KernelResult(
        x=[float(v) for v in x.tolist()],
        iterations=max_iter,
        residual=float(kernel_violation([float(v) for v in x.tolist()])),
        delta=float(0.0),
        argmax=last_argmax,
        active_bounds=set(active_bounds),
    )
