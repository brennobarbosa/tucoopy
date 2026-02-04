"""
# Kernel and pre-kernel (set-valued, sampling-based diagnostics).

This module provides `KernelSet` / `PreKernelSet` helpers aimed at
"small n" usage and diagnostics workflows.

The kernel and pre-kernel are defined via pairwise surplus comparisons and are
not polyhedral in general. The implementation therefore relies on:

- fast surplus evaluation helpers, and
- sampling points from the imputation set to probe membership / produce examples.

Notes
-----
For large games, kernel/pre-kernel computations can be expensive. This module
exposes explicit iteration limits and sampling limits to keep the runtime under
control.

Examples
--------
Instantiate the set-valued helpers for a small 3-player game:

>>> from tucoop import Game
>>> from tucoop.geometry.kernel_set import PreKernelSet, KernelSet
>>> g = Game.from_coalitions(n_players=3, values={
...     0: 0.0,
...     1: 1.0, 2: 1.0, 4: 1.0,
...     3: 2.0, 5: 2.0, 6: 2.0,
...     7: 4.0,
... })
>>> isinstance(PreKernelSet(g), PreKernelSet)
True
>>> isinstance(KernelSet(g), KernelSet)
True
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..base.config import (
    DEFAULT_KERNEL_APPROX_MAX_COALITIONS_PER_PAIR,
    DEFAULT_KERNEL_APPROX_SEED,
    DEFAULT_KERNEL_MAX_ITER,
    DEFAULT_KERNEL_MAX_PLAYERS,
    DEFAULT_KERNEL_TOL,
)
from ..base.types import GameProtocol
from ..base.exceptions import InvalidParameterError, NotSupportedError
from ..solutions._surplus import SurplusEvaluator, _all_values_list, _coalition_sums_dp
from .imputation_set import imputation_lower_bounds
from .sampling import sample_imputation_set
from ..diagnostics.allocation_diagnostics import is_efficient, is_imputation


_EVALUATOR_CACHE: dict[int, SurplusEvaluator] = {}


def _get_evaluator(n_players: int) -> SurplusEvaluator:
    n = int(n_players)
    ev = _EVALUATOR_CACHE.get(n)
    if ev is None:
        ev = SurplusEvaluator.for_n_players(n)
        _EVALUATOR_CACHE[n] = ev
    return ev


def _is_efficient(game: GameProtocol, x: list[float], *, tol: float) -> bool:
    return is_efficient(game, x, tol=tol)


def _is_imputation(game: GameProtocol, x: list[float], *, tol: float) -> bool:
    return is_imputation(game, x, tol=tol)


@dataclass(frozen=True)
class PairSurplusDiagnostics:
    """
    Diagnostics for a player pair (i, j) at an allocation x.

    Fields correspond to:

    - ``s_ij``: :math:`\\max_{S: i\\in S,\\ j\\notin S} e(S,x)`
    - ``s_ji``: :math:`\\max_{S: j\\in S,\\ i\\notin S} e(S,x)`
    - ``delta``: ``s_ij - s_ji``
    - ``argmax_ij`` / ``argmax_ji``: coalition masks achieving the maxima
      (tie-broken by smaller mask)

    Notes
    -----
    The argmax masks are useful to understand *which coalitions* witness
    the surplus imbalance for a given pair.

    Examples
    --------
    >>> from tucoop.geometry.kernel_set import PairSurplusDiagnostics
    >>> d = PairSurplusDiagnostics(i=0, j=1, s_ij=0.5, s_ji=0.25, delta=0.25, argmax_ij=3, argmax_ji=5)
    >>> d.delta
    0.25
    >>> isinstance(d.to_dict(), dict)
    True
    """

    i: int
    j: int
    s_ij: float
    s_ji: float
    delta: float
    argmax_ij: int
    argmax_ji: int

    def to_dict(self) -> dict[str, object]:
        """
        Convert to a JSON-friendly dict.
        """
        return asdict(self)


@dataclass(frozen=True)
class PreKernelCheckResult:
    """
    Result of a pre-kernel membership check.

    Attributes
    ----------
    in_set
        True if the point passes the pre-kernel test within tolerance.
    efficient
        Efficiency flag (sum x = v(N)).
    max_abs_delta
        Maximum absolute surplus imbalance:

        .. math::

            \\max_{i<j} |s_{ij}(x) - s_{ji}(x)|.
    pairs
        A (possibly truncated) list of per-pair diagnostics, sorted by decreasing
        |delta|.

    Notes
    -----
    - The pre-kernel condition requires efficiency and the equalities
      :math:`s_{ij}(x)=s_{ji}(x)` for all i != j.
    - ``pairs`` is intended for debugging/visualization: it contains the worst
      offending pairs.

    Examples
    --------
    >>> from tucoop.geometry.kernel_set import PairSurplusDiagnostics, PreKernelCheckResult
    >>> pairs = [PairSurplusDiagnostics(i=0, j=1, s_ij=0.0, s_ji=0.0, delta=0.0, argmax_ij=1, argmax_ji=2)]
    >>> res = PreKernelCheckResult(in_set=True, efficient=True, max_abs_delta=0.0, pairs=pairs)
    >>> res.in_set
    True
    """

    in_set: bool
    efficient: bool
    max_abs_delta: float
    pairs: list[PairSurplusDiagnostics]

    def to_dict(self) -> dict[str, object]:
        """
        Convert to a JSON-friendly dict.
        """
        return asdict(self)


@dataclass(frozen=True)
class KernelCheckResult:
    """
    Result of a kernel membership check.

    Attributes
    ----------
    in_set
        True if the point passes the kernel test within tolerance.
    efficient
        Efficiency flag (sum x = v(N)).
    imputation
        Imputation flag (efficiency + individual rationality).
    max_violation
        Maximum kernel complementarity violation detected.
    required_bounds
        List of players that, according to dominance relations, must be at their
        individual rationality bound (x_i = v({i})) for kernel membership.
    pairs
        A (possibly truncated) list of per-pair surplus diagnostics, sorted by
        decreasing |delta|.

    Notes
    -----
    Kernel membership (for imputations) can be stated as:

    - x is an imputation
    - for each pair (i, j):
      if :math:`s_{ij}(x) > s_{ji}(x)` then player j must be at its lower bound
      :math:`x_j = v(\\{j\\})`.

    Equivalently, whenever both i and j are interior (strictly above bounds),
    we must have :math:`s_{ij}(x)=s_{ji}(x)` within tolerance.

    Examples
    --------
    >>> from tucoop.geometry.kernel_set import KernelCheckResult
    >>> res = KernelCheckResult(in_set=False, efficient=True, imputation=False, max_violation=1.0, required_bounds=[0], pairs=[])
    >>> res.in_set
    False
    """

    in_set: bool
    efficient: bool
    imputation: bool
    max_violation: float
    required_bounds: list[int]
    pairs: list[PairSurplusDiagnostics]

    def to_dict(self) -> dict[str, object]:
        """
        Convert to a JSON-friendly dict.
        """
        return asdict(self)


@dataclass(frozen=True)
class PreKernelSet:
    """
    Pre-kernel set (set-valued).

    Definition
    ----------
    A (pre-)kernel element x (typically in the pre-imputation set) satisfies:

    - efficiency: :math:`\\sum_i x_i = v(N)`
    - pairwise surplus equalities: :math:`s_{ij}(x) = s_{ji}(x)` for all i != j

    where :math:`s_{ij}(x)` is the maximum excess over coalitions containing i
    and excluding j.

    Practical notes
    ---------------
    - Checking membership is **exponential** in n because it enumerates coalitions
      to compute argmax surpluses.
    - This class is meant for small-n visualization and diagnostics.

    Parameters
    ----------
    game
        TU game.
    tol
        Default tolerance used in membership tests.
    max_iter
        Iteration limit used by ``element()`` (solver).
    max_players
        Safety cap for exponential routines.
    approx_max_coalitions_per_pair
        If provided, approximate each pairwise surplus by checking only this many
        random coalitions per ordered pair (i, j). This keeps ``O(2^n)`` DP for
        coalition sums but avoids scanning all admissible coalitions per pair.
    approx_seed
        Random seed used for the approximation.

    Examples
    --------
    Run a basic check (the returned object contains detailed diagnostics):

    >>> from tucoop import Game
    >>> from tucoop.geometry.kernel_set import PreKernelSet
    >>> g = Game.from_coalitions(n_players=3, values={
    ...     0: 0.0,
    ...     1: 1.0, 2: 1.0, 4: 1.0,
    ...     3: 2.0, 5: 2.0, 6: 2.0,
    ...     7: 4.0,
    ... })
    >>> pk = PreKernelSet(g)
    >>> out = pk.check([1.0, 1.0, 2.0], top_k=2)
    >>> isinstance(out.in_set, bool)
    True
    """

    game: GameProtocol
    tol: float = DEFAULT_KERNEL_TOL
    max_iter: int = DEFAULT_KERNEL_MAX_ITER
    max_players: int = DEFAULT_KERNEL_MAX_PLAYERS
    approx_max_coalitions_per_pair: int | None = DEFAULT_KERNEL_APPROX_MAX_COALITIONS_PER_PAIR
    approx_seed: int | None = DEFAULT_KERNEL_APPROX_SEED

    def element(self) -> list[float]:
        """
        Compute one representative pre-kernel element using the iterative solver.

        Returns
        -------
        list[float]
            A candidate pre-kernel allocation.

        Notes
        -----
        Delegates to `tucoop.solutions.kernel.prekernel`.
        """
        from ..solutions.kernel import prekernel

        return prekernel(
            self.game,
            tol=float(self.tol),
            max_iter=int(self.max_iter),
            approx_max_coalitions_per_pair=self.approx_max_coalitions_per_pair,
            approx_seed=self.approx_seed,
        ).x

    def contains(self, x: list[float], *, tol: float | None = None) -> bool:
        """
        Return True iff x is in the pre-kernel (within tolerance).

        Parameters
        ----------
        x
            Candidate allocation.
        tol
            Optional override tolerance.

        Returns
        -------
        bool
        """
        return bool(self.check(x, tol=self.tol if tol is None else float(tol)).in_set)

    def check(self, x: list[float], *, tol: float | None = None, top_k: int = 8) -> PreKernelCheckResult:
        """
        Compute a pre-kernel membership check plus pairwise surplus diagnostics.

        Parameters
        ----------
        x
            Candidate allocation (length ``n_players``).
        tol
            Optional override tolerance.
        top_k
            Include at most this many (i, j) diagnostic entries (largest imbalances).

        Returns
        -------
        PreKernelCheckResult
            Membership result + diagnostics.

        Raises
        ------
        NotSupportedError
            If ``n_players`` is above ``max_players``.
        InvalidParameterError
            If ``x`` has the wrong length.

        Examples
        --------
        Minimal 3-player example (complete characteristic function):

        >>> from tucoop import Game
        >>> from tucoop.geometry.kernel_set import PreKernelSet
        >>> g = Game.from_coalitions(n_players=3, values={
        ...     0:0, 1:0, 2:0, 4:0,
        ...     3:0, 5:0, 6:0,
        ...     7:1,
        ... })
        >>> pk = PreKernelSet(g)
        >>> x = [1/3, 1/3, 1/3]
        >>> pk.check(x).efficient
        True
        """
        n = self.game.n_players
        if n > int(self.max_players):
            raise NotSupportedError(f"PreKernelSet.check is exponential; requires n<={self.max_players} (got n={n})")
        if len(x) != n:
            raise InvalidParameterError("x must have length n_players")

        t = self.tol if tol is None else float(tol)
        efficient = _is_efficient(self.game, x, tol=t)
        evaluator = _get_evaluator(n)
        values = None if self.approx_max_coalitions_per_pair is not None else _all_values_list(self.game)
        rng = None
        if self.approx_max_coalitions_per_pair is not None:
            from random import Random

            rng = Random(self.approx_seed)
        x_sums = _coalition_sums_dp(x, n_players=n)

        pairs: list[PairSurplusDiagnostics] = []
        worst = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                sij, Sij = evaluator.surplus_and_argmax(
                    self.game,
                    x,
                    i,
                    j,
                    values=values,
                    x_sums=x_sums,
                    approx_max_coalitions=self.approx_max_coalitions_per_pair,
                    rng=rng,
                )
                sji, Sji = evaluator.surplus_and_argmax(
                    self.game,
                    x,
                    j,
                    i,
                    values=values,
                    x_sums=x_sums,
                    approx_max_coalitions=self.approx_max_coalitions_per_pair,
                    rng=rng,
                )
                d = float(sij - sji)
                worst = max(worst, abs(d))
                pairs.append(
                    PairSurplusDiagnostics(
                        i=int(i),
                        j=int(j),
                        s_ij=float(sij),
                        s_ji=float(sji),
                        delta=float(d),
                        argmax_ij=int(Sij),
                        argmax_ji=int(Sji),
                    )
                )

        pairs.sort(key=lambda p: (-abs(p.delta), p.i, p.j))
        pairs = pairs[: max(0, int(top_k))]
        in_set = efficient and worst <= t
        return PreKernelCheckResult(in_set=bool(in_set), efficient=bool(efficient), max_abs_delta=float(worst), pairs=pairs)

    def explain(self, x: list[float], *, tol: float | None = None, top_k: int = 3) -> list[str]:
        """
        Return a short human-readable explanation of pre-kernel membership.

        Notes
        -----
        This is a thin wrapper around `check`.
        """
        d = self.check(x, tol=tol, top_k=top_k)
        if not d.efficient:
            return [
                f"Not efficient: sum(x)={float(sum(x)):.6g} but v(N)={float(self.game.value(self.game.grand_coalition)):.6g}."
            ]
        if d.in_set:
            return [f"In the pre-kernel (max |delta|={d.max_abs_delta:.6g})."]
        if d.pairs:
            p0 = d.pairs[0]
            return [
                f"Not in the pre-kernel (max |delta|={d.max_abs_delta:.6g}).",
                f"Worst pair: (i={p0.i}, j={p0.j}) delta={p0.delta:.6g} with argmax_ij={p0.argmax_ij}, argmax_ji={p0.argmax_ji}.",
            ]
        return [f"Not in the pre-kernel (max |delta|={d.max_abs_delta:.6g})."]

    def sample_points(
        self,
        *,
        n_samples: int = 2000,
        seed: int | None = None,
        max_points: int = 50,
        tol: float | None = None,
    ) -> list[list[float]]:
        """
        Heuristically search for pre-kernel points by sampling the imputation set.

        This routine draws random imputations and returns those that pass the
        pre-kernel test (within tolerance).

        Parameters
        ----------
        n_samples
            Number of imputations to sample.
        seed
            Optional RNG seed.
        max_points
            Stop once this many passing points are found.
        tol
            Optional override tolerance for membership testing.

        Returns
        -------
        list[list[float]]
            Up to ``max_points`` candidate pre-kernel points found by sampling.

        Notes
        -----
        - Intended for visualization and debugging (small n).
        - Does not guarantee finding any point, even if the set is non-empty.
        """
        n = self.game.n_players
        if n > int(self.max_players):
            raise NotSupportedError(f"PreKernelSet.sample_points is exponential; requires n<={self.max_players} (got n={n})")

        t = self.tol if tol is None else float(tol)
        xs = sample_imputation_set(self.game, n_samples=int(n_samples), seed=seed)
        out: list[list[float]] = []
        for x in xs:
            if self.contains(x, tol=t):
                out.append([float(v) for v in x])
                if len(out) >= int(max_points):
                    break
        return out

    def sample_point(
        self,
        *,
        n_samples: int = 2000,
        seed: int | None = None,
        max_points: int = 50,
        tol: float | None = None,
    ) -> list[float] | None:
        """
        Return one pre-kernel point found by sampling, or None.

        Parameters
        ----------
        n_samples, seed, max_points, tol
            Forwarded to `sample_points`.

        Returns
        -------
        list[float] | None
        """
        pts = self.sample_points(n_samples=n_samples, seed=seed, max_points=max_points, tol=tol)
        return pts[0] if pts else None


@dataclass(frozen=True)
class KernelSet:
    """
    Kernel set (set-valued).

    Definition
    ----------
    Kernel membership (for imputations) can be stated as:

    - x is an imputation (efficiency + individual rationality)
    - for each pair (i, j):
      if :math:`s_{ij}(x) > s_{ji}(x)` (by more than tolerance),
      then player j must be at its lower bound: :math:`x_j = v(\\{j\\})`.

    Equivalently, whenever both i and j are interior
    (:math:`x_i > v(\\{i\\})` and :math:`x_j > v(\\{j\\})`),
    we must have :math:`s_{ij}(x) = s_{ji}(x)` within tolerance.

    Practical notes
    ---------------
    - Checking membership is **exponential** in n (coalition enumeration).
    - This class is intended for small-n diagnostics/visualization.

    Parameters
    ----------
    game
        TU game.
    tol
        Default tolerance used in membership tests.
    max_iter
        Iteration limit used by ``element()`` (solver).
    max_players
        Safety cap for exponential routines.
    approx_max_coalitions_per_pair
        If provided, approximate each pairwise surplus by checking only this many
        random coalitions per ordered pair (i, j).
    approx_seed
        Random seed used for the approximation.

    Examples
    --------
    >>> from tucoop import Game
    >>> from tucoop.geometry.kernel_set import KernelSet
    >>> g = Game.from_coalitions(n_players=3, values={
    ...     0: 0.0,
    ...     1: 1.0, 2: 1.0, 4: 1.0,
    ...     3: 2.0, 5: 2.0, 6: 2.0,
    ...     7: 4.0,
    ... })
    >>> ks = KernelSet(g)
    >>> out = ks.check([1.0, 1.0, 2.0], top_k=2)
    >>> isinstance(out.in_set, bool)
    True
    """

    game: GameProtocol
    tol: float = DEFAULT_KERNEL_TOL
    max_iter: int = DEFAULT_KERNEL_MAX_ITER
    max_players: int = DEFAULT_KERNEL_MAX_PLAYERS
    approx_max_coalitions_per_pair: int | None = DEFAULT_KERNEL_APPROX_MAX_COALITIONS_PER_PAIR
    approx_seed: int | None = DEFAULT_KERNEL_APPROX_SEED

    def element(self) -> list[float]:
        """
        Compute one representative kernel element using the iterative solver.

        Returns
        -------
        list[float]
            A candidate kernel allocation (intended to lie in the imputation set).

        Notes
        -----
        Delegates to `tucoop.solutions.kernel.kernel`.
        """
        from ..solutions.kernel import kernel

        return kernel(
            self.game,
            tol=float(self.tol),
            max_iter=int(self.max_iter),
            approx_max_coalitions_per_pair=self.approx_max_coalitions_per_pair,
            approx_seed=self.approx_seed,
        ).x

    def contains(self, x: list[float], *, tol: float | None = None) -> bool:
        """
        Return True iff x is in the kernel (within tolerance).

        Parameters
        ----------
        x
            Candidate allocation.
        tol
            Optional override tolerance.

        Returns
        -------
        bool
        """
        return bool(self.check(x, tol=self.tol if tol is None else float(tol)).in_set)

    def check(self, x: list[float], *, tol: float | None = None, top_k: int = 8) -> KernelCheckResult:
        """
        Compute a kernel membership check plus pairwise surplus diagnostics.

        Parameters
        ----------
        x
            Candidate allocation (length ``n_players``).
        tol
            Optional override tolerance.
        top_k
            Include at most this many (i, j) diagnostic entries.

        Returns
        -------
        KernelCheckResult
            Membership result + diagnostics.

        Raises
        ------
        NotSupportedError
            If ``n_players`` is above ``max_players``.
        InvalidParameterError
            If ``x`` has the wrong length.

        Examples
        --------
        Minimal 2-player game (kernel coincides with the imputation segment):

        >>> from tucoop import Game
        >>> from tucoop.geometry.kernel_set import KernelSet
        >>> g = Game.from_coalitions(n_players=2, values={0:0, 1:0, 2:0, 3:1})
        >>> ks = KernelSet(g)
        >>> ks.check([0.5, 0.5]).imputation
        True
        """
        n = self.game.n_players
        if n > int(self.max_players):
            raise NotSupportedError(f"KernelSet.check is exponential; requires n<={self.max_players} (got n={n})")
        if len(x) != n:
            raise InvalidParameterError("x must have length n_players")

        t = self.tol if tol is None else float(tol)
        efficient = _is_efficient(self.game, x, tol=t)
        imputation = _is_imputation(self.game, x, tol=t)
        lb = imputation_lower_bounds(self.game)
        evaluator = _get_evaluator(n)
        values = None if self.approx_max_coalitions_per_pair is not None else _all_values_list(self.game)
        rng = None
        if self.approx_max_coalitions_per_pair is not None:
            from random import Random

            rng = Random(self.approx_seed)
        x_sums = _coalition_sums_dp(x, n_players=n)

        required_bounds: set[int] = set(i for i in range(n) if float(x[i]) <= float(lb[i]) + t)

        max_violation = 0.0
        pairs: list[PairSurplusDiagnostics] = []
        for i in range(n):
            for j in range(i + 1, n):
                sij, Sij = evaluator.surplus_and_argmax(
                    self.game,
                    x,
                    i,
                    j,
                    values=values,
                    x_sums=x_sums,
                    approx_max_coalitions=self.approx_max_coalitions_per_pair,
                    rng=rng,
                )
                sji, Sji = evaluator.surplus_and_argmax(
                    self.game,
                    x,
                    j,
                    i,
                    values=values,
                    x_sums=x_sums,
                    approx_max_coalitions=self.approx_max_coalitions_per_pair,
                    rng=rng,
                )
                d = float(sij - sji)

                # If s_ij > s_ji, then j must be at bound.
                if d > t:
                    required_bounds.add(int(j))
                    if float(x[j]) > float(lb[j]) + t:
                        max_violation = max(max_violation, float(d))
                # If s_ji > s_ij, then i must be at bound.
                if -d > t:
                    required_bounds.add(int(i))
                    if float(x[i]) > float(lb[i]) + t:
                        max_violation = max(max_violation, float(-d))

                pairs.append(
                    PairSurplusDiagnostics(
                        i=int(i),
                        j=int(j),
                        s_ij=float(sij),
                        s_ji=float(sji),
                        delta=float(d),
                        argmax_ij=int(Sij),
                        argmax_ji=int(Sji),
                    )
                )

        pairs.sort(key=lambda p: (-abs(p.delta), p.i, p.j))
        pairs = pairs[: max(0, int(top_k))]

        in_set = bool(imputation) and float(max_violation) <= t
        return KernelCheckResult(
            in_set=bool(in_set),
            efficient=bool(efficient),
            imputation=bool(imputation),
            max_violation=float(max_violation),
            required_bounds=sorted(required_bounds),
            pairs=pairs,
        )

    def explain(self, x: list[float], *, tol: float | None = None, top_k: int = 3) -> list[str]:
        """
        Return a short human-readable explanation of kernel membership.

        Notes
        -----
        This is a thin wrapper around `check`.
        """
        d = self.check(x, tol=tol, top_k=top_k)
        lines: list[str] = []
        if not d.efficient:
            lines.append(
                f"Not efficient: sum(x)={float(sum(x)):.6g} but v(N)={float(self.game.value(self.game.grand_coalition)):.6g}."
            )
        if not d.imputation:
            lines.append("Not an imputation (kernel is defined on imputations).")
        if d.in_set:
            lines.append(f"In the kernel (max_violation={d.max_violation:.6g}).")
            return lines
        lines.append(f"Not in the kernel (max_violation={d.max_violation:.6g}).")
        if d.required_bounds:
            lines.append(f"Required-at-bound players (by surplus dominance): {d.required_bounds}.")
        if d.pairs:
            p0 = d.pairs[0]
            lines.append(
                f"Worst pair: (i={p0.i}, j={p0.j}) delta={p0.delta:.6g} with argmax_ij={p0.argmax_ij}, argmax_ji={p0.argmax_ji}."
            )
        return lines

    def sample_points(
        self,
        *,
        n_samples: int = 5000,
        seed: int | None = None,
        max_points: int = 50,
        tol: float | None = None,
    ) -> list[list[float]]:
        """
        Heuristically search for kernel points by sampling the imputation set.

        Parameters
        ----------
        n_samples
            Number of imputations to sample.
        seed
            Optional RNG seed.
        max_points
            Stop once this many passing points are found.
        tol
            Optional override tolerance for membership testing.

        Returns
        -------
        list[list[float]]
            Up to ``max_points`` candidate kernel points found by sampling.

        Notes
        -----
        - Intended for visualization and debugging (small n).
        - Does not guarantee finding any point, even if the kernel is non-empty.
        - Points are de-duplicated with a cheap L-infinity rule within tolerance.
        """
        n = self.game.n_players
        if n > int(self.max_players):
            raise NotSupportedError(f"KernelSet.sample_points is exponential; requires n<={self.max_players} (got n={n})")

        t = self.tol if tol is None else float(tol)
        xs = sample_imputation_set(self.game, n_samples=int(n_samples), seed=seed)

        out: list[list[float]] = []
        seen: list[list[float]] = []
        for x in xs:
            if not self.contains(x, tol=t):
                continue
            # De-dup by L-infinity within tolerance (cheap).
            dup = False
            for y in seen:
                if max(abs(float(x[i]) - float(y[i])) for i in range(n)) <= 5.0 * t:
                    dup = True
                    break
            if dup:
                continue
            seen.append([float(v) for v in x])
            out.append([float(v) for v in x])
            if len(out) >= int(max_points):
                break
        return out

    def sample_point(
        self,
        *,
        n_samples: int = 5000,
        seed: int | None = None,
        max_points: int = 50,
        tol: float | None = None,
    ) -> list[float] | None:
        """
        Return one kernel point found by sampling, or None.

        Parameters
        ----------
        n_samples, seed, max_points, tol
            Forwarded to `sample_points`.

        Returns
        -------
        list[float] | None
        """
        pts = self.sample_points(n_samples=n_samples, seed=seed, max_points=max_points, tol=tol)
        return pts[0] if pts else None


__all__ = [
    "PairSurplusDiagnostics",
    "PreKernelCheckResult",
    "KernelCheckResult",
    "PreKernelSet",
    "KernelSet",
]
