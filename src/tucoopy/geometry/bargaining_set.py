"""
# Aumannâ€“Maschler bargaining set (small-$n$ helpers).

This module implements a pragmatic, *small-$n$* oriented toolkit around the
**bargaining set** for transferable-utility (TU) cooperative games.

The bargaining set is defined via objections and counter-objections and is
computationally expensive in general. The implementation here focuses on:

- clear data structures (`Objection`, `CounterObjection`),
- a test function suitable for diagnostics/visualization workflows, and
- a sampling-based approach (via :func:`tucoopy.geometry.sampling.sample_imputation_set`)
  for exploring candidate points when exhaustive checks are infeasible.

Notes
-----
This module is intended for use with `tucoopy.geometry.ImputationSet` and
other core-family objects. For large games, prefer approximate workflows and
explicit limits (number of samples, max attempts, etc.).

Examples
--------
Create a small game and instantiate the bargaining-set helper:

>>> from tucoopy import Game
>>> from tucoopy.geometry.bargaining_set import BargainingSet
>>> g = Game.from_coalitions(n_players=3, values={
...     0: 0.0,
...     1: 1.0, 2: 1.0, 4: 1.0,
...     3: 2.0, 5: 2.0, 6: 2.0,
...     7: 4.0,
... })
>>> bs = BargainingSet(g)
>>> isinstance(bs, BargainingSet)
True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from ..base.config import DEFAULT_BARGAINING_TOL, DEFAULT_MAX_PLAYERS
from ..base.coalition import Coalition, all_coalitions, coalition_sum, players
from ..base.types import GameProtocol
from ..base.exceptions import InvalidParameterError, NotSupportedError
from .imputation_set import ImputationSet, imputation_lower_bounds
from .sampling import sample_imputation_set


@dataclass(frozen=True)
class Objection:
    """
    Objection $(S, y)$ by player $i$ against player $j$.

    Background
    ----------
    In the Aumannâ€“Maschler bargaining set, an **objection** is a pair $(S, y)$ where:

    - $S$ is a coalition such that $i \\in S$ and $j \\notin S$
    - $y$ is an allocation for coalition $S$ (extended to $N$ by leaving non-members as $0$),
      satisfying feasibility on $S$ and improving all members of $S$ relative to $x$

    In this implementation we follow the common *imputation-based* notion:

    - $y$ is feasible for $S$:  $$\\sum_{k in S} y_k = v(S)$$
    - each $k \\in S$ is at least as well off as in $x: y_k \\geq x_k$
    - player $i$ strictly improves: $y_i > x_i$

    The pair $(i, j)$ identifies "who objects against whom".

    Attributes
    ----------
    i, j
        Player indices (0-based).
    coalition
        Coalition mask S.
    y
        Full length-n vector representing the objection allocation (entries outside S
        are typically unused / 0 in this implementation).

    Examples
    --------
    >>> from tucoopy.geometry.bargaining_set import Objection
    >>> obj = Objection(i=0, j=2, coalition=0b011, y=[1.5, 0.5, 0.0])
    >>> (obj.i, obj.j, obj.coalition)
    (0, 2, 3)
    """

    i: int
    j: int
    coalition: Coalition
    y: list[float]
    counter_search_attempts: list["CounterobjectionAttempt"] | None = None


@dataclass(frozen=True)
class CounterobjectionAttempt:
    """
    Attempt to find a counterobjection on a coalition $T$.

    This is optional diagnostic information returned when a bargaining-set
    check fails and the caller requests counter-search details.

    Examples
    --------
    >>> from tucoopy.geometry.bargaining_set import CounterobjectionAttempt
    >>> att = CounterobjectionAttempt(coalition=0b101, feasible=True, achieved_maximized_value=1.0, required_value=0.5)
    >>> att.feasible
    True
    """

    coalition: Coalition
    feasible: bool
    achieved_maximized_value: float | None
    required_value: float


@dataclass(frozen=True)
class BargainingCheckResult:
    """
    Result of a bargaining-set membership check.

    Attributes
    ----------
    in_set
        True iff the allocation passed the (heuristic) bargaining-set test.
    witness
        If `in_set` is False, an objection (S, y) that has **no counterobjection**
        under the heuristic search strategy. If `in_set` is True, this is None.

    Notes
    -----
    This result is intended primarily for debugging/visualization: a non-empty
    witness helps explain *why* a point was rejected.

    Examples
    --------
    >>> from tucoopy.geometry.bargaining_set import BargainingCheckResult
    >>> res = BargainingCheckResult(in_set=True)
    >>> res.in_set
    True
    """

    in_set: bool
    witness: Objection | None = None


@dataclass(frozen=True)
class BargainingSet:
    """
    Aumannâ€“Maschler bargaining set (small-$n$ heuristic membership test).

    Definition (informal)
    ---------------------
    Let $x$ be an **imputation** (efficient and individually rational). An objection
    $(S, y)$ by $i$ against $j$ is "justified" if it improves coalition members in $S$
    (and strictly improves $i$). The bargaining set contains imputations for which
    **every** objection has a counterobjection.

    This object provides:

    - `contains(x)` / `check(x)` as a **heuristic** membership test ($x$ must be an imputation)
    - `sample_points(...)` rejection sampling for visualization

    Implementation scope
    --------------------
    This implementation is intended for **very small games** (default $n \\leq 4$).
    It searches over coalitions and constructs objections/counterobjections using
    a closed-form solver for the subproblem:

    $$
    \\text{maximize } z_j
    $$
    
    subject to

    $$
    \\begin{cases} 
    \\sum_{k \\in T} z_k = v(T), \\\\
    z_k \\geq \\text{lower}_k \\text{ for } k \\in T
    \\end{cases}
    $$

    That subproblem is trivial: assign all players at their lower bounds and give
    all slack to the maximized player.

    Parameters
    ----------
    game
        TU game.
    tol
        Numerical tolerance used for comparisons (excess positivity, strict improvements, etc.).
    max_objections_per_pair
        Limit the number of candidate coalitions S tried for each ordered pair (i, j).
        Candidates are sorted by descending excess and truncated.
    n_max
        Safety limit on the number of players. The search is exponential in n.

    Notes
    -----
    - This is **not** a complete bargaining-set algorithm for arbitrary $n$.
      It is meant for pedagogical use and visualization.
    - The method can return false negatives/positives in principle, because it
      truncates the search (`max_objections_per_pair`) and uses simple tie-breaking.

    Examples
    --------
    Basic construction and type-checking:

    >>> from tucoopy import Game
    >>> g = Game.from_coalitions(n_players=3, values={
    ...     0: 0.0,
    ...     1: 1.0, 2: 1.0, 4: 1.0,
    ...     3: 2.0, 5: 2.0, 6: 2.0,
    ...     7: 4.0,
    ... })
    >>> bs = BargainingSet(g, max_objections_per_pair=4)
    >>> x = [1.0, 1.0, 2.0]
    >>> isinstance(bs.contains(x), bool)
    True

    Sampling imputations (deterministic with a seed):

    >>> pts = bs.sample_points(n_samples=20, seed=0, max_points=5)
    >>> len(pts) <= 5
    True
    """

    game: GameProtocol
    tol: float = DEFAULT_BARGAINING_TOL
    max_objections_per_pair: int = 8
    n_max: int = DEFAULT_MAX_PLAYERS
    _last_counter_attempts: list[CounterobjectionAttempt] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def _lp_maximize_component(
        self,
        *,
        coalition: Coalition,
        lower_bounds: dict[int, float],
        maximize_player: int,
    ) -> tuple[list[float] | None, float | None]:
        """
        Maximize one component over a coalition under a sum constraint and lower bounds.

        Problem
        -------
        For a fixed coalition $T$ and a fixed player $j \\in T$, solve:

        $$
        \\text{maximize } z_j
        $$
    
        subject to:
        
        $$
        \\begin{cases} 
        \\sum_{k \\in T} z_k = v(T), \\\\
        z_k \\geq \\text{lower}_k \\text{ for } k \\in T
        \\end{cases}
        $$

        Closed-form solution
        --------------------
        This LP has a trivial optimizer:

        1. Set every $k \\in T$, $k \\neq j$ to its lower bound.
        2. Give the remaining slack to $j$.

        Feasibility requires:
        
        $$
        \\sum_{k\\in T} \\text{lower}_k \\le v(T).
        $$

        Parameters
        ----------
        coalition
            Coalition mask T.
        lower_bounds
            Lower bounds for each k in T (must include every member).
        maximize_player
            Player index j to maximize.

        Returns
        -------
        (z_full, z_j)
            - z_full is a length-n vector (entries outside T are 0.0)
            - z_j is the achieved value for the maximized component

            If infeasible, returns (None, None).

        Raises
        ------
        InvalidParameterError
            If maximize_player is not in the coalition or lower bounds are missing.
        """
        n = self.game.n_players
        T = players(coalition, n_players=n)
        if maximize_player not in T:
            raise InvalidParameterError("maximize_player must be in coalition")

        vT = float(self.game.value(int(coalition)))

        missing = [p for p in T if p not in lower_bounds]
        if missing:
            raise InvalidParameterError(f"lower_bounds must include all members of the coalition; missing {missing}")

        lb_sum = float(sum(float(lower_bounds[p]) for p in T))
        slack = float(vT - lb_sum)
        if slack < -float(self.tol):
            return None, None
        if abs(slack) <= float(self.tol):
            slack = 0.0

        z_full = [0.0] * n
        for p in T:
            z_full[p] = float(lower_bounds[p])
        z_full[int(maximize_player)] = float(lower_bounds[int(maximize_player)]) + float(slack)
        return z_full, float(z_full[int(maximize_player)])

    def _counterobjection_exists(
        self,
        *,
        i: int,
        j: int,
        S: int,
        x: list[float],
        y: list[float],
        search: str = "top_excess",
        rng: Random | None = None,
        max_coalitions: int | None = None,
        record_attempts: bool = False,
    ) -> bool:
        """
        Check whether $j$ has a counterobjection against $i$ to an objection $(S, y)$.

        Counterobjection search (heuristic)
        -----------------------------------
        We brute-force coalitions $T$ such that:
        
        - $j \\in T$
        - $i \\notin T$
        - $T$ is a non-empty proper coalition

        For each such $T$, we build lower bounds on $T$ as:
        
        - for players $k \\in T \\cap S$:   $l_k = y_k$
        - for players $k \\in T \\setminus S$:  $l_k = x_k$

        Then we maximize $z_j$ over $T$ subject to feasibility and these lower bounds.
        If we can achieve:

        $$z_j > y_j + \\text{tol}$$

        we treat it as a valid counterobjection.

        Parameters
        ----------
        i, j
            Players in the objection relation (i objects against j).
        S
            Objection coalition mask.
        x
            Original imputation.
        y
            Objection allocation vector (length n).

        Parameters
        ----------
        search
            Search policy for counterobjection coalitions T. One of:

            - ``"top_excess"``: try T with largest excess first (deterministic)
            - ``"random"``: random order (requires ``rng``)
            - ``"all"``: exhaustive order (increasing mask)
        rng
            Optional RNG used when ``search="random"``.
        max_coalitions
            Optional limit on the number of T coalitions tested.
        record_attempts
            If True, populate ``self._last_counter_attempts`` for debugging.

        Returns
        -------
        bool
            True iff a counterobjection is found under the chosen policy.

        Notes
        -----
        This is tailored to $n \\leq 4$. For larger $n$ the brute-force over $T$ is too expensive.
        """
        n = self.game.n_players
        N = self.game.grand_coalition

        candidates_T: list[int] = []
        for T in all_coalitions(n):
            if T == 0 or T == N:
                continue
            if not (T & (1 << j)):
                continue
            if T & (1 << i):
                continue
            candidates_T.append(int(T))

        def excess_T(mask: int) -> float:
            return float(self.game.value(mask) - coalition_sum(int(mask), x, n_players=n))

        policy = str(search)
        if policy not in {"top_excess", "random", "all"}:
            raise InvalidParameterError("search must be one of: 'top_excess', 'random', 'all'")
        if policy == "top_excess":
            candidates_T.sort(key=excess_T, reverse=True)
        elif policy == "random":
            if rng is None:
                rng = Random(0)
            rng.shuffle(candidates_T)
        else:
            candidates_T.sort()

        attempts: list[CounterobjectionAttempt] = []
        limit = len(candidates_T) if max_coalitions is None else min(len(candidates_T), int(max_coalitions))
        required = float(y[j]) + float(self.tol)

        for T in candidates_T[:limit]:
            lower: dict[int, float] = {}
            for k in range(n):
                if not (int(T) & (1 << k)):
                    continue
                lower[k] = float(y[k]) if (S & (1 << k)) else float(x[k])

            z, zj = self._lp_maximize_component(
                coalition=int(T),
                lower_bounds=lower,
                maximize_player=j,
            )
            if z is None or zj is None:
                if record_attempts:
                    attempts.append(
                        CounterobjectionAttempt(
                            coalition=int(T),
                            feasible=False,
                            achieved_maximized_value=None,
                            required_value=required,
                        )
                    )
                continue
            if record_attempts:
                attempts.append(
                    CounterobjectionAttempt(
                        coalition=int(T),
                        feasible=True,
                        achieved_maximized_value=float(zj),
                        required_value=required,
                    )
                )
            if float(zj) > required:
                if record_attempts:
                    object.__setattr__(self, "_last_counter_attempts", attempts)
                return True

        if record_attempts:
            object.__setattr__(self, "_last_counter_attempts", attempts)
        return False

    def check(
        self,
        x: list[float],
        *,
        search: str = "top_excess",
        seed: int | None = None,
        max_objections_per_pair: int | None = None,
        max_counterobjections_per_pair: int | None = None,
        record_counter_attempts: bool = False,
    ) -> BargainingCheckResult:
        """
        Heuristic bargaining-set membership test (small $n$).

        Overview
        --------
        1. Verify that $x$ is an imputation.
        2. For each ordered pair $(i, j)$, enumerate candidate coalitions $S$ with:
           $i \\in S$, $j \\notin S$, and positive excess $e(S,x) > \\text{tol}$.
        3. For each candidate $S$ (up to max_objections_per_pair):

            - construct an objection $(S, y)$ by maximizing $y_i$ subject to
             $y_k \\geq x_k$ for $k \\in S$ and $\\sum_{k \\in S} y_k = v(S)$.
            - if $y_i > x_i + \\text{tol}$, this is a genuine objection candidate.
            - if no counterobjection exists for it, return a witness.
        
        4. If every considered objection is countered, return `in_set=True`.

        Parameters
        ----------
        x
            Candidate allocation (must be an imputation).
        search
            Search policy for objection coalitions S and counterobjection coalitions T.
            One of: ``"top_excess"``, ``"random"``, ``"all"``.
        seed
            Optional RNG seed used when ``search="random"``.
        max_objections_per_pair
            Optional override for the per-pair limit on objection coalitions S.
        max_counterobjections_per_pair
            Optional limit for counterobjection coalitions T tested per objection.
        record_counter_attempts
            If True and the check fails, include attempted counterobjections in the witness.

        Returns
        -------
        BargainingCheckResult
            Membership decision plus optional witness objection.

        Notes
        -----
        - Candidate coalitions S are sorted by descending excess to find strong
          objections first.
        - The search is truncated; passing this test does not formally prove
          bargaining-set membership for arbitrary games.
        """
        n = self.game.n_players
        if n > int(self.n_max):
            raise NotSupportedError(f"BargainingSet is restricted to n<={self.n_max} (got n={n})")

        if not ImputationSet(self.game).contains(x, tol=self.tol):
            return BargainingCheckResult(in_set=False, witness=None)

        N = self.game.grand_coalition

        def excess(mask: int) -> float:
            return float(self.game.value(mask) - coalition_sum(int(mask), x, n_players=n))

        rng = Random(seed) if seed is not None else None
        policy = str(search)
        if policy not in {"top_excess", "random", "all"}:
            raise InvalidParameterError("search must be one of: 'top_excess', 'random', 'all'")
        max_S = int(self.max_objections_per_pair) if max_objections_per_pair is None else int(max_objections_per_pair)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                candidates: list[int] = []
                for S in all_coalitions(n):
                    if S == 0 or S == N:
                        continue
                    if not (S & (1 << i)):
                        continue
                    if S & (1 << j):
                        continue
                    if excess(S) <= float(self.tol):
                        continue
                    candidates.append(S)

                if policy == "top_excess":
                    candidates.sort(key=lambda m: excess(m), reverse=True)
                elif policy == "random":
                    (rng or Random(0)).shuffle(candidates)
                else:
                    candidates.sort()

                tried = 0
                for S in candidates:
                    if max_S is not None and tried >= int(max_S):
                        break
                    tried += 1

                    lower: dict[int, float] = {k: float(x[k]) for k in range(n) if S & (1 << k)}
                    y, yi = self._lp_maximize_component(
                        coalition=S,
                        lower_bounds=lower,
                        maximize_player=i,
                    )
                    if y is None or yi is None:
                        continue
                    if yi <= float(x[i]) + float(self.tol):
                        continue

                    found = self._counterobjection_exists(
                        i=i,
                        j=j,
                        S=S,
                        x=x,
                        y=y,
                        search=policy,
                        rng=rng,
                        max_coalitions=max_counterobjections_per_pair,
                        record_attempts=record_counter_attempts,
                    )
                    if not found:
                        attempts = getattr(self, "_last_counter_attempts", None) if record_counter_attempts else None
                        return BargainingCheckResult(
                            in_set=False,
                            witness=Objection(i=i, j=j, coalition=S, y=y, counter_search_attempts=attempts),
                        )

        return BargainingCheckResult(in_set=True, witness=None)

    def contains(self, x: list[float]) -> bool:
        """
        Return True iff $x$ passes the bargaining-set heuristic check.
        """
        return self.check(x).in_set

    def explain(
        self,
        x: list[float],
        *,
        max_objections_per_pair: int | None = None,
        max_counterobjections_per_pair: int | None = None,
        search: str = "top_excess",
        seed: int | None = None,
        record_counter_attempts: bool = False,
    ) -> list[str]:
        """
        Return a short human-readable explanation of bargaining-set membership.

        Notes
        -----
        This is a thin wrapper around `check`.
        """
        d = self.check(
            x,
            search=search,
            seed=seed,
            max_objections_per_pair=max_objections_per_pair,
            max_counterobjections_per_pair=max_counterobjections_per_pair,
            record_counter_attempts=record_counter_attempts,
        )
        if d.in_set:
            return ["In the bargaining set (heuristic check passed)."]
        if d.witness is None:
            return ["Not in the bargaining set (heuristic check failed)."]

        w = d.witness
        lines = [
            "Not in the bargaining set (found an objection without counterobjection under the current search policy).",
            f"Witness objection: i={w.i} against j={w.j} on coalition S={int(w.coalition)}.",
        ]
        if record_counter_attempts and w.counter_search_attempts:
            # Keep this short: just one attempt summary.
            a0 = w.counter_search_attempts[0]
            lines.append(
                f"Counter-search example: tried T={int(a0.coalition)} feasible={a0.feasible} required={a0.required_value:.6g} achieved={a0.achieved_maximized_value}."
            )
        return lines

    def sample_points(
        self,
        *,
        n_samples: int,
        seed: int | None = None,
        max_attempts: int | None = None,
    ) -> list[list[float]]:
        """
        Sample bargaining-set points for visualization (small $n$) via rejection sampling.

        Procedure
        ---------
        1. Draw candidate points from the imputation set using `tucoopy.geometry.sampling.sample_imputation_set`.
        2. Keep points that satisfy `contains`.

        Parameters
        ----------
        n_samples
            Target number of accepted points.
        seed
            Optional seed for the underlying imputation sampler.
        max_attempts
            Maximum number of candidate points tested. If None, defaults to
            ``200 * n_samples``.

        Returns
        -------
        list[list[float]]
            Accepted bargaining-set points. May contain fewer than `n_samples` points
            if `max_attempts` is reached.

        Notes
        -----
        Degenerate imputation simplex:
        If the imputation set is empty, returns ``[]``.
        If it is a singleton, returns either ``[x0]`` or ``[]`` depending on membership.
        """
        n = self.game.n_players
        if n > int(self.n_max):
            raise NotSupportedError(f"BargainingSet is restricted to n<={self.n_max} (got n={n})")
        if n_samples < 1:
            raise InvalidParameterError("n_samples must be >= 1")

        vN = float(self.game.value(self.game.grand_coalition))
        l = imputation_lower_bounds(self.game)
        r = vN - sum(l)
        if r < -float(self.tol):
            return []
        if abs(r) <= float(self.tol):
            x0 = [float(v) for v in l]
            return [x0] if self.contains(x0) else []

        attempts = 0
        limit = max_attempts if max_attempts is not None else 200 * n_samples

        out: list[list[float]] = []
        batch_seed = seed
        while len(out) < n_samples and attempts < limit:
            remaining_attempts = int(limit - attempts)
            if remaining_attempts <= 0:
                break

            batch = min(max(8, n_samples - len(out)), remaining_attempts)
            candidates = sample_imputation_set(self.game, n_samples=batch, seed=batch_seed, tol=float(self.tol))
            attempts += len(candidates)
            if batch_seed is not None:
                batch_seed = int(batch_seed) + 1

            for x in candidates:
                if self.contains(x):
                    out.append([float(v) for v in x])
                    if len(out) >= n_samples:
                        break
        return out

    def sample_point(
        self,
        *,
        n_samples: int = 200,
        seed: int | None = None,
        max_attempts: int | None = None,
    ) -> list[float] | None:
        """
        Return one bargaining-set point found by sampling, or None.

        This is a convenience wrapper around `sample_points`.
        """
        pts = self.sample_points(n_samples=n_samples, seed=seed, max_attempts=max_attempts)
        return pts[0] if pts else None

    def scan_imputation_grid(
        self,
        *,
        step: float,
        max_points: int = 5000,
        search: str = "top_excess",
        seed: int | None = None,
        max_objections_per_pair: int | None = 1,
        max_counterobjections_per_pair: int | None = 1,
    ) -> list[tuple[list[float], bool]]:
        """
        Scan a coarse grid over the 3-player imputation simplex (sanity-check helper).

        This enumerates points in the imputation set for $n=3$ by discretizing the
        "slack" coordinates:

        - $x = l + y$ where $l$ is the imputation lower bound vector, and
        - $y_i \\ge 0$ with $\\sum_i y_i = r := v(N) - \\sum_i l_i$.

        The scan uses a simple grid spacing `step` in the payoff units of the game,
        then tests each point with `contains`.

        Parameters
        ----------
        step
            Grid spacing in payoff units. Larger values mean fewer points.
        max_points
            Hard limit on returned points.
        search, seed, max_objections_per_pair, max_counterobjections_per_pair
            Passed to `check`.

        Returns
        -------
        list[tuple[list[float], bool]]
            List of (x, in_set) pairs. Points are returned in a deterministic order.

        Notes
        -----
        - Implemented only for $n=3$.
        - This is a debugging/visualization helper; it does not provide guarantees
          about bargaining-set membership for arbitrary games.
        - The membership test requires an LP backend when non-trivial objections are
          present.

        Examples
        --------
        >>> from tucoopy import Game
        >>> from tucoopy.geometry import BargainingSet
        >>> g = Game.from_coalitions(
        ...     n_players=3,
        ...     values={
        ...         (): 0.0,
        ...         (0,): 1.0, (1,): 1.2, (2,): 0.8,
        ...         (0, 1): 2.8, (0, 2): 2.2, (1, 2): 2.0,
        ...         (0, 1, 2): 4.0,
        ...     },
        ... )
        >>> bs = BargainingSet(g)
        >>> pts = bs.scan_imputation_grid(step=0.5)  # doctest: +SKIP
        >>> len(pts) > 0  # doctest: +SKIP
        True
        """
        n = int(self.game.n_players)
        if n != 3:
            raise NotSupportedError("scan_imputation_grid is implemented only for n_players=3")
        if float(step) <= 0.0:
            raise InvalidParameterError("step must be > 0")
        if int(max_points) < 1:
            raise InvalidParameterError("max_points must be >= 1")

        vN = float(self.game.value(self.game.grand_coalition))
        l = imputation_lower_bounds(self.game)
        r = float(vN - sum(l))
        if r < -float(self.tol):
            return []
        if abs(r) <= float(self.tol):
            x0 = [float(v) for v in l]
            return [(x0, bool(self.contains(x0)))]

        h = float(step)
        out: list[tuple[list[float], bool]] = []
        y0 = 0.0
        while y0 <= r + 1e-12 and len(out) < int(max_points):
            y1 = 0.0
            while y1 <= r - y0 + 1e-12 and len(out) < int(max_points):
                y2 = r - y0 - y1
                if y2 < -1e-12:
                    y1 += h
                    continue
                if y2 < 0.0:
                    y2 = 0.0
                x = [float(l[0] + y0), float(l[1] + y1), float(l[2] + y2)]
                res = self.check(
                    x,
                    search=search,
                    seed=seed,
                    max_objections_per_pair=max_objections_per_pair,
                    max_counterobjections_per_pair=max_counterobjections_per_pair,
                )
                out.append((x, bool(res.in_set)))
                y1 += h
            y0 += h
        return out


__all__ = ["BargainingSet", "BargainingCheckResult", "Objection"]
