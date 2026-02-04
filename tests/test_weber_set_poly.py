from __future__ import annotations

from tucoopy.base.game import Game
from tucoopy.geometry import WeberSet


def _assert_point_satisfies_hrep(
    poly, x: list[float], *, tol: float = 1e-8  # PolyhedralSet, but keep loose typing in tests
) -> None:
    for row, rhs in zip(poly.A_eq, poly.b_eq):
        lhs = sum(float(a) * float(v) for a, v in zip(row, x))
        assert abs(lhs - float(rhs)) <= tol

    for row, rhs in zip(poly.A_ub, poly.b_ub):
        lhs = sum(float(a) * float(v) for a, v in zip(row, x))
        assert lhs <= float(rhs) + tol

    assert len(poly.bounds) == len(x)
    for xi, (lb, ub) in zip(x, poly.bounds):
        if lb is not None:
            assert float(xi) >= float(lb) - tol
        if ub is not None:
            assert float(xi) <= float(ub) + tol


def test_weber_set_poly_n2_contains_all_marginals() -> None:
    g = Game.from_coalitions(
        n_players=2,
        values={
            (): 0.0,
            (0,): 0.0,
            (1,): 0.0,
            (0, 1): 1.0,
        },
    )
    W = WeberSet(g, max_permutations=10)
    poly = W.poly

    pts = W.points()
    assert len(pts) >= 2
    for x in pts:
        _assert_point_satisfies_hrep(poly, x, tol=1e-8)


def test_weber_set_poly_n3_contains_all_marginals() -> None:
    g = Game.from_coalitions(
        n_players=3,
        values={
            (): 0.0,
            (0,): 1.0,
            (1,): 1.0,
            (2,): 1.0,
            (0, 1): 2.0,
            (0, 2): 2.0,
            (1, 2): 2.0,
            (0, 1, 2): 4.0,
        },
    )
    W = WeberSet(g, max_permutations=2000)
    poly = W.poly

    pts = W.points()
    assert len(pts) > 0
    for x in pts:
        _assert_point_satisfies_hrep(poly, x, tol=1e-7)

    assert len(poly.A_eq) >= 1


def test_weber_set_poly_n_gt_3_not_implemented() -> None:
    g = Game.from_coalitions(
        n_players=4,
        values={
            (): 0.0,
            (0, 1, 2, 3): 1.0,
        },
    )
    W = WeberSet(g)
    try:
        _ = W.poly
    except NotImplementedError:
        pass
    else:
        raise AssertionError("expected NotImplementedError for WeberSet.poly when n_players > 3")
