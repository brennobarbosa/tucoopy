import sys
from pathlib import Path
import unittest

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT / "src"))


class TestPolyhedralSet(unittest.TestCase):
    def test_extreme_points_unit_square(self) -> None:
        from tucoopy.geometry import PolyhedralSet  # noqa: E402

        # 0 <= x <= 1, 0 <= y <= 1
        A_ub = [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]
        b_ub = [1.0, 0.0, 1.0, 0.0]
        P = PolyhedralSet.from_hrep(A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None)])

        verts = P.extreme_points(tol=1e-9, max_dim=2)
        self.assertEqual(len(verts), 4)
        self.assertTrue(P.contains([0.0, 0.0]))
        self.assertTrue(P.contains([1.0, 1.0]))
        self.assertFalse(P.contains([1.1, 0.0]))

    def test_project_slack_and_residual(self) -> None:
        from tucoopy.geometry import PolyhedralSet  # noqa: E402

        # Simplex in 2D: x>=0, y>=0, x+y<=1.
        P = PolyhedralSet.from_hrep(A_ub=[[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]], b_ub=[0.0, 0.0, 1.0])
        verts = P.extreme_points(tol=1e-9, max_dim=2)
        self.assertEqual(verts, [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

        # Slack for x+y<=1 at (0.25,0.25) is 0.5.
        slacks = P.slack_ub([0.25, 0.25])
        self.assertAlmostEqual(slacks[2], 0.5, places=12)

        # Equality residuals.
        Q = PolyhedralSet.from_hrep(A_eq=[[1.0, 1.0]], b_eq=[1.0], bounds=[(0.0, 1.0), (0.0, 1.0)])
        self.assertEqual(Q.residual_eq([0.25, 0.75]), [0.0])
        self.assertEqual(Q.slack_eq([0.25, 0.75]), [0.0])

        # Projection helper (vertex enumeration + coordinate select).
        C = PolyhedralSet.from_hrep(bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])
        proj = C.project((0, 2), max_dim=3)
        self.assertIn([0.0, 0.0], proj)
        self.assertIn([1.0, 1.0], proj)

    def test_affine_dimension(self) -> None:
        from tucoopy.geometry import PolyhedralSet  # noqa: E402

        # Full-dimensional in 2D.
        P = PolyhedralSet.from_hrep(bounds=[(0.0, 1.0), (0.0, 1.0)])
        self.assertEqual(P.affine_dimension(), 2)

        # Equality x+y=1 in 2D => affine dimension 1.
        Q = PolyhedralSet.from_hrep(A_eq=[[1.0, 1.0]], b_eq=[1.0], bounds=[(0.0, 1.0), (0.0, 1.0)])
        self.assertEqual(Q.affine_dimension(), 1)

        # Fixed point => affine dimension 0.
        R = PolyhedralSet.from_hrep(bounds=[(0.5, 0.5), (0.25, 0.25)])
        self.assertEqual(R.affine_dimension(), 0)

    def test_extreme_points_redundant_constraints(self) -> None:
        from tucoopy.geometry import PolyhedralSet  # noqa: E402

        # Unit square with redundant (duplicate / scaled) inequalities.
        A_ub = [
            [1.0, 0.0],
            [2.0, 0.0],
            [-1.0, 0.0],
            [-2.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, -1.0],
            [0.0, -2.0],
        ]
        b_ub = [1.0, 2.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0]
        P = PolyhedralSet.from_hrep(A_ub=A_ub, b_ub=b_ub)
        verts = P.extreme_points(tol=1e-9, max_dim=2)
        self.assertEqual(verts, [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

    def test_extreme_points_redundant_equalities(self) -> None:
        from tucoopy.geometry import PolyhedralSet  # noqa: E402

        # Segment on x+y=1 with a redundant equality row.
        P = PolyhedralSet.from_hrep(
            A_eq=[[1.0, 1.0], [2.0, 2.0]],
            b_eq=[1.0, 2.0],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
        )
        verts = P.extreme_points(tol=1e-10, max_dim=2)
        self.assertEqual(verts, [[0.0, 1.0], [1.0, 0.0]])

    def test_extreme_points_segment_fixed_bound(self) -> None:
        from tucoopy.geometry import PolyhedralSet  # noqa: E402

        # Segment: x fixed to 0, 0<=y<=1.
        P = PolyhedralSet.from_hrep(bounds=[(0.0, 0.0), (0.0, 1.0)])
        verts = P.extreme_points(tol=1e-12, max_dim=2)
        self.assertEqual(verts, [[0.0, 0.0], [0.0, 1.0]])

    def test_is_empty_and_chebyshev_center_optional(self) -> None:
        try:
            import scipy  # noqa: F401
        except Exception:
            self.skipTest("SciPy not installed (install with tucoopy[lp])")

        from tucoopy.geometry import PolyhedralSet  # noqa: E402

        # Infeasible: x <= 0 and x >= 1
        P = PolyhedralSet.from_hrep(A_ub=[[1.0], [-1.0]], b_ub=[0.0, -1.0], bounds=[(None, None)])
        self.assertTrue(P.is_empty())
        self.assertIsNone(P.sample_point())

        # Unit interval: 0<=x<=1
        Q = PolyhedralSet.from_hrep(A_ub=[[1.0], [-1.0]], b_ub=[1.0, 0.0], bounds=[(None, None)])
        cc = Q.chebyshev_center()
        self.assertIsNotNone(cc)
        x, r = cc  # type: ignore[misc]
        self.assertAlmostEqual(x[0], 0.5, places=6)
        self.assertAlmostEqual(r, 0.5, places=6)

    def test_is_bounded_optional(self) -> None:
        try:
            import scipy  # noqa: F401
        except Exception:
            self.skipTest("SciPy not installed (install with tucoopy[lp])")

        from tucoopy.geometry import PolyhedralSet  # noqa: E402

        P = PolyhedralSet.from_hrep(bounds=[(0.0, 1.0), (0.0, 1.0)])
        self.assertTrue(P.is_bounded())

        # Unbounded ray: x >= 0 with no upper bound.
        Q = PolyhedralSet.from_hrep(A_ub=[[-1.0]], b_ub=[0.0], bounds=[(None, None)])
        self.assertFalse(Q.is_bounded())

    def test_core_and_imputation_wrappers_import(self) -> None:
        from tucoopy import Game  # noqa: E402
        from tucoopy.geometry import Core, EpsilonCore, ImputationSet, PreImputationSet  # noqa: E402

        g = Game.from_coalitions(
            n_players=2,
            values={
                (): 0.0,
                (0,): 0.0,
                (1,): 0.0,
                (0, 1): 1.0,
            },
        )

        self.assertIsNotNone(Core(g).poly)
        self.assertIsNotNone(EpsilonCore(g, epsilon=0.0).poly)
        self.assertIsNotNone(PreImputationSet(g).poly)
        self.assertIsNotNone(ImputationSet(g).poly)
