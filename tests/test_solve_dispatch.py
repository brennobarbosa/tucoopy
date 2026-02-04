import sys
from pathlib import Path
import unittest

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT / "src"))


class TestSolveDispatch(unittest.TestCase):
    def test_solve_gately(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.solutions import solve  # noqa: E402

        g = Game.from_coalitions(
            n_players=3,
            values={
                (): 0.0,
                (0,): 1.0,
                (1,): 1.0,
                (2,): 1.0,
                (0, 1, 2): 10.0,
            },
        )

        res = solve(g, method="gately")
        self.assertEqual(res.method, "gately")
        self.assertEqual(len(res.x), 3)
        self.assertIsNotNone(res.meta)
        self.assertIn("d", res.meta)
        self.assertAlmostEqual(sum(res.x), 10.0, places=6)

    def test_solve_least_squares_requires_x0(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.solutions import solve  # noqa: E402

        g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))

        with self.assertRaises(ValueError):
            solve(g, method="least_squares")

    def test_solve_least_squares_projection(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.solutions import solve  # noqa: E402

        g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
        res = solve(g, method="least_squares", x0=[1.0, 1.0, 1.0])
        self.assertEqual(res.method, "least_squares")
        self.assertAlmostEqual(res.x[0], 1.0, places=9)
        self.assertAlmostEqual(res.x[1], 1.0, places=9)
        self.assertAlmostEqual(res.x[2], 1.0, places=9)
        self.assertIsNotNone(res.meta)
        self.assertTrue(bool(res.meta.get("feasible", False)))

    def test_solve_myerson_requires_edges(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.solutions import solve  # noqa: E402

        g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
        with self.assertRaises(ValueError):
            solve(g, method="myerson")

    def test_solve_myerson_additive(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.solutions import solve  # noqa: E402

        g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
        res = solve(g, method="myerson", edges=[(0, 1)])
        self.assertEqual(res.method, "myerson")
        self.assertAlmostEqual(res.x[0], 1.0, places=6)
        self.assertAlmostEqual(res.x[1], 1.0, places=6)
        self.assertAlmostEqual(res.x[2], 1.0, places=6)

    def test_solve_owen_requires_unions(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.solutions import solve  # noqa: E402

        g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
        with self.assertRaises(ValueError):
            solve(g, method="owen")

    def test_solve_owen_additive(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.solutions import solve  # noqa: E402

        g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
        res = solve(g, method="owen", unions=[[0, 1], [2]])
        self.assertEqual(res.method, "owen")
        self.assertAlmostEqual(res.x[0], 1.0, places=6)
        self.assertAlmostEqual(res.x[1], 1.0, places=6)
        self.assertAlmostEqual(res.x[2], 1.0, places=6)

    def test_solve_least_core_point_selection_validation(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.solutions import solve  # noqa: E402

        g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
        with self.assertRaises(ValueError):
            solve(g, method="least_core_point", selection="nope")


if __name__ == "__main__":
    unittest.main()

