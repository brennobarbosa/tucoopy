import sys
from pathlib import Path
import unittest
from unittest import mock

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT / "src"))


class TestSolutionsNew(unittest.TestCase):
    def test_gately_point_additive(self) -> None:
        from tucoopy import Game  # noqa: E402
        from tucoopy.solutions import gately_point  # noqa: E402

        # Essential game: v(N) > sum v({i})
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
        res = gately_point(g)
        self.assertEqual(len(res.x), 3)
        self.assertAlmostEqual(sum(res.x), 4.0, places=9)

    def test_owen_value_respects_symmetry_in_blocks(self) -> None:
        from tucoopy import Game  # noqa: E402
        from tucoopy.solutions import owen_value  # noqa: E402

        # Additive game v(S)=|S| => Owen equals (1,1,1,1) regardless of unions.
        g = Game.from_value_function(n_players=4, value_fn=lambda S: float(len(S)))
        res = owen_value(g, unions=[[0, 1], [2, 3]], require_complete=False)
        for i in range(4):
            self.assertAlmostEqual(res.x[i], 1.0, places=9)

    def test_myerson_value_requires_complete_by_default(self) -> None:
        from tucoopy import Game  # noqa: E402
        from tucoopy.solutions import myerson_value  # noqa: E402

        g = Game.from_coalitions(n_players=3, values={(): 0.0, (0, 1, 2): 1.0}, require_complete=False)
        with self.assertRaises(ValueError):
            myerson_value(g, edges=[(0, 1)])

    def test_least_core_point_missing_numpy_message(self) -> None:
        from tucoopy import Game  # noqa: E402
        from tucoopy.solutions import least_core_point  # noqa: E402

        g = Game.from_coalitions(
            n_players=2,
            values={
                (): 0.0,
                (0,): 0.0,
                (1,): 0.0,
                (0, 1): 1.0,
            },
        )
        with mock.patch("importlib.import_module", side_effect=ModuleNotFoundError("numpy")):
            with self.assertRaises(ImportError) as ctx:
                least_core_point(g, method="any_feasible")
        self.assertIn("tucoopy[lp]", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
