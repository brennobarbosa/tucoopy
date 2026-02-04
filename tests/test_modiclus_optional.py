import sys
from pathlib import Path
import unittest

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT / "src"))


class TestModiclusOptional(unittest.TestCase):
    def test_modiclus_additive(self) -> None:
        try:
            import scipy  # noqa: F401
        except Exception:
            self.skipTest("SciPy not installed (install with tucoopy[lp])")

        from tucoopy import Game  # noqa: E402
        from tucoopy.solutions import modiclus  # noqa: E402

        g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
        res = modiclus(g, tol=1e-9, max_players=6, require_complete=True)
        self.assertAlmostEqual(res.x[0], 1.0, places=6)
        self.assertAlmostEqual(res.x[1], 1.0, places=6)
        self.assertAlmostEqual(res.x[2], 1.0, places=6)
        self.assertEqual(res.levels, sorted(res.levels))

    def test_modiclus_require_complete_rejects_incomplete(self) -> None:
        try:
            import scipy  # noqa: F401
        except Exception:
            self.skipTest("SciPy not installed (install with tucoopy[lp])")

        from tucoopy import Game  # noqa: E402
        from tucoopy.solutions import modiclus  # noqa: E402

        g = Game.from_coalitions(n_players=3, values={(): 0.0, (0, 1, 2): 1.0})
        with self.assertRaises(ValueError):
            modiclus(g, require_complete=True)

    def test_modiclus_max_players_guard(self) -> None:
        try:
            import scipy  # noqa: F401
        except Exception:
            self.skipTest("SciPy not installed (install with tucoopy[lp])")

        from tucoopy import Game  # noqa: E402
        from tucoopy.solutions import modiclus  # noqa: E402

        g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
        with self.assertRaises(ValueError):
            modiclus(g, max_players=2, require_complete=False)

    def test_solve_modiclus_dispatch(self) -> None:
        try:
            import scipy  # noqa: F401
        except Exception:
            self.skipTest("SciPy not installed (install with tucoopy[lp])")

        from tucoopy import Game  # noqa: E402
        from tucoopy.solutions import solve  # noqa: E402

        g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
        res = solve(g, method="modiclus")
        self.assertEqual(res.method, "modiclus")
        self.assertEqual(len(res.x), 3)


if __name__ == "__main__":
    unittest.main()
