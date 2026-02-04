import sys
from pathlib import Path
import unittest

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT / "src"))


class TestCoreDiagnostics(unittest.TestCase):
    def test_in_core_true_for_core_point(self) -> None:
        from tucoopy import Game  # noqa: E402
        from tucoopy.diagnostics import core_diagnostics  # noqa: E402
        from tucoopy.solutions import shapley_value  # noqa: E402

        g = Game.from_coalitions(
            n_players=2,
            values={
                (): 0.0,
                (0,): 0.0,
                (1,): 0.0,
                (0, 1): 1.0,
            },
        )
        x = shapley_value(g)
        d = core_diagnostics(g, x, tol=1e-9, top_k=8)
        self.assertTrue(d.in_core)
        self.assertLessEqual(d.max_excess, 1e-9)
        self.assertEqual(d.n_players, 2)
        self.assertTrue(d.efficient)
        self.assertAlmostEqual(d.sum_x, d.vN, places=9)
        self.assertIsInstance(d.tight_coalitions, list)
        self.assertIsInstance(d.violations, list)

    def test_in_core_false_has_violations(self) -> None:
        from tucoopy import Game  # noqa: E402
        from tucoopy.diagnostics import core_diagnostics  # noqa: E402

        g = Game.from_coalitions(
            n_players=3,
            values={
                (): 0.0,
                (0,): 0.0,
                (1,): 0.0,
                (2,): 0.0,
                (0, 1): 2.0,
                (0, 2): 2.0,
                (1, 2): 2.0,
                (0, 1, 2): 3.0,
            },
        )
        x = [1.0, 1.0, 1.0]  # efficient (sum=3), but pairs can deviate (v=2 > xS=2?) actually tight.
        d = core_diagnostics(g, x, tol=1e-9, top_k=8)
        self.assertTrue(d.in_core)

        y = [0.5, 0.5, 2.0]  # efficient, but coalition (0,1) has v=2 > xS=1.0
        d2 = core_diagnostics(g, y, tol=1e-9, top_k=8)
        self.assertFalse(d2.in_core)
        self.assertGreater(d2.max_excess, 0.0)
        self.assertGreaterEqual(len(d2.tight_coalitions), 1)
        self.assertGreaterEqual(len(d2.violations), 1)
        top = d2.violations[0]
        self.assertIn(top.coalition_mask, (3,))  # (0,1)
        self.assertEqual(top.players, [0, 1])


if __name__ == "__main__":
    unittest.main()
