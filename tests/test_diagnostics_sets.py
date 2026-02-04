import sys
from pathlib import Path
import unittest

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT / "src"))


class TestDiagnosticsSets(unittest.TestCase):
    def test_epsilon_core_diagnostics(self) -> None:
        from tucoopy import Game  # noqa: E402
        from tucoopy.diagnostics.epsilon_core_diagnostics import epsilon_core_diagnostics  # noqa: E402
        from tucoopy.solutions import shapley_value  # noqa: E402

        # Simple 2-player game with a non-empty core.
        g = Game.from_coalitions(
            n_players=2,
            values={
                (): 0.0,
                (0,): 0.5,
                (1,): 0.5,
                (0, 1): 1.0,
            },
        )
        x = shapley_value(g)
        d0 = epsilon_core_diagnostics(g, x, epsilon=0.0, tol=1e-9, top_k=8)
        self.assertTrue(d0.efficient)
        self.assertTrue(d0.in_set)
        self.assertLessEqual(d0.max_excess, 1e-9)
        self.assertIsInstance(d0.tight_coalitions, list)
        self.assertIsInstance(d0.violations, list)

        y = [0.9, 0.1]  # efficient but violates singleton constraints since v({1})=0.5 > 0.1
        d1 = epsilon_core_diagnostics(g, y, epsilon=0.0, tol=1e-9, top_k=8)
        self.assertTrue(d1.efficient)
        self.assertFalse(d1.in_set)
        self.assertGreater(d1.max_excess, 0.0)
        self.assertGreaterEqual(len(d1.violations), 1)

    def test_reasonable_set_diagnostics(self) -> None:
        from tucoopy import Game  # noqa: E402
        from tucoopy.diagnostics.reasonable_diagnostics import reasonable_set_diagnostics  # noqa: E402

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
                (0, 1, 2): 3.0,
            },
        )

        x = [1.0, 1.0, 1.0]
        d = reasonable_set_diagnostics(g, x, tol=1e-9)
        self.assertTrue(d.efficient)
        self.assertTrue(d.in_set)
        self.assertEqual(d.violations, [])

        # Inefficient point should fail even if within bounds.
        y = [1.0, 1.0, 0.0]
        d2 = reasonable_set_diagnostics(g, y, tol=1e-9)
        self.assertFalse(d2.efficient)
        self.assertFalse(d2.in_set)

    def test_least_core_diagnostics_is_diagnostic_friendly(self) -> None:
        from tucoopy import Game  # noqa: E402
        from tucoopy.diagnostics.least_core_diagnostics import least_core_diagnostics  # noqa: E402

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
        x = [1.0, 1.0, 1.0]
        d = least_core_diagnostics(g, x, tol=1e-9, top_k=8)

        # Works both with and without an LP backend.
        self.assertIsInstance(d.available, bool)
        if d.available:
            self.assertIsNotNone(d.epsilon_star)
            self.assertIsNotNone(d.epsilon_core)
        else:
            self.assertIsInstance(d.reason, str)
            self.assertIsNone(d.epsilon_star)
            self.assertIsNone(d.epsilon_core)


if __name__ == "__main__":
    unittest.main()
