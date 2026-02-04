import sys
from pathlib import Path
import unittest

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT / "src"))


class TestTheoryCanonic(unittest.TestCase):
    def test_additive_game_core_equals_imputation_singleton(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.geometry import Core, ImputationSet  # noqa: E402

        weights = [1.0, 2.0, 3.0]

        def v(players) -> float:
            return float(sum(weights[int(i)] for i in players))

        g = Game.from_value_function(n_players=3, value_fn=v)
        x_star = list(weights)

        C = Core(g)
        I = ImputationSet(g)

        self.assertTrue(C.contains(x_star))
        self.assertTrue(I.contains(x_star))

        core_vertices = C.vertices(max_players=3, max_dim=3)
        imp_vertices = I.vertices(max_players=3, max_dim=3)

        self.assertEqual(core_vertices, [x_star])
        self.assertEqual(imp_vertices, [x_star])

    def test_convex_game_shapley_in_core(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.geometry import Core  # noqa: E402
        from tucoop.solutions import shapley_value  # noqa: E402

        # Symmetric convex / supermodular example (n=3):
        # v(i)=0, v(ij)=1, v(123)=3
        g = Game.from_coalitions(
            n_players=3,
            values={
                0: 0.0,
                1: 0.0,
                2: 0.0,
                4: 0.0,
                3: 1.0,
                5: 1.0,
                6: 1.0,
                7: 3.0,
            },
        )

        phi = shapley_value(g)
        self.assertEqual(len(phi), 3)

        C = Core(g)
        self.assertTrue(C.contains(phi))

    def test_least_core_epsilon_zero_iff_core_nonempty_when_lp_available(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.base.exceptions import MissingOptionalDependencyError  # noqa: E402

        try:
            import numpy  # noqa: F401
            import scipy.optimize  # noqa: F401
        except Exception:
            self.skipTest("requires tucoop[lp] (SciPy + NumPy)")

        from tucoop.solutions import least_core  # noqa: E402

        # Core non-empty (additive): epsilon* should be 0.
        g_nonempty = Game.from_value_function(n_players=3, value_fn=lambda ps: float(len(ps)))
        res_nonempty = least_core(g_nonempty)
        self.assertAlmostEqual(res_nonempty.epsilon, 0.0, places=9)

        # Core empty (3-player majority game): epsilon* should be > 0.
        def majority(ps) -> float:
            return 1.0 if len(ps) >= 2 else 0.0

        g_empty = Game.from_value_function(n_players=3, value_fn=majority)
        res_empty = least_core(g_empty)
        self.assertGreater(res_empty.epsilon, 0.0)

        # Controlled failure: if the backend disappears, error should be explicit.
        # (This branch rarely triggers in CI, but it documents intended behavior.)
        try:
            _ = least_core(g_empty)
        except MissingOptionalDependencyError as e:  # pragma: no cover
            self.assertIn("tucoop[lp]", str(e))

    def test_least_core_requires_lp_extra_message(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.base.exceptions import MissingOptionalDependencyError  # noqa: E402
        from tucoop.solutions import least_core  # noqa: E402

        g = Game.from_coalitions(n_players=2, values={0: 0.0, 3: 1.0})

        # In the minimal install (no SciPy/NumPy), this should fail with a clear hint.
        try:
            _ = least_core(g)
        except MissingOptionalDependencyError as e:
            self.assertIn("tucoop[lp]", str(e))
        except ImportError as e:
            # Backward-compatible: MissingOptionalDependencyError subclasses ImportError,
            # but tests should accept plain ImportError with the same message.
            self.assertIn("tucoop[lp]", str(e))


if __name__ == "__main__":
    unittest.main()
