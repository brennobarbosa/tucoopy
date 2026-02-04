import sys
from pathlib import Path
import unittest
from unittest import mock

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT / "src"))


class TestEdgeCases(unittest.TestCase):
    def test_prekernel_max_iter_zero_does_not_crash(self) -> None:
        try:
            import numpy  # noqa: F401
        except Exception:
            self.skipTest("NumPy not installed (install with tucoopy[fast])")

        from tucoopy import Game  # noqa: E402
        from tucoopy.solutions import prekernel  # noqa: E402

        g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
        res = prekernel(g, max_iter=0)
        self.assertEqual(res.iterations, 0)
        self.assertEqual(len(res.x), 3)

    def test_lp_normalize_bounds_variants(self) -> None:
        # Internal helper: stable, deterministic, and does not require SciPy/PuLP.
        from tucoopy.backends import lp as lp_mod  # noqa: E402

        self.assertEqual(lp_mod._normalize_bounds(None, 2), [(0.0, None), (0.0, None)])
        self.assertEqual(lp_mod._normalize_bounds((None, None), 2), [(None, None), (None, None)])
        self.assertEqual(lp_mod._normalize_bounds((0.0, 1.0), 2), [(0.0, 1.0), (0.0, 1.0)])
        self.assertEqual(lp_mod._normalize_bounds([(0.0, None), (1.0, 2.0)], 2), [(0.0, None), (1.0, 2.0)])
        self.assertEqual(lp_mod._normalize_bounds([0.0, 1.0], 2), [(0.0, 1.0), (0.0, 1.0)])

        with self.assertRaises(ValueError):
            lp_mod._normalize_bounds([(0.0, 1.0)], 2)

        with self.assertRaises(ValueError):
            lp_mod._normalize_bounds([0.0, 1.0, 2.0], 2)

    def test_balancedness_requires_fun(self) -> None:
        # Ensures our defensive check around `res.fun` is exercised.
        try:
            import numpy as np  # noqa: F401
        except Exception:
            self.skipTest("NumPy not installed (install with tucoopy[lp] or tucoopy[fast])")

        from tucoopy import Game  # noqa: E402
        from tucoopy.properties.balancedness import balancedness_check  # noqa: E402

        g = Game.from_coalitions(
            n_players=2,
            values={
                (): 0.0,
                (0,): 0.0,
                (1,): 0.0,
                (0, 1): 1.0,
            },
        )

        class _FakeRes:
            x = [0.0, 0.0]
            fun = None

        def _fake_linprog(*args, **kwargs):
            return _FakeRes()

        # Patch the LP adapter to return a result without `fun`.
        with mock.patch("tucoopy.backends.lp.linprog_solve", side_effect=_fake_linprog):
            with self.assertRaises(RuntimeError) as ctx:
                balancedness_check(g)
        self.assertIn("objective value", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

