import sys
from pathlib import Path
import unittest

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT / "src"))


class TestKernelSet(unittest.TestCase):
    def test_kernel_set_membership_additive(self) -> None:
        from tucoopy import Game  # noqa: E402
        from tucoopy.geometry import KernelSet, PreKernelSet  # noqa: E402

        g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
        x = [1.0, 1.0, 1.0]

        pk = PreKernelSet(g, max_players=6)
        self.assertTrue(pk.contains(x, tol=1e-9))
        self.assertTrue(pk.check(x, tol=1e-9).in_set)

        k = KernelSet(g, max_players=6)
        self.assertTrue(k.contains(x, tol=1e-9))
        self.assertTrue(k.check(x, tol=1e-9).in_set)

    def test_kernel_set_argmax_masks_and_pair_order(self) -> None:
        from tucoopy import Game  # noqa: E402
        from tucoopy.geometry import KernelSet  # noqa: E402

        g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
        x = [1.0, 1.0, 1.0]
        k = KernelSet(g, max_players=6)
        res = k.check(x, tol=1e-9, top_k=10)
        self.assertTrue(res.in_set)

        # Ties everywhere => sorted by (i, j).
        self.assertGreaterEqual(len(res.pairs), 3)
        self.assertEqual([(p.i, p.j) for p in res.pairs[:3]], [(0, 1), (0, 2), (1, 2)])

        N = (1 << 3) - 1
        for p in res.pairs:
            self.assertAlmostEqual(p.delta, 0.0, places=12)

            self.assertGreaterEqual(p.argmax_ij, 0)
            self.assertLessEqual(p.argmax_ij, N)
            self.assertGreaterEqual(p.argmax_ji, 0)
            self.assertLessEqual(p.argmax_ji, N)

            self.assertTrue(bool(p.argmax_ij & (1 << p.i)))
            self.assertFalse(bool(p.argmax_ij & (1 << p.j)))
            self.assertTrue(bool(p.argmax_ji & (1 << p.j)))
            self.assertFalse(bool(p.argmax_ji & (1 << p.i)))

    def test_kernel_set_rejects_non_imputation(self) -> None:
        from tucoopy import Game  # noqa: E402
        from tucoopy.geometry import KernelSet  # noqa: E402

        g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
        # sum=3 but violates IR (x1 < v({1}) = 1).
        x = [2.0, 0.5, 0.5]
        k = KernelSet(g, max_players=6)
        res = k.check(x, tol=1e-9)
        self.assertFalse(res.imputation)
        self.assertFalse(res.in_set)

    def test_kernel_set_sampling_singleton(self) -> None:
        from tucoopy import Game  # noqa: E402
        from tucoopy.geometry import KernelSet  # noqa: E402

        g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
        k = KernelSet(g, max_players=6)
        pts = k.sample_points(n_samples=20, seed=123, max_points=10, tol=1e-9)
        self.assertEqual(len(pts), 1)
        self.assertAlmostEqual(pts[0][0], 1.0, places=9)
        self.assertAlmostEqual(pts[0][1], 1.0, places=9)
        self.assertAlmostEqual(pts[0][2], 1.0, places=9)

    def test_kernel_set_max_players_guard(self) -> None:
        from tucoopy import Game  # noqa: E402
        from tucoopy.geometry import KernelSet  # noqa: E402

        g = Game.from_value_function(n_players=5, value_fn=lambda S: float(len(S)))
        k = KernelSet(g, max_players=4)
        with self.assertRaises(ValueError):
            k.check([1.0] * 5, tol=1e-9)


if __name__ == "__main__":
    unittest.main()
