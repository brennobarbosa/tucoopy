import sys
from pathlib import Path
import unittest

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT / "src"))


class TestLPOptional(unittest.TestCase):
    def test_least_core_and_nucleolus_import(self) -> None:
        try:
            import scipy  # noqa: F401
        except Exception:
            self.skipTest("SciPy not installed (install with tucoop[lp])")

        from tucoop import Game  # noqa: E402
        from tucoop.solutions import least_core, nucleolus  # noqa: E402

        g = Game.from_coalitions(
            n_players=3,
            values={
                (): 0.0,
                (0,): 1.0,
                (1,): 1.2,
                (2,): 0.8,
                (0, 1): 2.8,
                (0, 2): 2.2,
                (1, 2): 2.0,
                (0, 1, 2): 4.0,
            },
        )

        lc = least_core(g)
        self.assertEqual(len(lc.x), 3)
        self.assertIsNotNone(lc.tight)

        nu = nucleolus(g)
        self.assertEqual(len(nu.x), 3)
        # Efficiency sanity check
        self.assertAlmostEqual(sum(nu.x), 4.0, places=6)
        self.assertIsNotNone(nu.lp_rounds)
        self.assertEqual(len(nu.lp_rounds), len(nu.levels))

        from tucoop.io import build_analysis  # noqa: E402

        a = build_analysis(g, max_players=4, include_lp_explanations=True, lp_explanations_max_players=4)
        self.assertIn("lp", a["diagnostics"])
        self.assertIn("balancedness_check", a["diagnostics"]["lp"])
        self.assertIn("least_core", a["diagnostics"]["lp"])
        self.assertIn("lp_explanation", a["diagnostics"]["lp"]["balancedness_check"])
        self.assertIn("lp_explanation", a["diagnostics"]["lp"]["least_core"])

    def test_balancedness_check(self) -> None:
        try:
            import scipy  # noqa: F401
        except Exception:
            self.skipTest("SciPy not installed (install with tucoop[lp])")

        from tucoop import Game  # noqa: E402
        from tucoop.properties import balancedness_check  # noqa: E402

        # Core-empty example:
        # v(i)=0, v(ij)=1, v(123)=1 -> infeasible (pair constraints imply 2>=3).
        g = Game.from_coalitions(
            n_players=3,
            values={
                (): 0.0,
                (0,): 0.0,
                (1,): 0.0,
                (2,): 0.0,
                (0, 1): 1.0,
                (0, 2): 1.0,
                (1, 2): 1.0,
                (0, 1, 2): 1.0,
            },
        )

        res = balancedness_check(g)
        self.assertFalse(res.core_nonempty)
        self.assertGreater(res.objective, 1.0)

    def test_prenucleolus_vs_nucleolus_additive(self) -> None:
        try:
            import scipy  # noqa: F401
        except Exception:
            self.skipTest("SciPy not installed (install with tucoop[lp])")

        from tucoop import Game  # noqa: E402
        from tucoop.solutions import nucleolus, prenucleolus  # noqa: E402

        g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
        nu = nucleolus(g)
        pnu = prenucleolus(g)
        self.assertAlmostEqual(nu.x[0], 1.0, places=6)
        self.assertAlmostEqual(pnu.x[0], 1.0, places=6)
        self.assertIsNotNone(nu.lp_rounds)
        self.assertIsNotNone(pnu.lp_rounds)
        self.assertEqual(nu.levels, sorted(nu.levels))
        self.assertEqual(pnu.levels, sorted(pnu.levels))

    def test_nucleolus_rejects_empty_imputation_set(self) -> None:
        try:
            import scipy  # noqa: F401
        except Exception:
            self.skipTest("SciPy not installed (install with tucoop[lp])")

        from tucoop import Game  # noqa: E402
        from tucoop.solutions import nucleolus  # noqa: E402

        # sum v({i}) = 3 > v(N) = 2 -> imputation set is empty.
        g = Game.from_coalitions(
            n_players=3,
            values={
                (): 0.0,
                (0,): 1.0,
                (1,): 1.0,
                (2,): 1.0,
                (0, 1, 2): 2.0,
            },
        )
        with self.assertRaises(ValueError):
            nucleolus(g, tol=1e-9)

    def test_prekernel_additive(self) -> None:
        try:
            import numpy  # noqa: F401
        except Exception:
            self.skipTest("NumPy not installed (install with tucoop[fast])")

        from tucoop import Game  # noqa: E402
        from tucoop.solutions import prekernel  # noqa: E402

        # Additive game: v(S)=|S| has unique imputation/core point (1,1,1).
        g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
        res = prekernel(g, tol=1e-9, max_iter=50)
        self.assertAlmostEqual(res.x[0], 1.0, places=6)
        self.assertAlmostEqual(res.x[1], 1.0, places=6)
        self.assertAlmostEqual(res.x[2], 1.0, places=6)

    def test_kernel_additive(self) -> None:
        try:
            import numpy  # noqa: F401
        except Exception:
            self.skipTest("NumPy not installed (install with tucoop[fast])")

        from tucoop import Game  # noqa: E402
        from tucoop.solutions import kernel  # noqa: E402

        g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
        res = kernel(g, tol=1e-9, max_iter=50)
        self.assertAlmostEqual(res.x[0], 1.0, places=6)
        self.assertAlmostEqual(res.x[1], 1.0, places=6)
        self.assertAlmostEqual(res.x[2], 1.0, places=6)

    def test_bargaining_set_additive(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.geometry import BargainingSet  # noqa: E402

        # Additive game: imputation set is a singleton (1,1,1), should be in bargaining set.
        g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
        x = [1.0, 1.0, 1.0]
        res = BargainingSet(g, n_max=4).check(x)
        self.assertTrue(res.in_set)

        samp = BargainingSet(g, n_max=4).sample_points(n_samples=5, seed=123)
        # r=0 => either [] or [x]; for additive it should return [x]
        self.assertEqual(len(samp), 1)
        self.assertAlmostEqual(samp[0][0], 1.0, places=9)
        self.assertAlmostEqual(samp[0][1], 1.0, places=9)
        self.assertAlmostEqual(samp[0][2], 1.0, places=9)

    def test_bargaining_set_rejects_n_gt_nmax(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.geometry import BargainingSet  # noqa: E402

        g = Game.from_value_function(n_players=5, value_fn=lambda S: float(len(S)))
        bs = BargainingSet(g, n_max=4)
        with self.assertRaises(ValueError):
            bs.contains([1.0] * 5)

    def test_bargaining_set_scan_imputation_grid_optional(self) -> None:
        try:
            import scipy  # noqa: F401
        except Exception:
            self.skipTest("SciPy not installed (install with tucoop[lp])")

        from tucoop import Game  # noqa: E402
        from tucoop.geometry import BargainingSet, ImputationSet  # noqa: E402

        g = Game.from_coalitions(
            n_players=3,
            values={
                (): 0.0,
                (0,): 1.0,
                (1,): 1.2,
                (2,): 0.8,
                (0, 1): 2.8,
                (0, 2): 2.2,
                (1, 2): 2.0,
                (0, 1, 2): 4.0,
            },
        )
        bs = BargainingSet(g, n_max=4, tol=1e-9)
        grid = bs.scan_imputation_grid(
            step=0.5,
            max_points=100,
            max_objections_per_pair=1,
            max_counterobjections_per_pair=1,
            seed=0,
        )
        self.assertGreater(len(grid), 0)
        imp = ImputationSet(g)
        for x, _ in grid:
            self.assertTrue(imp.contains(x, tol=1e-8))


if __name__ == "__main__":
    unittest.main()
