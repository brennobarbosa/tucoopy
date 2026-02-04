import sys
from pathlib import Path
import unittest
from unittest import mock

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT / "src"))


class TestGeometryNewModules(unittest.TestCase):
    def test_sample_imputation_set(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.geometry import sample_imputation_set  # noqa: E402

        g = Game.from_coalitions(
            n_players=3,
            values={
                (): 0.0,
                (0,): 1.0,
                (1,): 2.0,
                (2,): 0.5,
                (0, 1, 2): 10.0,
            },
        )
        pts = sample_imputation_set(g, n_samples=5, seed=123)
        self.assertEqual(len(pts), 5)
        for x in pts:
            self.assertEqual(len(x), 3)
            self.assertAlmostEqual(sum(x), 10.0, places=9)
            self.assertGreaterEqual(x[0] + 1e-12, 1.0)
            self.assertGreaterEqual(x[1] + 1e-12, 2.0)
            self.assertGreaterEqual(x[2] + 1e-12, 0.5)

    def test_project_allocation_n3(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.geometry import project_allocation  # noqa: E402

        g = Game.from_coalitions(
            n_players=3,
            values={
                (): 0.0,
                (0,): 0.0,
                (1,): 0.0,
                (2,): 0.0,
                (0, 1, 2): 1.0,
            },
        )
        x = [1 / 3, 1 / 3, 1 / 3]
        p = project_allocation(g, x)
        self.assertEqual(len(p), 2)

    def test_least_core_missing_numpy_message(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.geometry import LeastCore  # noqa: E402

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
                _ = LeastCore(g).epsilon
        self.assertIn("tucoop[lp]", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
