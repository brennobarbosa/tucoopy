import sys
from pathlib import Path
import unittest

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT / "src"))


class TestBlockingRegions(unittest.TestCase):
    def test_blocking_regions_n3_returns_barycentric_polygons(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.diagnostics import blocking_regions  # noqa: E402

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

        res = blocking_regions(g)
        self.assertEqual(res.coordinate_system, "barycentric_imputation")
        # We may have some empty regions, but for this game we should find at least one polygon.
        self.assertGreaterEqual(len(res.regions), 1)

        for r in res.regions:
            self.assertIsInstance(r.coalition_mask, int)
            self.assertGreaterEqual(len(r.vertices), 3)
            for b in r.vertices:
                self.assertEqual(len(b), 3)
                s = sum(b)
                self.assertAlmostEqual(s, 1.0, places=6)
                for bi in b:
                    self.assertGreaterEqual(bi, -1e-6)

    def test_blocking_regions_non_n3_is_empty(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.diagnostics import blocking_regions  # noqa: E402

        g2 = Game.from_coalitions(
            n_players=2,
            values={
                (): 0.0,
                (0,): 0.0,
                (1,): 0.0,
                (0, 1): 1.0,
            },
        )
        res2 = blocking_regions(g2)
        self.assertEqual(res2.coordinate_system, "barycentric_imputation")
        self.assertEqual(res2.regions, [])


if __name__ == "__main__":
    unittest.main()
