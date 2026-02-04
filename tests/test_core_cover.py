from pathlib import Path
import sys
import unittest

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT / "src"))


class TestCoreCover(unittest.TestCase):
    def test_reasonable_set_vertices_additive_game(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.geometry import ReasonableSet  # noqa: E402

        g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
        verts = ReasonableSet(g).vertices(max_dim=4)
        # Additive game => M_i = v({i}) = 1, so only one feasible point.
        self.assertEqual(verts, [[1.0, 1.0, 1.0]])

    def test_core_cover_vertices_additive_game(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.geometry import CoreCover  # noqa: E402

        g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
        verts = CoreCover(g).vertices(max_dim=4)
        self.assertEqual(verts, [[1.0, 1.0, 1.0]])

    def test_build_analysis_includes_core_cover_and_reasonable(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.io import build_analysis  # noqa: E402

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
        a = build_analysis(g, max_players=4, include_sets=True, include_weber=False, include_solutions=False, include_diagnostics=False)
        self.assertIn("sets", a)
        self.assertIn("core_cover", a["sets"])
        self.assertIn("reasonable", a["sets"])
        self.assertIn("vertices", a["sets"]["core_cover"])
        self.assertIn("bounds", a["sets"]["core_cover"])
        self.assertIn("vertices", a["sets"]["reasonable"])
        self.assertIn("bounds", a["sets"]["reasonable"])


if __name__ == "__main__":
    unittest.main()
