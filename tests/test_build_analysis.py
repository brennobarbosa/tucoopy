import sys
from pathlib import Path
import unittest

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT / "src"))


class TestBuildAnalysis(unittest.TestCase):
    def test_build_analysis_includes_solutions_and_sets(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.io import build_analysis  # noqa: E402

        g = Game.from_coalitions(
            n_players=2,
            values={
                (): 0.0,
                (0,): 1.0,
                (1,): 1.0,
                (0, 1): 3.0,
            },
            player_labels=["P1", "P2"],
        )

        a = build_analysis(g, include_blocking_regions=True, max_players=4)
        self.assertIn("solutions", a)
        self.assertIn("diagnostics", a)
        self.assertIn("input", a["diagnostics"])
        self.assertIn("solutions", a["diagnostics"])
        self.assertIn("shapley", a["diagnostics"]["solutions"])
        self.assertIn("core", a["diagnostics"]["solutions"]["shapley"])
        self.assertIn("tight_coalitions", a["diagnostics"]["solutions"]["shapley"]["core"])
        self.assertIn("explanation", a["diagnostics"]["solutions"]["shapley"]["core"])
        self.assertIsInstance(a["diagnostics"]["solutions"]["shapley"]["core"]["explanation"], list)
        self.assertIn("sets", a)
        self.assertIn("shapley", a["solutions"])
        self.assertIn("normalized_banzhaf", a["solutions"])
        self.assertIn("imputation", a["sets"])
        self.assertIn("core", a["sets"])

        # n=2: no blocking region output
        self.assertNotIn("blocking_regions", a)
        self.assertIn("meta", a)
        self.assertIn("build_analysis", a["meta"])
        self.assertIn("limits", a["meta"])
        self.assertIn("computed", a["meta"])
        self.assertTrue(a["meta"]["computed"]["solutions"])
        self.assertTrue(a["meta"]["computed"]["diagnostics"])

    def test_build_analysis_blocking_regions_n3(self) -> None:
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

        a = build_analysis(g, include_blocking_regions=True, max_players=4)
        self.assertIn("blocking_regions", a)
        br = a["blocking_regions"]
        self.assertEqual(br["coordinate_system"], "barycentric_imputation")
        self.assertIsInstance(br["regions"], list)

    def test_build_analysis_bundle_for_large_n(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.io import build_analysis  # noqa: E402

        g = Game.from_coalitions(
            n_players=6,
            values={
                (): 0.0,
                (0, 1, 2, 3, 4, 5): 10.0,
            },
            player_labels=[f"P{i+1}" for i in range(6)],
        )

        a = build_analysis(g, max_players=4, include_bundle=True)
        self.assertIn("bundle", a)
        self.assertIn("game_summary", a["bundle"])
        self.assertFalse(a["meta"]["computed"]["solutions"])
        self.assertIn("solutions", a["meta"]["skipped"])
        self.assertIn("tables", a["bundle"])
        self.assertIn("players", a["bundle"]["tables"])
        self.assertEqual(len(a["bundle"]["tables"]["players"]), 6)

    def test_build_analysis_bundle_power_indices_simple_complete(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.io import build_analysis  # noqa: E402

        # Simple game with complete v(S): winning iff |S| >= 4 (n=6).
        g = Game.from_value_function(
            n_players=6,
            value_fn=lambda S: 1.0 if len(S) >= 4 else 0.0,
        )
        a = build_analysis(
            g,
            max_players=4,
            include_bundle=True,
            bundle_max_players=8,
            bundle_include_power_indices=True,
            bundle_include_tau=True,
            include_sets=False,
            include_solutions=False,
            include_diagnostics=False,
        )
        b = a["bundle"]
        self.assertIn("tables", b)
        self.assertIn("power_indices", b["tables"])
        self.assertEqual(len(b["tables"]["power_indices"]), 6)
        row0 = b["tables"]["power_indices"][0]
        self.assertIn("shapley_shubik", row0)
        self.assertIn("banzhaf", row0)
        self.assertIn("deegan_packel", row0)
        self.assertIn("holler", row0)
        self.assertIn("johnston", row0)
        self.assertIn("coleman_prevent", row0)
        self.assertIn("coleman_initiate", row0)
        self.assertIn("tau_vectors", b["tables"])

    def test_build_analysis_simple_game_monotonicity(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.io import build_analysis  # noqa: E402

        # Not monotone: singleton is winning, but grand coalition is losing.
        g = Game.from_coalitions(
            n_players=2,
            values={
                (): 0.0,
                (0,): 1.0,
                (1,): 0.0,
                (0, 1): 0.0,
            },
        )
        a = build_analysis(g, max_players=4, include_sets=False, include_solutions=False)
        inp = a["diagnostics"]["input"]
        self.assertTrue(inp["simple_game"])
        self.assertFalse(inp["monotone_simple_game"])
        self.assertIsNotNone(inp["monotone_counterexample"])

    def test_build_analysis_diagnostics_truncation(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.io import build_analysis  # noqa: E402

        g = Game.from_coalitions(
            n_players=12,
            values={(): 0.0, tuple(range(12)): 1.0},
        )
        a = build_analysis(g, max_players=12, diagnostics_max_list=10, include_sets=False, include_solutions=False)
        inp = a["diagnostics"]["input"]
        self.assertTrue(inp["missing_coalition_masks_truncated"])
        self.assertEqual(len(inp["missing_coalition_masks"]), 10)
        self.assertGreater(inp["missing_coalition_mask_count"], 10)

    def test_build_analysis_max_points_truncates_weber(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.io import build_analysis  # noqa: E402

        g = Game.from_value_function(n_players=4, value_fn=lambda S: float(len(S)))
        a = build_analysis(
            g,
            max_players=4,
            include_sets=True,
            include_weber=True,
            max_points=2,
            include_solutions=False,
            include_diagnostics=False,
            include_blocking_regions=False,
        )
        w = a["sets"]["weber"]
        self.assertEqual(len(w["points"]), 2)
        self.assertTrue(w["meta"]["truncated"])
        self.assertEqual(w["meta"]["count_total"], 24)

    def test_build_analysis_bundle_includes_approx_shapley(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.io import build_analysis  # noqa: E402

        g = Game.from_value_function(n_players=6, value_fn=lambda S: float(len(S)))
        a = build_analysis(
            g,
            max_players=4,
            include_bundle=True,
            bundle_max_players=8,
            bundle_include_approx_solutions=True,
            bundle_shapley_samples=50,
            bundle_seed=123,
            include_sets=False,
            include_solutions=False,
            include_diagnostics=False,
        )
        b = a["bundle"]
        self.assertIn("tables", b)
        self.assertIn("approx_solutions", b["tables"])
        self.assertIn("shapley", b["tables"]["approx_solutions"])
        s = b["tables"]["approx_solutions"]["shapley"]
        self.assertEqual(len(s["allocation"]), 6)
        self.assertEqual(len(s["stderr"]), 6)
        self.assertEqual(s["meta"]["n_samples"], 50)
        self.assertEqual(s["meta"]["seed"], 123)
        self.assertTrue(a["meta"]["approx"])
        self.assertIsInstance(a["meta"]["approx_reasons"], list)


if __name__ == "__main__":
    unittest.main()
