import sys
from pathlib import Path
import unittest

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT / "src"))


class TestBuildAnimationSpec(unittest.TestCase):
    def test_build_animation_spec_includes_analysis_and_highlights(self) -> None:
        from tucoopy import Game  # noqa: E402
        from tucoopy.io import build_animation_spec  # noqa: E402

        g = Game.from_coalitions(
            n_players=3,
            values={
                (): 0.0,
                (0,): 1.0,
                (1,): 1.0,
                (2,): 1.0,
                (0, 1, 2): 3.0,
            },
        )

        allocs = [[1.0, 1.0, 1.0], [0.0, 0.0, 3.0]]
        spec = build_animation_spec(
            g,
            series_id="demo",
            allocations=allocs,
            dt=0.5,
            include_analysis=True,
            analysis_kwargs={"max_players": 4},
            include_frame_diagnostics=True,
            frame_diagnostics_max_players=4,
        )

        self.assertEqual(spec.schema_version, "0.1.0")
        self.assertIsNotNone(spec.analysis)
        self.assertIn("meta", spec.analysis)
        self.assertEqual(len(spec.series), 1)
        self.assertEqual(len(spec.series[0].frames), 2)
        h = spec.series[0].frames[0].highlights
        self.assertIsNotNone(h)
        self.assertIn("diagnostics", h)
        self.assertIn("core", h["diagnostics"])


if __name__ == "__main__":
    unittest.main()
