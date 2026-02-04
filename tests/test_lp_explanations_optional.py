import sys
from pathlib import Path
import unittest
from unittest import mock

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT / "src"))


class TestLPExplanationsOptional(unittest.TestCase):
    def test_build_analysis_lp_explanations_no_scipy_does_not_raise(self) -> None:
        from tucoopy import Game  # noqa: E402
        from tucoopy.io import build_analysis  # noqa: E402

        g = Game.from_coalitions(n_players=2, values={(): 0.0, (0, 1): 1.0})

        def fake_import(name: str, *args, **kwargs):
            if name.startswith("scipy"):
                raise ModuleNotFoundError("scipy")
            return __import__(name)

        with mock.patch("importlib.import_module", side_effect=fake_import):
            a = build_analysis(g, max_players=4, include_lp_explanations=True, lp_explanations_max_players=4)
        self.assertIn("meta", a)
        # We should not have raised; if attempted, it should be marked as skipped.
        self.assertIn("lp_explanations", a["meta"]["computed"])


if __name__ == "__main__":
    unittest.main()
