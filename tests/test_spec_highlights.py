import sys
from pathlib import Path
import unittest

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT / "src"))


class TestSeriesHighlights(unittest.TestCase):
    def test_series_from_allocations_with_highlights(self) -> None:
        from tucoopy.io import series_from_allocations  # noqa: E402

        allocs = [[0.0, 1.0], [0.5, 0.5]]
        hs = [{"k": 1}, None]
        s = series_from_allocations(series_id="x", allocations=allocs, dt=1.0, highlights=hs)
        self.assertEqual(len(s.frames), 2)
        self.assertEqual(s.frames[0].highlights, {"k": 1})
        self.assertIsNone(s.frames[1].highlights)

    def test_series_from_allocations_with_highlight_fn(self) -> None:
        from tucoopy.io import series_from_allocations  # noqa: E402

        allocs = [[0.0, 1.0], [0.5, 0.5]]

        def fn(idx: int, _alloc):
            return {"i": idx}

        s = series_from_allocations(series_id="x", allocations=allocs, dt=1.0, highlight_fn=fn)
        self.assertEqual(s.frames[0].highlights, {"i": 0})
        self.assertEqual(s.frames[1].highlights, {"i": 1})

    def test_series_from_allocations_highlights_length_mismatch(self) -> None:
        from tucoopy.io import series_from_allocations  # noqa: E402

        allocs = [[0.0, 1.0], [0.5, 0.5]]
        with self.assertRaises(ValueError):
            series_from_allocations(series_id="x", allocations=allocs, dt=1.0, highlights=[{"k": 1}])

    def test_series_from_allocations_highlights_and_fn_conflict(self) -> None:
        from tucoopy.io import series_from_allocations  # noqa: E402

        allocs = [[0.0, 1.0]]
        with self.assertRaises(ValueError):
            series_from_allocations(
                series_id="x",
                allocations=allocs,
                dt=1.0,
                highlights=[None],
                highlight_fn=lambda _i, _a: None,
            )


if __name__ == "__main__":
    unittest.main()
