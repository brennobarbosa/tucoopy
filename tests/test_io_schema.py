import sys
from pathlib import Path
import unittest

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT / "src"))


class TestIoSchema(unittest.TestCase):
    def test_animation_spec_schema_loads(self) -> None:
        from tucoop.io import animation_spec_schema  # noqa: E402

        s = animation_spec_schema()
        self.assertIsInstance(s, dict)
        self.assertEqual(s.get("type"), "object")
        self.assertIn("properties", s)

    def test_game_schema_loads(self) -> None:
        from tucoop.io import game_schema  # noqa: E402

        s = game_schema()
        self.assertIsInstance(s, dict)
        self.assertEqual(s.get("type"), "object")
        self.assertIn("properties", s)

    def test_build_animation_spec_smoke(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.io.animation_spec import build_animation_spec  # noqa: E402
        from tucoop.io import animation_spec_schema  # noqa: E402

        g = Game.from_coalitions(n_players=2, values={(): 0.0, (0, 1): 1.0})
        spec = build_animation_spec(
            g,
            series_id="x",
            allocations=[[0.0, 1.0]],
            dt=1 / 60,
            include_analysis=True,
        )
        self.assertEqual(spec.schema_version, "0.1.0")

        schema = animation_spec_schema()
        self.assertIn("properties", schema)


if __name__ == "__main__":
    unittest.main()
