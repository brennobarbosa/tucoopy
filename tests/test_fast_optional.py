import sys
from pathlib import Path
import unittest
from unittest import mock

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT / "src"))


class TestFastOptional(unittest.TestCase):
    def test_require_numpy_message(self) -> None:
        from tucoopy.backends.numpy_fast import require_numpy  # noqa: E402

        with mock.patch("importlib.import_module", side_effect=ModuleNotFoundError("numpy")):
            with self.assertRaises(ImportError) as ctx:
                require_numpy(context="kernel")
        self.assertIn("tucoopy[fast]", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
