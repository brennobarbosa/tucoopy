import sys
from pathlib import Path
import unittest
from unittest import mock

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT / "src"))


class TestLinprogAdapter(unittest.TestCase):
    def test_missing_scipy_message(self) -> None:
        from tucoop.backends.lp import linprog_solve  # noqa: E402

        with mock.patch("importlib.import_module", side_effect=ModuleNotFoundError("scipy")):
            with self.assertRaises(ImportError) as ctx:
                linprog_solve([0.0], bounds=[(0.0, 1.0)], require_success=False)
        self.assertIn("tucoop[lp]", str(ctx.exception))

    def test_linprog_solve_success(self) -> None:
        try:
            import scipy  # noqa: F401
        except Exception:
            self.skipTest("SciPy not installed (install with tucoop[lp])")

        import numpy as np  # noqa: E402
        from tucoop.backends.lp import linprog_solve  # noqa: E402

        # Minimize x subject to 0 <= x <= 1, so optimum is x=0.
        c = np.array([1.0], dtype=float)
        res = linprog_solve(c, bounds=[(0.0, 1.0)], context="test")
        self.assertTrue(getattr(res, "success", False))
        x = float(res.x.tolist()[0])
        self.assertAlmostEqual(x, 0.0, places=9)

    def test_missing_pulp_message(self) -> None:
        from tucoop.backends.lp import linprog_solve  # noqa: E402

        with mock.patch("importlib.import_module", side_effect=ModuleNotFoundError("pulp")):
            with self.assertRaises(ImportError) as ctx:
                linprog_solve([0.0], bounds=[(0.0, 1.0)], backend="pulp", require_success=False)
        self.assertIn("tucoop[lp_alt]", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
