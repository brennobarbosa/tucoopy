import sys
from pathlib import Path
import unittest

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT / "src"))


class TestDiagnosticsHelpers(unittest.TestCase):
    def test_check_allocation_and_explain(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.diagnostics import check_allocation, explain_core_membership  # noqa: E402

        # Simple additive game: x=[1,1,1] is efficient and in the core.
        g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
        x = [1.0, 1.0, 1.0]
        chk = check_allocation(g, x, tol=1e-9, core_top_k=3)
        self.assertTrue(chk.efficient)
        self.assertTrue(chk.imputation)
        self.assertTrue(chk.core.in_core)

        lines = explain_core_membership(g, x, tol=1e-9, top_k=2)
        self.assertTrue(any("In the core" in s for s in lines))

    def test_core_violations(self) -> None:
        from tucoop import Game  # noqa: E402
        from tucoop.diagnostics import core_violations  # noqa: E402

        # Game where coalition {0,1} has value 2, but x gives them total 1.
        g = Game.from_coalitions(
            n_players=3,
            values={
                (): 0.0,
                (0,): 0.0,
                (1,): 0.0,
                (2,): 0.0,
                (0, 1): 2.0,
                (0, 1, 2): 2.0,
            },
        )
        v = core_violations(g, [0.0, 1.0, 1.0], tol=1e-9)
        self.assertTrue(any(r.coalition_mask == 0b011 for r in v))


if __name__ == "__main__":
    unittest.main()

