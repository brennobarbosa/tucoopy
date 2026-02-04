import sys
from pathlib import Path
import unittest

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT / "src"))


class TestWeightedValues(unittest.TestCase):
    def test_weighted_shapley_unanimity_n2(self) -> None:
        from tucoopy import Game  # noqa: E402
        from tucoopy.solutions import weighted_shapley_value  # noqa: E402

        g = Game.from_coalitions(
            n_players=2,
            values={
                (): 0.0,
                (0, 1): 1.0,
            },
            require_complete=False,
        )
        x = weighted_shapley_value(g, weights=[2.0, 1.0])
        self.assertAlmostEqual(x[0], 2.0 / 3.0, places=9)
        self.assertAlmostEqual(x[1], 1.0 / 3.0, places=9)

    def test_weighted_shapley_additive(self) -> None:
        from tucoopy import Game  # noqa: E402
        from tucoopy.solutions import weighted_shapley_value  # noqa: E402

        g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
        x = weighted_shapley_value(g, weights=[1.0, 2.0, 3.0])
        self.assertAlmostEqual(x[0], 1.0, places=9)
        self.assertAlmostEqual(x[1], 1.0, places=9)
        self.assertAlmostEqual(x[2], 1.0, places=9)

    def test_semivalue_shapley_weights(self) -> None:
        from tucoopy import Game  # noqa: E402
        from tucoopy.solutions import semivalue, shapley_value  # noqa: E402

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
            require_complete=True,
        )

        # Shapley semivalue weights: p_k = k!(n-k-1)!/n!
        n = g.n_players
        weights_by_k = [1.0 / 3.0, 1.0 / 6.0, 1.0 / 3.0]  # for n=3: k=0,1,2
        x1 = shapley_value(g)
        x2 = semivalue(g, weights_by_k=weights_by_k, normalize=False)
        for i in range(n):
            self.assertAlmostEqual(x1[i], x2[i], places=9)

    def test_weighted_banzhaf_p_half_equals_banzhaf(self) -> None:
        from tucoopy import Game  # noqa: E402
        from tucoopy.solutions import banzhaf_value, weighted_banzhaf_value  # noqa: E402

        g = Game.from_value_function(n_players=4, value_fn=lambda S: float(len(S) ** 2))
        x1 = banzhaf_value(g)
        x2 = weighted_banzhaf_value(g, p=0.5)
        for i in range(g.n_players):
            self.assertAlmostEqual(x1[i], x2[i], places=9)


if __name__ == "__main__":
    unittest.main()
