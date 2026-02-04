import sys
from pathlib import Path
import unittest

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT / "src"))

from tucoop import Game  # noqa: E402
from tucoop.properties import is_convex  # noqa: E402
from tucoop.diagnostics import is_in_core  # noqa: E402
from tucoop.geometry import Core, EpsilonCore, ImputationSet  # noqa: E402
from tucoop.solutions import banzhaf_value, shapley_value, shapley_value_sample, tau_value  # noqa: E402
from tucoop.transforms import harsanyi_dividends  # noqa: E402


class TestBasic(unittest.TestCase):
    def test_shapley_sum_efficiency(self) -> None:
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
        phi = shapley_value(g)
        self.assertAlmostEqual(sum(phi), 4.0, places=9)

    def test_shapley_sample_additive_is_exact(self) -> None:
        g = Game.from_value_function(n_players=6, value_fn=lambda S: float(len(S)))
        mean, stderr = shapley_value_sample(g, n_samples=25, seed=123)
        for i in range(6):
            self.assertAlmostEqual(mean[i], 1.0, places=9)
            self.assertAlmostEqual(stderr[i], 0.0, places=9)

    def test_banzhaf_length(self) -> None:
        g = Game.from_coalitions(n_players=2, values={(): 0.0, (0,): 1.0, (1,): 1.0, (0, 1): 3.0})
        beta = banzhaf_value(g)
        self.assertEqual(len(beta), 2)

    def test_harsanyi_reconstruct(self) -> None:
        g = Game.from_coalitions(
            n_players=2,
            values={
                (): 0.0,
                (0,): 1.0,
                (1,): 2.0,
                (0, 1): 5.0,
            },
        )
        d = harsanyi_dividends(g)
        # Reconstruct v(S) = sum_{T subset S} d(T)
        for S in range(1 << g.n_players):
            total = 0.0
            T = S
            while True:
                total += d.get(T, 0.0)
                if T == 0:
                    break
                T = (T - 1) & S
            self.assertAlmostEqual(total, g.value(S), places=9)

    def test_core_vertices_small(self) -> None:
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
        verts = Core(g).extreme_points(max_dim=6)
        self.assertGreaterEqual(len(verts), 1)

    def test_is_convex_trivial(self) -> None:
        # Additive game is convex.
        g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
        self.assertTrue(is_convex(g))

    def test_tau_value_additive(self) -> None:
        g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
        x = tau_value(g)
        self.assertAlmostEqual(sum(x), 3.0, places=9)
        self.assertTrue(is_in_core(g, x))

    def test_imputation_vertices(self) -> None:
        g = Game.from_coalitions(
            n_players=3,
            values={
                (): 0.0,
                (0,): 1.0,
                (1,): 2.0,
                (2,): 0.5,
                (0, 1, 2): 10.0,
            },
        )
        verts = ImputationSet(g).vertices()
        # For n=3 and r>0, simplex has 3 vertices.
        self.assertEqual(len(verts), 3)
        for v in verts:
            self.assertAlmostEqual(sum(v), 10.0, places=9)
            self.assertGreaterEqual(v[0] + 1e-12, 1.0)
            self.assertGreaterEqual(v[1] + 1e-12, 2.0)
            self.assertGreaterEqual(v[2] + 1e-12, 0.5)

    def test_epsilon_core_vertices_matches_core_when_eps0(self) -> None:
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
        c = Core(g).extreme_points(max_dim=6)
        e0 = EpsilonCore(g, 0.0).vertices(max_players=6)
        self.assertEqual(len(c), len(e0))


if __name__ == "__main__":
    unittest.main()
