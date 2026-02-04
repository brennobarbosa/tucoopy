import sys
from pathlib import Path
import unittest

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT / "src"))

from tucoop.games import (  # noqa: E402
    airport_game,
    apex_game,
    bankruptcy_game,
    glove_game,
    savings_game,
    unanimity_game,
    weighted_voting_game,
)
from tucoop.solutions import (  # noqa: E402
    banzhaf_value,
    normalized_banzhaf_value,
    shapley_value,
)
from tucoop.power import (  # noqa: E402
    banzhaf_index_weighted_voting,
    banzhaf_index,
    coleman_initiate_index,
    coleman_prevent_index,
    deegan_packel_index,
    holler_index,
    johnston_index,
    rae_index,
    shapley_shubik_index,
    shapley_shubik_index_weighted_voting,
)
from tucoop.properties import validate_simple_game  # noqa: E402
from tucoop.geometry import marginal_vector, weber_marginal_vectors  # noqa: E402


class TestGames(unittest.TestCase):
    def test_glove_game(self) -> None:
        g = glove_game([1, 0], [0, 1], unit_value=10.0)
        # Alone: no pairs.
        self.assertAlmostEqual(g.value(0b01), 0.0)
        self.assertAlmostEqual(g.value(0b10), 0.0)
        # Together: 1 pair * 10
        self.assertAlmostEqual(g.value(0b11), 10.0)

    def test_weighted_voting_game(self) -> None:
        g = weighted_voting_game([2, 1, 1], quota=3)
        validate_simple_game(g)
        self.assertAlmostEqual(g.value(0b001), 0.0)
        self.assertAlmostEqual(g.value(0b011), 1.0)  # 2+1 meets quota
        self.assertAlmostEqual(g.value(0b110), 0.0)  # 1+1 fails

        # Power indices for this classic example.
        ssi = shapley_shubik_index(g)
        self.assertAlmostEqual(ssi[0], 2 / 3, places=9)
        self.assertAlmostEqual(ssi[1], 1 / 6, places=9)
        self.assertAlmostEqual(ssi[2], 1 / 6, places=9)

        bzi = banzhaf_index(g, normalized=True)
        self.assertAlmostEqual(bzi[0], 0.6, places=9)
        self.assertAlmostEqual(bzi[1], 0.2, places=9)
        self.assertAlmostEqual(bzi[2], 0.2, places=9)

        # DP variants (integer weights) should match.
        ssi_dp = shapley_shubik_index_weighted_voting([2, 1, 1], 3)
        self.assertAlmostEqual(ssi_dp[0], 2 / 3, places=9)
        self.assertAlmostEqual(ssi_dp[1], 1 / 6, places=9)
        self.assertAlmostEqual(ssi_dp[2], 1 / 6, places=9)

        bzi_dp = banzhaf_index_weighted_voting([2, 1, 1], 3, normalized=True)
        self.assertAlmostEqual(bzi_dp[0], 0.6, places=9)
        self.assertAlmostEqual(bzi_dp[1], 0.2, places=9)
        self.assertAlmostEqual(bzi_dp[2], 0.2, places=9)

        # Minimal-winning based indices (complete simple games).
        dp = deegan_packel_index(g)
        hi = holler_index(g)
        ji = johnston_index(g)
        self.assertAlmostEqual(sum(dp), 1.0, places=9)
        self.assertAlmostEqual(sum(hi), 1.0, places=9)
        self.assertAlmostEqual(sum(ji), 1.0, places=9)

        cprev = coleman_prevent_index(g)
        cinit = coleman_initiate_index(g)
        self.assertAlmostEqual(cprev[0], 1.0, places=9)
        self.assertAlmostEqual(cprev[1], 1 / 3, places=9)
        self.assertAlmostEqual(cprev[2], 1 / 3, places=9)
        self.assertAlmostEqual(cinit[0], 3 / 5, places=9)
        self.assertAlmostEqual(cinit[1], 1 / 5, places=9)
        self.assertAlmostEqual(cinit[2], 1 / 5, places=9)

        rae = rae_index(g)
        self.assertAlmostEqual(rae[0], 7 / 8, places=9)
        self.assertAlmostEqual(rae[1], 5 / 8, places=9)
        self.assertAlmostEqual(rae[2], 5 / 8, places=9)

        hi_raw = holler_index(g, normalized=False)
        self.assertEqual(hi_raw, [2.0, 1.0, 1.0])
        hi_norm = holler_index(g, normalized=True)
        self.assertAlmostEqual(sum(hi_norm), 1.0, places=9)
        for i in range(3):
            self.assertAlmostEqual(hi_norm[i], hi[i], places=9)

    def test_validate_simple_game_rejects_non_simple(self) -> None:
        # Additive TU game is not simple (values are not restricted to {0,1}).
        g = weighted_voting_game([1, 1], quota=2)
        validate_simple_game(g)  # ok

        g2 = g.with_values({0: 0.0, 1: 0.2, 2: 0.0, 3: 1.0})
        with self.assertRaises(ValueError):
            validate_simple_game(g2)

    def test_airport_game_sign(self) -> None:
        g = airport_game([1.0, 3.0, 2.0])
        # Worth is negative of max requirement
        self.assertAlmostEqual(g.value(0b001), -1.0)
        self.assertAlmostEqual(g.value(0b010), -3.0)
        self.assertAlmostEqual(g.value(0b111), -3.0)

    def test_bankruptcy_game(self) -> None:
        g = bankruptcy_game(estate=100.0, claims=[70.0, 60.0])
        # v({1}) = max(0, E - c2) = 40
        self.assertAlmostEqual(g.value(0b01), 40.0)
        # v({2}) = max(0, E - c1) = 30
        self.assertAlmostEqual(g.value(0b10), 30.0)
        # v(N) = E
        self.assertAlmostEqual(g.value(0b11), 100.0)

    def test_unanimity_game(self) -> None:
        g = unanimity_game((0, 2), n_players=3, value=5.0)
        self.assertAlmostEqual(g.value(0b001), 0.0)
        self.assertAlmostEqual(g.value(0b101), 5.0)
        self.assertAlmostEqual(g.value(0b111), 5.0)

    def test_apex_game(self) -> None:
        # Apex player 0 must be in coalition and others must reach quota 2.
        g = apex_game(apex_player=0, weights=[0, 1, 1], quota=2)
        self.assertAlmostEqual(g.value(0b110), 0.0)  # no apex
        self.assertAlmostEqual(g.value(0b111), 1.0)  # apex + 1+1 meets quota
        self.assertAlmostEqual(g.value(0b101), 0.0)  # apex + 1 fails

    def test_savings_game(self) -> None:
        # n=2
        c = [5.0, 6.0]
        # coalition costs C(S) indexed by mask: 0,1,2,3
        C = [0.0, 5.0, 6.0, 9.0]  # together is cheaper than 5+6
        g = savings_game(c, C)
        self.assertAlmostEqual(g.value(0b01), 0.0)
        self.assertAlmostEqual(g.value(0b10), 0.0)
        self.assertAlmostEqual(g.value(0b11), 2.0)  # savings 11 - 9

    def test_weber_marginal_vectors_simple_game(self) -> None:
        # Majority game: quota=2 over 3 players => grand coalition worth 1.
        g = weighted_voting_game([1, 1, 1], quota=2)
        mv = weber_marginal_vectors(g, max_permutations=10)
        self.assertEqual(len(mv), 6)  # 3! permutations
        for x in mv:
            self.assertAlmostEqual(sum(x), 1.0, places=9)

        # Spot-check one permutation [0,1,2]:
        x012 = marginal_vector(g, [0, 1, 2])
        # v({0})=0, v({0,1})=1, v({0,1,2})=1 => [0,1,0] in that order
        self.assertAlmostEqual(x012[0], 0.0, places=9)
        self.assertAlmostEqual(x012[1], 1.0, places=9)
        self.assertAlmostEqual(x012[2], 0.0, places=9)

        # Symmetry: all players should get the same value under these indices.
        dp = deegan_packel_index(g)
        hi = holler_index(g)
        ji = johnston_index(g)
        for i in range(3):
            self.assertAlmostEqual(dp[i], 1 / 3, places=9)
            self.assertAlmostEqual(hi[i], 1 / 3, places=9)
            self.assertAlmostEqual(ji[i], 1 / 3, places=9)


if __name__ == "__main__":
    unittest.main()
