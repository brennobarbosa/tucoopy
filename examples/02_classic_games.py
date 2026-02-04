from __future__ import annotations

"""
Classic toy games: generators + one solution / power index.

This example demonstrates a handful of standard game generators and computes:

- Shapley value for TU games, and
- classic voting power indices for a simple weighted voting game.
"""

from _bootstrap import add_src_to_path

add_src_to_path()

from tucoopy.games.airport import airport_game
from tucoopy.games.bankruptcy import bankruptcy_game
from tucoopy.games.glove import glove_game
from tucoopy.games.unanimity import unanimity_game
from tucoopy.games.weighted_voting import weighted_voting_game
from tucoopy.power.banzhaf import banzhaf_index
from tucoopy.power.shapley_shubik import shapley_shubik_index
from tucoopy.solutions.shapley import shapley_value


def main() -> None:
    print("== Glove game ==")
    g = glove_game([1, 0], [0, 1], unit_value=10.0, player_labels=["L", "R"])
    print("v(N) =", g.value(g.grand_coalition))
    print("Shapley:", shapley_value(g))

    print("\n== Weighted voting (simple game) ==")
    g = weighted_voting_game([2, 1, 1], quota=3, player_labels=["A", "B", "C"])
    print("SSI:", shapley_shubik_index(g))
    print("Banzhaf (normalized):", banzhaf_index(g, normalized=True))

    print("\n== Airport game (cost sharing style; worth = -max requirement) ==")
    g = airport_game([1.0, 3.0, 2.0], player_labels=["P1", "P2", "P3"])
    print("v({P2}) =", g.value(0b010))
    print("Shapley:", shapley_value(g))

    print("\n== Bankruptcy game ==")
    g = bankruptcy_game(estate=100.0, claims=[70.0, 60.0], player_labels=["C1", "C2"])
    print("v(N) =", g.value(g.grand_coalition))
    print("Shapley:", shapley_value(g))

    print("\n== Unanimity game ==")
    g = unanimity_game((0, 2), n_players=3, value=5.0, player_labels=["P1", "P2", "P3"])
    print("v({P1,P3}) =", g.value(0b101))
    print("Shapley:", shapley_value(g))


if __name__ == "__main__":
    main()
