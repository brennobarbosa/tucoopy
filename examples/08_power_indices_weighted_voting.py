from __future__ import annotations

"""
Power indices for a weighted voting game.

This example computes several classic power indices for a weighted voting game
and also demonstrates the shortcuts that apply directly to a weighted voting
representation.
"""

from _bootstrap import add_src_to_path

add_src_to_path()

from tucoop.games.weighted_voting import weighted_voting_game
from tucoop.power.banzhaf import banzhaf_index, banzhaf_index_weighted_voting
from tucoop.power.coleman import coleman_collectivity_power_to_act, coleman_prevent_index
from tucoop.power.deegan_packel import deegan_packel_index
from tucoop.power.holler import holler_index
from tucoop.power.johnston import johnston_index
from tucoop.power.rae import rae_index
from tucoop.power.shapley_shubik import (
    shapley_shubik_index,
    shapley_shubik_index_weighted_voting,
)


def main() -> None:
    weights = [4, 3, 2, 1]
    quota = 6

    g = weighted_voting_game(weights, quota=quota, player_labels=["A", "B", "C", "D"])

    print(f"Weighted voting game: weights={weights}, quota={quota}")
    print("Shapley-Shubik:", shapley_shubik_index(g))
    print("Banzhaf:", banzhaf_index(g))
    print("Johnston:", johnston_index(g))
    print("Deegan-Packel:", deegan_packel_index(g))
    print("Holler:", holler_index(g))
    print("Rae:", rae_index(g))
    print("Coleman (power to act):", coleman_collectivity_power_to_act(g))
    print("Coleman (prevent):", coleman_prevent_index(g))

    print("\nShortcuts for weighted voting games:")
    print(
        "Shapley-Shubik (weighted voting shortcut):",
        shapley_shubik_index_weighted_voting(weights, quota),
    )
    print(
        "Banzhaf (weighted voting shortcut):",
        banzhaf_index_weighted_voting(weights, quota),
    )


if __name__ == "__main__":
    main()
