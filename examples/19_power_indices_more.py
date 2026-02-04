"""
Power indices: a small "dashboard" for voting games.

This example computes several classic power indices for a weighted voting game.
"""

from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

from tucoopy.games.weighted_voting import weighted_voting_game  # noqa: E402
from tucoopy.power.banzhaf import banzhaf_index  # noqa: E402
from tucoopy.power.coleman import coleman_initiate_index, coleman_prevent_index  # noqa: E402
from tucoopy.power.deegan_packel import deegan_packel_index  # noqa: E402
from tucoopy.power.holler import holler_index  # noqa: E402
from tucoopy.power.johnston import johnston_index  # noqa: E402
from tucoopy.power.rae import rae_index  # noqa: E402
from tucoopy.power.shapley_shubik import shapley_shubik_index  # noqa: E402


def main() -> None:
    g = weighted_voting_game(weights=[4, 3, 2, 1], quota=6, player_labels=["A", "B", "C", "D"])

    print("Game:", "weights=[4,3,2,1], quota=6")
    print("SSI:", shapley_shubik_index(g))
    print("Banzhaf (normalized):", banzhaf_index(g, normalized=True))
    print("Rae:", rae_index(g))
    print("Holler:", holler_index(g))
    print("Deegan-Packel:", deegan_packel_index(g))
    print("Johnston:", johnston_index(g))
    print("Coleman (to act):", coleman_initiate_index(g))
    print("Coleman (to prevent):", coleman_prevent_index(g))


if __name__ == "__main__":
    main()

