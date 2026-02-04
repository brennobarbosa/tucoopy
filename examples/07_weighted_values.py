from __future__ import annotations

"""
Weighted values and semivalues.

This example demonstrates:

- weighted Shapley value,
- semivalue with user-chosen size weights $p_k$,
- weighted Banzhaf value.
"""

from _bootstrap import add_src_to_path

add_src_to_path()

from tucoopy.base.game import Game
from tucoopy.solutions.banzhaf import weighted_banzhaf_value
from tucoopy.solutions.shapley import semivalue, weighted_shapley_value


def main() -> None:
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
        player_labels=["P1", "P2", "P3"],
    )

    print("Weighted Shapley (weights=[1,2,1]):")
    print(weighted_shapley_value(g, weights=[1.0, 2.0, 1.0]))

    print("\nSemivalue (weights_by_k=[1,1,1], normalize=True):")
    print(semivalue(g, weights_by_k=[1.0, 1.0, 1.0], normalize=True))

    print("\nWeighted Banzhaf (p=0.6):")
    print(weighted_banzhaf_value(g, p=0.6))


if __name__ == "__main__":
    main()
