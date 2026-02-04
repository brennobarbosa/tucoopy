from __future__ import annotations

"""
KernelSet and BargainingSet (set-valued objects).

This example checks whether a candidate allocation belongs to:

- the kernel set (sampling-based),
- the bargaining set (may require an LP backend).
"""

from _bootstrap import add_src_to_path

add_src_to_path()

from tucoop.base.game import Game
from tucoop.geometry.bargaining_set import BargainingSet
from tucoop.geometry.kernel_set import KernelSet
from tucoop.solutions.shapley import shapley_value


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

    x = shapley_value(g)
    print("Candidate allocation (Shapley):", x)

    ks = KernelSet(g)
    br = BargainingSet(g)

    print("In KernelSet?", ks.contains(x))
    try:
        print("In BargainingSet?", br.contains(x))
    except Exception as e:
        print("BargainingSet.contains failed (likely missing LP backend).")
        print(f"Error: {type(e).__name__}: {e}")

    print("\nSample kernel points (for visualization):")
    for p in ks.sample_points(n_samples=5, seed=0):
        print("  ", p)

    print("\nSample bargaining points (for visualization):")
    try:
        for p in br.sample_points(n_samples=5, seed=0):
            print("  ", p)
    except Exception as e:
        print("BargainingSet.sample_points failed (likely missing LP backend).")
        print(f"Error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
