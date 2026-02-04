from __future__ import annotations

"""
Shapley value and the core (small n).

This example builds a small 3-player TU game, computes the Shapley value, and
prints the core vertices (intended for visualization / small instances).
"""

from _bootstrap import add_src_to_path

add_src_to_path()

from tucoop.base.game import Game
from tucoop.geometry.core_set import Core
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

    phi = shapley_value(g)
    print("Shapley value:", phi)

    try:
        import scipy  # noqa: F401
    except Exception:
        print("\nCore vertices require an LP backend (recommended: SciPy).")
        print('Install with: pip install "tucoop[lp]"')
        return

    verts = Core(g).vertices()
    print("\nCore vertices (n=3):")
    for v in verts:
        print(" ", v)


if __name__ == "__main__":
    main()
