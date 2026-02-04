from __future__ import annotations

"""
Modiclus (optional LP backend).

This example computes the modiclus of a small TU game.

Warning
-------
This example requires an LP backend at runtime (recommended: SciPy).
Install with: `pip install \"tucoop[lp]\"`.
"""

from _bootstrap import add_src_to_path

add_src_to_path()

from tucoop.base.game import Game
from tucoop.solutions.modiclus import modiclus


def main() -> None:
    try:
        import scipy  # noqa: F401
    except Exception:
        print('This example requires SciPy. Install with: pip install "tucoop[lp]"')
        return

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

    res = modiclus(g)
    print("Modiclus x:", res.x)
    print("Levels:", res.levels)


if __name__ == "__main__":
    main()
