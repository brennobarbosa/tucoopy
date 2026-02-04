"""
Diagnostics: core membership and excess scan.

This example computes:

- max excess and tight coalitions for a candidate allocation,
- a compact explanation suitable for logging / UI.
"""

from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

from tucoopy import Game  # noqa: E402
from tucoopy.diagnostics.core_diagnostics import (  # noqa: E402
    explain_core_membership,
    is_in_core,
    max_excess,
    tight_coalitions,
)
from tucoopy.solutions.shapley import shapley_value  # noqa: E402


def main() -> None:
    g = Game.from_coalitions(
        n_players=3,
        values={
            0: 0.0,
            1: 1.0,
            2: 1.2,
            4: 0.8,
            3: 2.8,
            5: 2.2,
            6: 2.0,
            7: 4.0,
        },
        player_labels=["P1", "P2", "P3"],
    )

    x = shapley_value(g)
    print("x (Shapley):", x)
    print("in_core:", is_in_core(g, x))
    print("max_excess:", max_excess(g, x))
    print("tight coalitions (mask):", tight_coalitions(g, x))

    exp = explain_core_membership(g, x, top_k=5)
    print("\nExplanation:")
    for line in exp:
        print("-", line)


if __name__ == "__main__":
    main()

