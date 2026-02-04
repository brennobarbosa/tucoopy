"""
Diagnostics: blocking regions in the imputation simplex (n=3).

Blocking regions are geometric objects that can be plotted on the ternary
diagram. This example prints the computed regions (coalition mask + vertices).
"""

from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

from tucoopy import Game  # noqa: E402
from tucoopy.diagnostics.blocking_regions import blocking_regions  # noqa: E402
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
    br = blocking_regions(g)
    regions = br.regions

    print("x:", x)
    print("coordinate_system:", br.coordinate_system)
    print("regions:", len(regions))
    for r in regions:
        print(f"\ncoalition_mask={r.coalition_mask} vertices:")
        for v in r.vertices:
            print(" ", v)


if __name__ == "__main__":
    main()
