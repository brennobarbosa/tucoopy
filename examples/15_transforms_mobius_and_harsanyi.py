"""
Transforms: Mobius transform and Harsanyi dividends.

This example shows how to:

- build a small TU game from a tabular characteristic function,
- compute the Mobius transform (unanimity coefficients),
- compute Harsanyi dividends,
- reconstruct the original characteristic function.
"""

from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

from tucoopy import Game  # noqa: E402
from tucoopy.transforms.harsanyi import harsanyi_dividends  # noqa: E402
from tucoopy.transforms.mobius import inverse_mobius_transform, mobius_transform  # noqa: E402


def main() -> None:
    # A small 3-player TU game in tabular form (mask -> value).
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

    # The Mobiüs transform operates on a dense vector indexed by coalition mask.
    values = [g.value(mask) for mask in range(1 << g.n_players)]

    mu = mobius_transform(values, n_players=g.n_players)
    d = harsanyi_dividends(g)
    v_rec = inverse_mobius_transform(mu, n_players=g.n_players)

    print("Mobius coefficients (mask -> mu[mask]):")
    for mask, muS in enumerate(mu):
        if abs(muS) > 1e-12:
            print(f"  {mask:03b}: {muS:.6g}")

    print("\nHarsanyi dividends (mask -> d[mask]):")
    for mask in sorted(d):
        if abs(d[mask]) > 1e-12:
            print(f"  {mask:03b}: {d[mask]:.6g}")

    print("\nReconstruction check (v_rec == v):")
    for mask in range(1 << g.n_players):
        v0 = float(g.value(mask))
        v1 = float(v_rec[mask])
        print(f"  {mask:03b}: v={v0:.6g}  v_rec={v1:.6g}  diff={v1 - v0:+.3e}")


if __name__ == "__main__":
    main()
