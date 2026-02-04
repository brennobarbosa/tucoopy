"""
Geometry: compute vertices for polyhedral sets (optional LP backend).

This example is intentionally small and is meant for "sanity checking" the
polyhedral construction of:

- Core,
- Core cover,
- Reasonable set.

Warning
-------
This example requires an LP backend at runtime.
Install with: `pip install "tucoopy[lp]"`.
"""

from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()


def main() -> None:
    try:
        import scipy  # noqa: F401
    except Exception:
        print('This example requires SciPy. Install with: pip install "tucoopy[lp]"')
        return

    from tucoopy import Game  # noqa: E402
    from tucoopy.geometry.core_set import Core  # noqa: E402
    from tucoopy.geometry.core_cover_set import CoreCover  # noqa: E402
    from tucoopy.geometry.reasonable_set import ReasonableSet  # noqa: E402

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

    core = Core(g)
    cc = CoreCover(g)
    rs = ReasonableSet(g)

    print("Core vertices:")
    for v in core.vertices():
        print(" ", v)

    print("\nCore cover vertices:")
    for v in cc.vertices():
        print(" ", v)

    print("\nReasonable set vertices:")
    for v in rs.vertices():
        print(" ", v)


if __name__ == "__main__":
    main()

