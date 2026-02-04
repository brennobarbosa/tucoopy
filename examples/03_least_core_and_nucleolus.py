from __future__ import annotations

"""
Least-core and nucleolus (optional LP backend).

Warning
-------
This example requires an LP backend at runtime (recommended: SciPy).
Install with: `pip install \"tucoop[lp]\"`.
"""

from _bootstrap import add_src_to_path

add_src_to_path()


def main() -> None:
    try:
        import scipy  # noqa: F401
    except Exception:
        print("This example requires SciPy. Install with: pip install \"tucoop[lp]\"")
        return

    from tucoop.base.game import Game
    from tucoop.solutions.least_core import least_core
    from tucoop.solutions.nucleolus import nucleolus, prenucleolus

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
    )

    lc = least_core(g)
    nu = nucleolus(g)
    pnu = prenucleolus(g)

    print("Least-core epsilon:", lc.epsilon)
    print("Least-core x:", lc.x)
    print("Nucleolus x:", nu.x, "levels:", nu.levels)
    print("Pre-nucleolus x:", pnu.x, "levels:", pnu.levels)


if __name__ == "__main__":
    main()
