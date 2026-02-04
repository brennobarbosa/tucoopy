from __future__ import annotations

"""
Matplotlib viz (n=3): ternary (simplex) plot.

Warning
-------
This example requires Matplotlib at runtime. Install with:
`pip install tucoopy[viz]`.

Some set-valued computations in the plot (core polygon) require an LP backend
at runtime. Install with:
`pip install "tucoopy[lp]"`.
"""

from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

import argparse

from tucoopy.base.game import Game
from tucoopy.viz.mpl3 import plot_ternary


def _resolve_out_dir(out: str) -> Path:
    p = Path(out)
    if p.is_absolute():
        return p
    return Path(__file__).resolve().parent / p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="out",
        help="Output directory for PNG (absolute, or relative to this file). Default: out",
    )
    args = parser.parse_args()

    game = Game.from_coalitions(
        n_players=3,
        values={
            (): 0.0,
            (0,): 1.0,
            (1,): 0.8,
            (2,): 0.7,
            (0, 1): 2.0,
            (0, 2): 1.6,
            (1, 2): 1.5,
            (0, 1, 2): 3.5,
        },
        player_labels=["Alice", "Bob", "Cara"],
    )

    fig, _ = plot_ternary(
        game,
        point_sets={"trajectory": [[0.5, 0.5, 0.5]]},
        show_imputation=True,
        show_core=True,
    )
    fig.suptitle("3-player simplex viz (mpl3)", fontsize=14)
    fig.tight_layout()

    out_dir = _resolve_out_dir(str(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "mpl3-ternary.png", dpi=160, bbox_inches="tight")
    print("== Output ==")
    print("Wrote:", out_dir / "mpl3-ternary.png")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        raise SystemExit(str(e)) from e
    except Exception as e:
        raise SystemExit(f"{type(e).__name__}: {e}") from e
