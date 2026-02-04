from __future__ import annotations

"""
Matplotlib viz (n=2): segment plot.

Warning
-------
This example requires Matplotlib at runtime. Install with:
`pip install tucoopy[viz]`.

Some set-valued computations in the plot (imputation/core segments) may require
an LP backend at runtime. Install with:
`pip install "tucoopy[lp]"`.
"""

from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

import argparse

from tucoopy.base.game import Game
from tucoopy.viz.mpl2 import plot_segment


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
        n_players=2,
        values={
            (): 0.0,
            (0,): 2.25,
            (1,): 2.0,
            (0, 1): 3.0,
        },
        player_labels=["Alice", "Bob"],
    )

    fig, _ = plot_segment(
        game,
        show_imputation=True,
        show_core=True,
        points=[[1.0, 2.0], [0.5, 2.5]],
        point_sets={"trajectory": [[0.1, 2.9], [0.4, 2.4]]},
    )
    fig.suptitle("2-player segment viz (mpl2)", fontsize=14)
    fig.tight_layout()

    out_dir = _resolve_out_dir(str(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "mpl2-segment.png", dpi=160, bbox_inches="tight")
    print("== Output ==")
    print("Wrote:", out_dir / "mpl2-segment.png")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        raise SystemExit(str(e)) from e
    except Exception as e:
        raise SystemExit(f"{type(e).__name__}: {e}") from e
