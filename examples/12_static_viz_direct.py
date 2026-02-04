from __future__ import annotations

"""
Static Matplotlib visualization without AnimationSpec JSON.

This example builds games directly and renders:

- n=2: segment plot,
- n=3: ternary plot (with a Weber point cloud).

Warning
-------
This example requires Matplotlib at runtime. Install with:
`pip install tucoopy[viz]`.

Some set-valued computations (vertices / core-like polytopes) require an LP
backend at runtime. Install with:
`pip install "tucoopy[lp]"`.
"""

import sys
from pathlib import Path
import argparse

# Ensure the examples folder (for `_bootstrap.py`) is importable even when this file is
# executed from outside `packages/tucoopy-py/examples`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import add_src_to_path

add_src_to_path()

from tucoopy.base.game import Game
from tucoopy.geometry.core_cover_set import CoreCover
from tucoopy.geometry.imputation_set import ImputationSet
from tucoopy.geometry.reasonable_set import ReasonableSet
from tucoopy.geometry.weber_set import weber_marginal_vectors
from tucoopy.solutions.banzhaf import normalized_banzhaf_value
from tucoopy.solutions.shapley import shapley_value
from tucoopy.solutions.tau import tau_value
from tucoopy.viz.mpl2 import plot_segment
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
        help="Output directory for PNGs (absolute, or relative to this file). Default: out",
    )
    args = parser.parse_args()

    out_dir = _resolve_out_dir(str(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- n=2 (segment) ---
    g2 = Game.from_coalitions(
        n_players=2,
        values={
            0: 0.0,
            1: 1.0,
            2: 0.8,
            3: 3.0,
        },
        player_labels=["P1", "P2"],
    )

    sets2 = {}
    try:
        sets2 = {
            "imputation": ImputationSet(g2).vertices(),
            "core_cover": CoreCover(g2).vertices(),
            "reasonable": ReasonableSet(g2).vertices(),
        }
    except Exception as e:
        print("Could not compute polyhedral set vertices (likely missing LP backend).")
        print(f"Error: {type(e).__name__}: {e}")
    sols2 = {
        "shapley": shapley_value(g2),
        "normalized_banzhaf": normalized_banzhaf_value(g2),
        "tau": tau_value(g2),
    }
    fig2, _ = plot_segment(
        g2,
        sets_vertices=sets2 if sets2 else None,
        points_by_label=sols2,
        show_imputation=not bool(sets2),
        show_core=False,
    )
    fig2.savefig(out_dir / "viz_direct_2p.png", dpi=160, bbox_inches="tight")

    # --- n=3 (ternary) ---
    g3 = Game.from_coalitions(
        n_players=3,
        values={
            0: 0.0,
            1: 1.0,
            2: 1.2,
            4: 0.8,
            3: 2.8,
            5: 2.2,
            6: 1.9,
            7: 5.0,
        },
        player_labels=["P1", "P2", "P3"],
    )

    sols3 = {
        "shapley": shapley_value(g3),
        "normalized_banzhaf": normalized_banzhaf_value(g3),
        "tau": tau_value(g3),
    }
    clouds3 = {
        "weber": weber_marginal_vectors(g3),
    }

    fig3, _ = plot_ternary(
        g3,
        points_by_label=sols3,
        point_sets=clouds3,
        show_imputation=True,
        show_core=False,
    )
    fig3.savefig(out_dir / "viz_direct_3p.png", dpi=160, bbox_inches="tight")

    print("== Output ==")
    print("Wrote:")
    print("-", out_dir / "viz_direct_2p.png")
    print("-", out_dir / "viz_direct_3p.png")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        # Most likely: matplotlib not installed (tucoopy[viz]).
        raise SystemExit(str(e)) from e
