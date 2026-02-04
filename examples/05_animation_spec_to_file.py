from __future__ import annotations

"""
Build an AnimationSpec JSON file (for the JS demo contract).

This example computes the Shapley value of a small game and writes an
AnimationSpec-like JSON with analysis and frame diagnostics included.
"""

from pathlib import Path
import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from tucoop.base.game import Game
from tucoop.io.animation_spec import build_animation_spec
from tucoop.solutions.shapley import shapley_value


def _resolve_out_dir(out: str) -> Path:
    p = Path(out)
    if p.is_absolute():
        return p
    return Path(__file__).resolve().parent / p


def main() -> None:
    try:
        import scipy  # noqa: F401
    except Exception:
        print('This example requires SciPy (analysis + core diagnostics). Install with: pip install "tucoop[lp]"')
        return

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="out",
        help="Output directory (absolute, or relative to this file). Default: out",
    )
    args = parser.parse_args()

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
    spec = build_animation_spec(
        g,
        schema_version="0.1.0",
        series_id="shapley",
        allocations=[phi] * 60,
        dt=1 / 30,
        include_analysis=True,
        analysis_kwargs={"include_blocking_regions": True, "include_weber": False, "max_players": 4},
        include_frame_diagnostics=True,
        frame_diagnostics_max_players=4,
        meta={"generator": "tucoop-py/examples/05_animation_spec_to_file.py"},
    )

    out_dir = _resolve_out_dir(str(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "shapley_core_demo.json"
    spec.write_json(out_path)
    print("== Output ==")
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
