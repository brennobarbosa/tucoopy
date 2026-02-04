from __future__ import annotations

"""
Generate JSON specs for the JS demo (2p/3p/4p).

This script writes a few small AnimationSpec JSON files into `examples/out/`.
These can be used as input to both:

- the JS renderer demo, and
- the Python static Matplotlib helpers (for n=2 and n=3).
"""

from pathlib import Path
import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from tucoopy.base.game import Game
from tucoopy.io.animation_spec import AnimationSpec, build_animation_spec, game_to_spec
from tucoopy.solutions.shapley import shapley_value


def _resolve_out_dir(out: str) -> Path:
    p = Path(out)
    if p.is_absolute():
        return p
    return Path(__file__).resolve().parent / p


def write(spec: AnimationSpec, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / name
    spec.write_json(path)
    print("Wrote", path)


def make_2p() -> AnimationSpec:
    # Trivial TU game: v(S)=0 for proper S, v(N)=1. Core = imputation segment.
    g = Game.from_coalitions(
        n_players=2,
        values={(): 0.0, (0, 1): 1.0},
        player_labels=["P1", "P2"],
    )

    # Animate along the efficient segment.
    n_frames = 180
    allocs = []
    for k in range(n_frames):
        t = k / (n_frames - 1)
        # back-and-forth
        u = 2 * t if t <= 0.5 else 2 * (1 - t)
        allocs.append([u, 1.0 - u])

    spec = build_animation_spec(
        g,
        schema_version="0.1.0",
        series_id="segment_walk",
        allocations=allocs,
        dt=1 / 60,
        series_description="Back-and-forth along the efficient segment (core = imputation).",
        include_analysis=True,
        analysis_kwargs={"include_blocking_regions": False, "include_weber": False, "max_players": 4},
        include_frame_diagnostics=True,
        frame_diagnostics_max_players=4,
        meta={"generator": "examples/06_generate_specs_for_js_demo.py", "demo": "2p"},
    )
    # Keep this generator explicit about the embedded game spec (for simple downstream readers).
    return AnimationSpec(
        schema_version=spec.schema_version,
        meta=spec.meta,
        game=game_to_spec(g),
        analysis=spec.analysis,
        series=spec.series,
        visualization_hints=spec.visualization_hints,
    )


def make_3p() -> AnimationSpec:
    # Classic 3p TU example (non-trivial core polygon).
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
    V = g.value(g.grand_coalition)
    equal = [V / 3.0] * 3

    n_frames = 240
    allocs = []
    for k in range(n_frames):
        t = k / (n_frames - 1)
        allocs.append([(1 - t) * equal[i] + t * phi[i] for i in range(3)])

    spec = build_animation_spec(
        g,
        schema_version="0.1.0",
        series_id="equal_to_shapley",
        allocations=allocs,
        dt=1 / 60,
        series_description="Interpolation: equal split -> Shapley.",
        include_analysis=True,
        analysis_kwargs={"include_blocking_regions": True, "include_weber": False, "max_players": 4},
        include_frame_diagnostics=True,
        frame_diagnostics_max_players=4,
        meta={"generator": "examples/06_generate_specs_for_js_demo.py", "demo": "3p"},
    )
    return AnimationSpec(
        schema_version=spec.schema_version,
        meta=spec.meta,
        game=game_to_spec(g),
        analysis=spec.analysis,
        series=spec.series,
        visualization_hints=spec.visualization_hints,
    )


def make_4p() -> AnimationSpec:
    # Trivial 4p TU example: core = imputation tetra (good for 3D projection demo).
    g = Game.from_coalitions(
        n_players=4,
        values={(): 0.0, (0, 1, 2, 3): 1.0},
        player_labels=["P1", "P2", "P3", "P4"],
    )

    # Walk around tetra vertices in a loop.
    verts = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    n_frames = 240
    allocs = []
    for k in range(n_frames):
        t = k / (n_frames - 1)
        # segment index
        seg = int(t * 4) % 4
        u = (t * 4) - seg
        a = verts[seg]
        b = verts[(seg + 1) % 4]
        allocs.append([(1 - u) * a[i] + u * b[i] for i in range(4)])

    spec = build_animation_spec(
        g,
        schema_version="0.1.0",
        series_id="tetra_walk",
        allocations=allocs,
        dt=1 / 60,
        series_description="Walk around tetra edges (core = imputation). Drag to rotate.",
        include_analysis=True,
        analysis_kwargs={"include_blocking_regions": False, "include_weber": False, "max_players": 4},
        include_frame_diagnostics=True,
        frame_diagnostics_max_players=4,
        meta={"generator": "examples/06_generate_specs_for_js_demo.py", "demo": "4p"},
    )
    return AnimationSpec(
        schema_version=spec.schema_version,
        meta=spec.meta,
        game=game_to_spec(g),
        analysis=spec.analysis,
        series=spec.series,
        visualization_hints=spec.visualization_hints,
    )


def main() -> None:
    try:
        import scipy  # noqa: F401
    except Exception:
        print('This example requires SciPy (analysis/sets). Install with: pip install "tucoopy[lp]"')
        return

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="out",
        help="Output directory (absolute, or relative to this file). Default: out",
    )
    args = parser.parse_args()

    out_dir = _resolve_out_dir(str(args.out))

    print("== Output ==")
    write(make_2p(), out_dir, "demo_2p.json")
    write(make_3p(), out_dir, "demo_3p.json")
    write(make_4p(), out_dir, "demo_4p.json")


if __name__ == "__main__":
    main()
