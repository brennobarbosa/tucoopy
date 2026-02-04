"""
IO: round-trip an AnimationSpec as JSON.

This example demonstrates a "spec" workflow:

- build an `AnimationSpec` from a `Game` + allocations,
- serialize it to JSON (ASCII-safe string),
- parse it back as a Python dict,
- rebuild the `Game` from `spec.game`.
"""

from __future__ import annotations

import json

from _bootstrap import add_src_to_path

add_src_to_path()

from tucoopy import Game  # noqa: E402
from tucoopy.io.animation_spec import build_animation_spec  # noqa: E402
from tucoopy.io.game_spec import game_from_animation_spec  # noqa: E402


def main() -> None:
    g = Game.from_coalitions(
        n_players=2,
        values={(): 0.0, (0,): 0.0, (1,): 0.0, (0, 1): 1.0},
        player_labels=["P1", "P2"],
    )

    allocations = [[0.0, 1.0], [0.25, 0.75], [0.5, 0.5]]
    spec = build_animation_spec(game=g, series_id="demo", allocations=allocations, dt=1 / 30)

    s = spec.to_json(indent=2)
    data = json.loads(s)

    g2 = game_from_animation_spec(data)

    print("schema_version:", spec.schema_version)
    print("n_players:", g2.n_players)
    print("v(N):", g2.value(g2.grand_coalition))
    print("player_labels:", getattr(g2, "player_labels", None))
    print("json length:", len(s))


if __name__ == "__main__":
    main()

