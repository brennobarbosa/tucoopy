"""
# Parsing and conversion helpers for spec-like JSON objects.

This module provides:

- tolerant "field access" helpers for dict-like / attribute-like objects,
- JSON loading helpers for specs (string/bytes/path),
- converters between `tucoopy.base.game.Game` and a stable wire dict, and
- helpers to extract relevant pieces from an animation spec (game, sets, frames).

The wire formats are designed to remain simple, JSON-friendly, and compatible
with the schema files shipped in `tucoopy.io.schemas`.

Examples
--------
>>> from tucoopy.io.game_spec import spec_from_json
>>> spec = spec_from_json(
...     '{"n_players": 2, "characteristic_function": [{"coalition_mask": 0, "value": 0.0}] }'
... )
>>> isinstance(spec, dict)
True
"""
from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from ..base.game import Game
from ..base.types import TabularGameProtocol
from ..base.exceptions import InvalidParameterError, InvalidSpecError


# ----------------------------
# Generic field access
# ----------------------------

def get_field(obj: Any, key: str, default: Any = None) -> Any:
    """Get key from mapping-like or attribute-like objects."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ----------------------------
# JSON loading helpers
# ----------------------------

def spec_from_json(data: str | bytes | Path) -> dict[str, Any]:
    """
    Load a spec-like dict from JSON content.

    Accepts:
      - JSON string
      - JSON bytes
      - Path to a JSON file
      - str path to a .json file (if exists)
    """
    if isinstance(data, Path):
        text = data.read_text(encoding="utf-8")
        out = json.loads(text)
    elif isinstance(data, bytes):
        out = json.loads(data.decode("utf-8"))
    elif isinstance(data, str):
        p = Path(data)
        if "\n" not in data and p.suffix.lower() == ".json" and p.exists():
            out = json.loads(p.read_text(encoding="utf-8"))
        else:
            out = json.loads(data)
    else:
        raise InvalidParameterError("spec_from_json expects str | bytes | Path")

    if not isinstance(out, dict):
        raise InvalidSpecError("JSON root must be an object (dict)")
    return out


# ----------------------------
# Game <-> wire JSON (values)
# ----------------------------

def game_to_wire_dict(game: TabularGameProtocol) -> dict[str, Any]:
    """
    Serialize a `Game` into a JSONable dict (wire format).

    Stable format:
      {
        "n_players": int,
        "player_labels": [str] | null,
        "values": { "<mask>": float, ... }
      }
    """
    labels = getattr(game, "player_labels", None)
    player_labels = list(labels) if isinstance(labels, Sequence) else None
    return {
        "n_players": int(game.n_players),
        "player_labels": player_labels,
        "values": {str(int(k)): float(v) for k, v in game.v.items()},
    }


def game_from_wire_dict(data: Mapping[str, Any]) -> Game:
    """
    Deserialize a `Game` from wire format produced by `game_to_wire_dict`.
    """
    n = get_field(data, "n_players")
    if not isinstance(n, int) or n < 1:
        raise InvalidSpecError("wire game must have int n_players >= 1")

    labels = get_field(data, "player_labels", None)
    player_labels = list(labels) if isinstance(labels, Sequence) else None

    values_obj = get_field(data, "values", None)
    if not isinstance(values_obj, Mapping):
        raise InvalidSpecError("wire game must have 'values' as an object mapping mask->value")

    v: dict[int, float] = {}
    for k, val in values_obj.items():
        try:
            mask = int(k)
        except Exception as e:
            raise InvalidSpecError(f"wire values key must be int-like (got {k!r})") from e
        if not isinstance(val, (int, float)):
            raise InvalidSpecError(f"wire values[{k!r}] must be numeric")
        v[int(mask)] = float(val)

    if 0 not in v:
        v[0] = 0.0
    return Game(n_players=int(n), v=v, player_labels=player_labels)


# ----------------------------
# STRICT: AnimationSpec -> Game (CF)
# ----------------------------

def game_from_animation_spec(spec: Any) -> Game:
    """
    STRICT builder: construct a Game from an AnimationSpec-like object that uses
    the canonical CF shape:

      spec.game = {
        "n_players": int,
        "player_labels": [... optional ...],
        "characteristic_function": [{"coalition_mask": int, "value": number}, ...]
      }

    Inputs accepted:
      - dict-like spec
      - dataclass/attr-like spec
      - JSON string/bytes/Path that parses to a dict
    """
    if isinstance(spec, (str, bytes, Path)):
        spec = spec_from_json(spec)

    # Accept either top-level AnimationSpec dict/obj with .game,
    # or a direct game dict/obj (still CF format).
    root_game = spec
    g = get_field(spec, "game", None)
    if g is not None:
        root_game = g

    n = get_field(root_game, "n_players")
    if not isinstance(n, int) or n < 1:
        raise InvalidSpecError("AnimationSpec game must have int n_players >= 1")

    labels = get_field(root_game, "player_labels", None)
    player_labels = list(labels) if isinstance(labels, Sequence) else None

    cf = get_field(root_game, "characteristic_function", None)
    if not isinstance(cf, Sequence):
        raise InvalidSpecError("AnimationSpec game must have 'characteristic_function' as a list")

    values: dict[int, float] = {}
    for e in cf:
        mask = get_field(e, "coalition_mask")
        val = get_field(e, "value")
        if not isinstance(mask, int):
            raise InvalidSpecError("characteristic_function entries must have int coalition_mask")
        if not isinstance(val, (int, float)):
            raise InvalidSpecError("characteristic_function entries must have numeric value")
        values[int(mask)] = float(val)

    # Important: for consistency, ensure v(∅)=0 exists
    if 0 not in values:
        values[0] = 0.0

    return Game.from_coalitions(
        n_players=int(n),
        values=values,
        player_labels=player_labels,
    )


# ----------------------------
# Small helpers used by viz (public, not underscored)
# ----------------------------

def analysis_sets(spec: Any) -> Any:
    a = get_field(spec, "analysis", None)
    return None if a is None else get_field(a, "sets", None)


def analysis_solutions(spec: Any) -> Any:
    a = get_field(spec, "analysis", None)
    return None if a is None else get_field(a, "solutions", None)


def select_frame_allocation(
    spec: Any,
    *,
    series_id: str | None,
    frame_index: int | None,
    t: float | None,
) -> list[float] | None:
    series = get_field(spec, "series", None)
    if not isinstance(series, Sequence) or len(series) == 0:
        return None

    chosen: Any | None = None
    if series_id is None:
        chosen = series[0]
    else:
        for s in series:
            if get_field(s, "id", None) == series_id:
                chosen = s
                break
        if chosen is None:
            raise InvalidSpecError(f"series_id not found: {series_id!r}")

    frames = get_field(chosen, "frames", None)
    if not isinstance(frames, Sequence) or len(frames) == 0:
        return None

    if frame_index is not None:
        idx = max(0, min(int(frame_index), len(frames) - 1))
        f = frames[idx]
        alloc = get_field(f, "allocation", None)
        return [float(x) for x in alloc] if isinstance(alloc, Sequence) else None

    if t is not None:
        tt = float(t)
        best_alloc: Sequence[float] | None = None
        best_dt = float("inf")
        for f in frames:
            ft = get_field(f, "t", None)
            alloc = get_field(f, "allocation", None)
            if not isinstance(ft, (int, float)) or not isinstance(alloc, Sequence):
                continue
            dt = abs(float(ft) - tt)
            if dt < best_dt:
                best_alloc = alloc
                best_dt = dt
        return [float(x) for x in best_alloc] if best_alloc is not None else None

    alloc0 = get_field(frames[0], "allocation", None)
    return [float(x) for x in alloc0] if isinstance(alloc0, Sequence) else None


__all__ = [
    "get_field",
    "spec_from_json",
    "game_to_wire_dict",
    "game_from_wire_dict",
    "game_from_animation_spec",
    "analysis_sets",
    "analysis_solutions",
    "select_frame_allocation",
]

