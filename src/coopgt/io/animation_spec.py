"""
# JSON-ready animation spec dataclasses.

This module defines a small set of dataclasses that match the animation/spec
contract used by the JS demo renderer, plus helpers to build specs from:

- a `tucoop.base.game.Game` object, and
- a sequence of allocations (frames).

Notes
-----
`AnimationSpec.to_json` uses ``ensure_ascii=True`` to produce a strictly
ASCII JSON string (any non-ASCII content is escaped). This keeps the output
portable across environments while still being valid UTF-8 when written to disk.

Examples
--------
>>> from tucoop import Game
>>> from tucoop.io.animation_spec import build_animation_spec
>>> g = Game.from_coalitions(
...     n_players=2,
...     values={(): 0.0, (0,): 0.0, (1,): 0.0, (0, 1): 1.0},
... )
>>> spec = build_animation_spec(game=g, series_id="demo", allocations=[[0.0, 1.0]], dt=1/60)
>>> spec.schema_version
'0.1.0'
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from ..base.exceptions import InvalidParameterError
from ..base.types import GameProtocol, require_tabular_game


@dataclass(frozen=True)
class CharacteristicEntry:
    coalition_mask: int
    value: float


@dataclass(frozen=True)
class GameSpec:
    n_players: int
    characteristic_function: list[CharacteristicEntry]
    player_labels: list[str] | None = None


@dataclass(frozen=True)
class FrameSpec:
    t: float
    allocation: list[float]
    highlights: dict[str, Any] | None = None


@dataclass(frozen=True)
class SeriesSpec:
    id: str
    frames: list[FrameSpec]
    description: str | None = None


@dataclass(frozen=True)
class AnimationSpec:
    schema_version: str
    game: GameSpec
    series: list[SeriesSpec]
    analysis: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None
    visualization_hints: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        # asdict() is enough here; keep schema simple and explicit.
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=True)

    def write_json(self, path: str | Path, *, indent: int = 2) -> None:
        Path(path).write_text(self.to_json(indent=indent), encoding="utf-8")


def game_to_spec(game: GameProtocol) -> GameSpec:
    if game.n_players < 1:
        raise InvalidParameterError("n_players must be >= 1")
    tabular = require_tabular_game(game, context="game_to_spec")
    entries = [
        CharacteristicEntry(coalition_mask=int(mask), value=float(val))
        for mask, val in sorted(tabular.v.items(), key=lambda kv: kv[0])
    ]
    return GameSpec(
        n_players=game.n_players,
        player_labels=getattr(game, "player_labels", None),
        characteristic_function=entries,
    )


def series_from_allocations(
    *,
    series_id: str,
    allocations: Sequence[Sequence[float]],
    dt: float,
    highlights: Sequence[dict[str, Any] | None] | None = None,
    highlight_fn: Callable[[int, Sequence[float]], dict[str, Any] | None] | None = None,
    description: str | None = None,
) -> SeriesSpec:
    if dt <= 0:
        raise InvalidParameterError("dt must be > 0")
    if highlights is not None and highlight_fn is not None:
        raise InvalidParameterError("provide at most one of highlights or highlight_fn")
    if highlights is not None and len(highlights) != len(allocations):
        raise InvalidParameterError("highlights length must match allocations length")
    frames: list[FrameSpec] = []
    for k, alloc in enumerate(allocations):
        h = highlights[k] if highlights is not None else (highlight_fn(k, alloc) if highlight_fn else None)
        frames.append(FrameSpec(t=float(k) * float(dt), allocation=[float(x) for x in alloc], highlights=h))
    return SeriesSpec(id=str(series_id), frames=frames, description=description)


def build_animation_spec(
    game: GameProtocol,
    *,
    series_id: str,
    allocations: Sequence[Sequence[float]],
    dt: float,
    series_description: str | None = None,
    include_analysis: bool = False,
    analysis_kwargs: dict[str, Any] | None = None,
    include_frame_diagnostics: bool = False,
    frame_diagnostics_max_players: int = 4,
    analysis: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    visualization_hints: dict[str, Any] | None = None,
    schema_version: str = "0.1.0",
) -> AnimationSpec:
    s = series_from_allocations(series_id=series_id, allocations=allocations, dt=dt, description=series_description)

    if include_frame_diagnostics and game.n_players <= int(frame_diagnostics_max_players):
        from ..diagnostics.core_diagnostics import is_in_core, max_excess

        new_frames: list[FrameSpec] = []
        for fr in s.frames:
            diag = {"core": {"in_core": bool(is_in_core(game, fr.allocation)), "max_excess": float(max_excess(game, fr.allocation))}}
            h = dict(fr.highlights) if isinstance(fr.highlights, dict) else {}
            h["diagnostics"] = diag
            new_frames.append(replace(fr, highlights=h))
        s = replace(s, frames=new_frames)

    if include_analysis:
        from ..io.analysis import build_analysis as _build_analysis

        analysis = _build_analysis(game, **(analysis_kwargs or {}))

    return AnimationSpec(
        schema_version=str(schema_version),
        game=game_to_spec(game),
        series=[s],
        analysis=analysis,
        meta=meta,
        visualization_hints=visualization_hints,
    )


__all__ = [
    "CharacteristicEntry",
    "GameSpec",
    "FrameSpec",
    "SeriesSpec",
    "AnimationSpec",
    "game_to_spec",
    "series_from_allocations",
    "build_animation_spec",
]
