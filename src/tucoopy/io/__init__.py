"""
# I/O utilities for cooperative game specifications, analysis, and serialization.

This submodule exposes helpers for:

- animation/game specs (contract objects and builders),
- schema loaders (bundled JSON Schema),
- analysis builders (compute solutions/sets/diagnostics into JSON-friendly dicts),
- and parsing/conversion utilities for wire JSON formats.

Main imports:
    - AnimationSpec, GameSpec, FrameSpec, SeriesSpec
    - build_analysis, build_animation_spec, game_to_spec, series_from_allocations
    - game_to_wire_dict, game_from_wire_dict, spec_from_json, get_field
    - animation_spec_schema, game_schema, write_schema

Examples
--------
>>> from tucoopy import Game
>>> from tucoopy.io import build_analysis
>>> g = Game.from_coalitions(n_players=2, values={(): 0.0, (0,): 0.0, (1,): 0.0, (0, 1): 1.0})
>>> analysis = build_analysis(g, include_blocking_regions=False, include_bundle=False)
>>> "solutions" in analysis
True
"""
from .animation_spec import (
    AnimationSpec,
    CharacteristicEntry,
    FrameSpec,
    GameSpec,
    SeriesSpec,
    build_animation_spec,
    game_to_spec,
    series_from_allocations,
)
from .game_spec import (
    get_field,
    spec_from_json,
    game_to_wire_dict,
    game_from_wire_dict,
    game_from_animation_spec,
    analysis_sets,
    analysis_solutions,
    select_frame_allocation,
)
from .analysis import build_analysis
from .schema import animation_spec_schema, game_schema, write_schema

__all__ = [
    "AnimationSpec",
    "CharacteristicEntry",
    "FrameSpec",
    "GameSpec",
    "SeriesSpec",
    "build_animation_spec",
    "game_to_spec",
    "series_from_allocations",
    "analysis_sets",
    "analysis_solutions",
    "game_to_wire_dict",
    "game_from_wire_dict",
    "game_from_animation_spec",
    "get_field",
    "select_frame_allocation",
    "spec_from_json",
    "build_analysis",
    "animation_spec_schema",
    "game_schema",
    "write_schema",

]
