"""
# Bundled JSON Schema helpers.

This module loads JSON Schemas bundled with the package under
``tucoopy.io.schemas`` and exposes small helpers to:

- return schema dicts (for validation tooling), and
- write schema files to disk (for editors/CI).

Examples
--------
>>> from tucoopy.io.schema import animation_spec_schema
>>> schema = animation_spec_schema()
>>> isinstance(schema, dict)
True
"""
from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from ..base.exceptions import InvalidParameterError, InvalidSpecError


def _load_schema(filename: str) -> dict[str, Any]:
    root = resources.files("tucoopy.io").joinpath("schemas")
    with root.joinpath(filename).open("rb") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise InvalidSpecError(f"Schema file must contain a JSON object: {filename}")
    return data


def animation_spec_schema() -> dict[str, Any]:
    """
    Return the JSON Schema for the tucoopy animation spec.
    """
    return _load_schema("tucoop-animation.schema.json")


def game_schema() -> dict[str, Any]:
    """
    Return the JSON Schema for `tucoop.io.json.game_to_dict` output.
    """
    return _load_schema("tucoop-game.schema.json")


def write_schema(
    path: str | Path,
    *,
    which: str = "animation_spec",
    indent: int = 2,
) -> None:
    """
    Write a bundled schema JSON file to disk.

    Parameters
    ----------
    path
        Output path for the schema JSON.
    which
        "animation_spec" or "game".
    """
    which = str(which).strip().lower()
    if which == "animation_spec":
        data = animation_spec_schema()
    elif which == "game":
        data = game_schema()
    else:
        raise InvalidParameterError("which must be 'animation_spec' or 'game'")

    Path(path).write_text(json.dumps(data, indent=indent), encoding="utf-8")


__all__ = ["animation_spec_schema", "game_schema", "write_schema"]
