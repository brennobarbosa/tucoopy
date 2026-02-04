# Public API (minimal contract)

This document defines the **minimal public contract** of `tucoopy` (Python).
It exists to reduce churn: the imports listed here should remain stable.

## General rule

- The top-level `tucoopy` is **small** and serves as an entry point.
- The full surface lives in subpackages (`tucoopy.geometry`, `tucoopy.solutions`, etc.).
- If something is not documented here, it may change more freely (especially in `0.x`).

## Canonical imports (stable)

Recommended for users:

```py
from tucoopy import Game
from tucoopy.games import weighted_voting_game
from tucoopy.solutions import shapley_value, nucleolus
from tucoopy.geometry import Core, EpsilonCore, LeastCore
from tucoopy.power import banzhaf_index, shapley_shubik_index
```

## Top-level (`tucoopy`)

The top-level should expose only a few high-level items (convenience).
Everything else should be imported from subpackages.

Currently exposed:

- `Game`
- `mask_from_players`
- `glove_game`, `weighted_voting_game`
- `Core`
- `shapley_value`
- `nucleolus` (requires an LP backend at runtime when called)

## Subpackages (source of truth)

- `tucoopy.base`: primitives (coalitions, games, config, types/exceptions)
- `tucoopy.games`: classic game generators
- `tucoopy.solutions`: point solutions (payoff vectors)
- `tucoopy.geometry`: sets/polyhedra (core, least-core, etc.)
- `tucoopy.diagnostics`: checks and explanations (per set / per allocation)
- `tucoopy.power`: power indices for simple games/voting
- `tucoopy.transforms`: transforms and representations
- `tucoopy.io`: JSON specs and schemas
- `tucoopy.backends`: adapters for optional dependencies

## What is experimental?

While the project is in `0.x`, we consider the following more subject to change:

- diagnostic details (additional fields, internal structures);
- LP implementations and detailed solver explanations;
- some visualization and sampling utilities.

When a feature moves from "experimental" to "stable", it should:

- have a complete docstring (NumPy style),
- have unit tests,
- be added to this document.

