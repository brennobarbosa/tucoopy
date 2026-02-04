# Deprecation policy

This document defines how `tucoopy` (Python) handles API changes.

## Goal

- Avoid drift and repeated refactors.
- Provide predictability for users and for the documentation.

## Versions `0.x`

The project is in an alpha phase (`0.x`):

- Breaking API changes may still happen.
- Still, we try to **deprecate before removing** when it does not add too much cost.

## Recommended process (when possible)

1. **Mark as deprecated**
   - Document in the changelog.
   - Update docs (reference page, examples).
2. **Emit a warning**
   - Use `DeprecationWarning` (or a specific exception) only when it makes sense.
3. **Remove**
   - Prefer removing in the next "minor" (e.g. `0.2.0`), or at most in `+2` releases.

## "Ghost files" and duplicates

Empty files (or files with `# delete`) should not remain:

- Prefer **actually removing** them.
- If removal is not possible (environment constraints), keep a module that:
  - fails on import with an explicit error, and
  - explains the replacement (new path / new source-of-truth layer).

## JSON contract compatibility

Compatibility should be maintained at the **JSON contract** level (`tucoopy.io.schema`), not via module shims/aliases.

