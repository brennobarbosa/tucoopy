<!--
This guide is short by design. Theory/API docs live on other pages.
-->

# Contributing

This document describes a simple workflow for contributing to `tucoopy`.

## Setup (dev)

Recommended: create a virtual environment at the repo root and install the package in editable mode.

```bash
python -m venv .venv
```

Activate the environment and install with dev/docs extras:

```bash
pip install -e ".[dev,docs]"
```

Optional extras:

- `lp`: `pip install -e ".[lp]"` (SciPy; enables LP-based methods)
- `lp_alt`: `pip install -e ".[lp_alt]"` (PuLP; fallback)
- `fast`: `pip install -e ".[fast]"` (NumPy; speedups)
- `viz`: `pip install -e ".[viz]"` (Matplotlib; static visualization)

## Running local checks

### Tests

```bash
pytest -q
```

### Type checking

```bash
mypy src/tucoopy
```

### Docs (MkDocs)

Build the Portuguese docs:

```bash
mkdocs build -f mkdocs.pt.yml --dirty
```

Build the English docs:

```bash
mkdocs build -f mkdocs.yml --dirty
```

## Quality rules (project)

- Avoid duplication: keep one source of truth per concept/module.
- Avoid shims/aliases for "compatibility": prefer canonical imports.
- Prefer docstrings in **NumPy style** (as configured in mkdocstrings).

## Windows/OneDrive

If you are developing inside OneDrive, see:

- `../guides/dev_windows_onedrive.md`

## Pull requests

Before opening a PR:

- run `pytest` and `mypy`;
- validate `mkdocs build` (EN/PT) if you touched docs/docstrings;
- describe API impact and changes to optional extras/dependencies when applicable.

