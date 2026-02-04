# Roadmap (reset)

This roadmap was reset to start a new phase of the project (product release).
Use the checkboxes to track progress.

## Legend

- `[x]` done
- `[ ]` pending
- `(*)` optional / later

---

## 1) MVP (minimal theory + stability)

- [ ] Define MVP scope (what's in / what's out)
- [ ] Lock down a "minimal public API" and canonical imports
- [ ] Review numerics / tolerances (document `tol` and limits)
- [ ] Review error messages (baseline + optional extras)
- [ ] Review the JSON contract (schemas + examples)

## 2) Docs (release)

- [ ] Review theory pages (minimal theory)
- [ ] Review guides (quickstart, diagnostics, large games, performance)
- [ ] Review the examples page (more short, canonical examples)
- [ ] Review API reference (mkdocstrings): docstrings with examples where it makes sense
- [ ] Check MkDocs navigation (complete nav, no orphan pages)

## 3) Quality (CI + tests)

- [ ] Run `pytest` on Windows and Linux (CI)
- [ ] Run `mypy` on Windows and Linux (CI)
- [ ] Run `mkdocs build` (EN + PT) as part of CI
- [ ] Increase coverage in higher-risk areas (LP/geometry/diagnostics)
- [ ] Evaluate parallelization (hotspots in coalition loops, sampling, excess scans)

## 4) Product (packaging)

- [ ] Review `pyproject.toml` (extras, classifiers, python_requires)
- [ ] Review Python package README (short, with a "feature -> extra" table)
- [ ] Review `CHANGELOG.md` (minimal for release)
- [ ] Prepare release checklist (tag, build, publish)

## 5) Later (do not block the release)

- [x] (*) Translate docs to EN
- [ ] (*) Optional integration with a polyhedron backend (cddlib/polymake) for `n>3`
- [ ] (*) More solutions/sets (as needed)
- [ ] (*) Parallelization: implement via `concurrent.futures` (process) for naturally independent tasks (permutations, coalitions, sampling), controlled by `max_workers`

