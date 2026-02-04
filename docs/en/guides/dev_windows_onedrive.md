# Development on Windows/OneDrive

This guide focuses on common issues when developing `tucoopy` on Windows while the repo lives inside OneDrive.
The goal is to avoid wasting time with slow IO, locked files, and inconsistent caches.

## Common symptoms

- `pytest` is much slower than expected.
- `.pyc` files and caches appear/disappear and create weird diffs.
- OneDrive "holds" files in use (locks) and some operations fail intermittently.
- `mkdocs build` may get heavy when reprocessing many files.

## Practical recommendations

### 1) Prefer working outside OneDrive (recommended)

If possible, move the repo to a path that is not being synced (e.g. `C:\\dev\\tucoopy`).
This usually resolves 80% of performance/locking issues.

### 2) If you must stay in OneDrive, reduce bytecode writes

To avoid `.pyc` spam and lock/cache issues:

- Set `PYTHONPYCACHEPREFIX` to a directory outside OneDrive.
- Or, for short sessions, use `PYTHONDONTWRITEBYTECODE=1`.

Example (PowerShell, current session):

```powershell
$env:PYTHONPYCACHEPREFIX = "C:\\temp\\pycache"
# or
$env:PYTHONDONTWRITEBYTECODE = "1"
```

### 3) Pytest: keep caches under control

In slow environments, it helps to:

- Use `-q` to reduce output.
- Run narrower tests first (a file / a single test).
- If cache instability shows up, clear the pytest cache when needed:

```powershell
Remove-Item -Recurse -Force .pytest_cache -ErrorAction SilentlyContinue
```

### 4) MkDocs: incremental builds

When writing docs, prefer incremental builds (`--dirty`) when possible:

```powershell
python -m mkdocs build -f mkdocs.pt.yml --dirty
```

## Note on reproducibility

When reporting performance/IO bugs:

- mention whether the repo is in OneDrive,
- report your Python version,
- and report whether you're using `scipy` and/or `numpy` (optional backends).

