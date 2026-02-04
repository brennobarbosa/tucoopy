# LP backends (SciPy vs PuLP)

Some `tucoopy` algorithms need to solve Linear Programming (LP) problems, for example:

- least-core / nucleolus / modiclus
- balancedness checks (Bondareva-Shapley)

These methods use a single adapter: `tucoopy.backends.lp.linprog_solve`.

## Options

### SciPy (recommended)

- Extra: `pip install "tucoopy[lp]"`
- Implementation: `scipy.optimize.linprog` (HiGHS)
- Best when you already have constraints in matrix form (`A_ub`, `A_eq`, ...), which is how `tucoopy` solvers are written.

### PuLP (alternative / fallback)

- Extra: `pip install "tucoopy[lp_alt]"`
- Implementation: a PuLP model solved by CBC (default)
- Useful if you cannot install SciPy, or if you later need MILP (integer/binary).

## How to choose

- Prefer SciPy for performance/robustness in continuous LP.
- Use PuLP when SciPy is not available or you need MILP (tucoopy does not use MILP today).

## Example

```py
from tucoopy.backends.lp import linprog_solve

res = linprog_solve(
    [1.0, 0.0],               # minimize x
    A_eq=[[1.0, 1.0]],         # x + y = 1
    b_eq=[1.0],
    bounds=[(None, None), (None, None)],
    backend="scipy",           # or "pulp"
)
print(res.x.tolist())
```

