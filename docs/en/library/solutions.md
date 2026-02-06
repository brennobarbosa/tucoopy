#
# Solutions (point-valued)

This page is a map of the `tucoopy.solutions` package.

Goal: explain **where to look** for each concept, without duplicating the API docstrings.
For full details (signatures, examples, parameters), use the **API Reference** section.

## See also

- API Reference -> solutions: `../reference/index.md`
- `tucoopy.solutions` (API): `../reference/solutions/shapley.md`

## Contents

### Axiomatic / marginal values

- Shapley: `../reference/solutions/shapley.md`
- Banzhaf (value): `../reference/solutions/banzhaf.md`

### LP-based solutions (optional deps)

> Warning
> Modules like `nucleolus` / `modiclus` require an LP backend at runtime
> (recommended: `pip install "tucoopy[lp]"`).

- Least-core (set): `../reference/geometry/least_core_set.md`
- Nucleolus: `../reference/solutions/nucleolus.md`
- Modiclus: `../reference/solutions/modiclus.md`

### Other

- Tau value: `../reference/solutions/tau.md`
- Gately point: `../reference/solutions/gately.md`
- Dispatch (solve): `../reference/solutions/solve.md`
