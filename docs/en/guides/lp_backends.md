# Backends de LP (SciPy vs PuLP)

Alguns algoritmos do `tucoopy` precisam resolver problemas de Programação Linear (LP), por exemplo:

- least-core / nucleolus / modiclus
- checagem de balancedness (Bondareva–Shapley)

Esses métodos usam um único adapter: `tucoopy.backends.lp.linprog_solve`.

## Opções

### SciPy (recomendado)

- Extra: `pip install "tucoopy[lp]"`
- Implementação: `scipy.optimize.linprog` (HiGHS)
- Melhor quando você já tem as restrições em forma matricial (`A_ub`, `A_eq`, ...), que é como os solvers do `tucoopy` são escritos.

### PuLP (alternativa / fallback)

- Extra: `pip install "tucoopy[lp_alt]"`
- Implementação: modelo PuLP resolvido pelo CBC (padrão)
- Útil se você não consegue instalar SciPy, ou se futuramente precisar de MILP (inteiro/binário).

## Como escolher

- Prefira SciPy por performance/robustez em LP contínuo.
- Use PuLP quando SciPy não estiver disponível ou você precisar de MILP (o `tucoopy` não usa MILP hoje).

## Exemplo

```py
from tucoopy.backends.lp import linprog_solve

res = linprog_solve(
    [1.0, 0.0],              # minimizar x
    A_eq=[[1.0, 1.0]],        # x + y = 1
    b_eq=[1.0],
    bounds=[(None, None), (None, None)],
    backend="scipy",          # ou "pulp"
)
print(res.x.tolist())
```
