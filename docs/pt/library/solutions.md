#
# Solucoes (point-valued)

Esta pagina e um mapa da pasta `tucoop.solutions`.

Objetivo: explicar **onde procurar** cada conceito, sem duplicar as docstrings
da API. Para detalhes completos (assinaturas, exemplos, parametros), use a
secao **Referencia (API)**.

See Also
--------
- Referencia (API) -> solutions: `../reference/index.md`
- `tucoop.solutions` (API): `../reference/solutions/shapley.md`

Conteudo
--------

## Valores axiomáticos / marginais

- Shapley: `../reference/solutions/shapley.md`
- Banzhaf (valor): `../reference/solutions/banzhaf.md`

## Solucoes baseadas em LP (deps opcionais)

> Warning
> Modulos como `nucleolus`/`modiclus` dependem de um backend de LP em runtime
> (recomendado: `pip install "tucoop[lp]"`).

- Least-core (ponto/solver): `../reference/solutions/least_core.md`
- Nucleolus: `../reference/solutions/nucleolus.md`
- Modiclus: `../reference/solutions/modiclus.md`

## Kernel / prekernel

- Kernel (solver): `../reference/solutions/kernel.md`

## Outros

- Tau value: `../reference/solutions/tau.md`
- Gately point: `../reference/solutions/gately.md`
- Dispatch (solve): `../reference/solutions/solve.md`
