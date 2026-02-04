# JS vs Python (projetos paralelos)

Este repo tem dois projetos que se complementam:

- **`tucoop-py` (Python)**: biblioteca de teoria dos jogos cooperativos (TU), com metodos pesados (LP, varreduras maiores, diagnosticos completos) e geracao de JSON.
- **`tucoop-js` / `@tucoop/core` (TypeScript/JS)**: runtime leve para **validar, completar analises baratas e renderizar** specs no browser (Canvas).

## Contrato de compatibilidade

A compatibilidade entre os dois projetos e pelo **contrato JSON** (schemas), nao pela arquitetura interna:

- `schema/tucoop-animation.schema.json` (canonico no repo)
- schema empacotado no Python: `tucoop.io.schemas/tucoop-animation.schema.json`

Recomendacao: trate o schema como a "interface" entre os projetos.

## O que o JS consegue computar sozinho (client-side)

O `@tucoop/core` tem um modo de "completar" o spec quando ele chega "cru" (apenas `game` + `series`):

- `analysis.solutions` (quando viavel para `n` pequeno):
  - Shapley (enumeracao direta, custo $O(n 2^n)$)
  - Banzhaf e Banzhaf normalizado (enumeracao direta)
- `analysis.sets` (quando viavel para `n` pequeno):
  - vertices do **imputation set** (simplex deslocado; barato)
  - vertices do **core** (enumeracao por ativacao de restricoes + solve linear; pensado para `n` pequeno)
  - pontos geradores do **Weber set** (vetores marginais por permutacoes; custo $n!$)

Essas computacoes estao concentradas em:

- `packages/tucoop-js/src/compute.ts` (`deriveAnalysis(spec, opts)`)

Limites:

- Por padrao, o JS so tenta completar `analysis` se `n_players <= 4` (`maxPlayers`).
- Nao ha backend de LP no browser: o JS evita algoritmos que exigem solver.

## O que deve ficar no Python (offline ou backend)

Regra pratica: tudo que depende de **LP** ou algoritmos mais instaveis/numericamente delicados deve ser computado no Python.

Exemplos tipicos:

- least-core, nucleolus, modiclus (LP)
- balancedness (Bondareva-Shapley via LP)
- bargaining set (caro; sampling/LP)
- kernel/pre-kernel (iterativo; pode exigir rotinas numericas e cuidados com degenerescencia)
- enumeracao de vertices/projecoes para dimensoes maiores

O pipeline recomendado e:

1. Rodar o Python para computar `analysis` (solucoes, sets, diagnosticos).
2. Exportar o spec JSON.
3. O JS so **renderiza** e, opcionalmente, completa o que estiver faltando e for barato.

## Por que manter separado?

- JS: foco em **experiencia de visualizacao** e runtime leve (zero dependencias pesadas).
- Python: foco em **correcao**, **diagnosticos** e algoritmos classicos (com dependencias opcionais como `scipy/numpy`).

Isso evita que:

- o JS vire uma biblioteca numerica pesada, e
- o Python fique "preso" ao design do renderer.

## See Also

- `packages/tucoop-js/README.md` (como usar o renderer e `deriveAnalysis`)
- `guides/animation_spec.md` (como gerar specs no Python)
- `guides/analysis_contract.md` (o que entra em `analysis`)

