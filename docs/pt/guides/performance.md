# Performance, limites e custos computacionais

Esta pagina resume os principais **custos assintoticos** no `tucoop` e como escolher limites praticos para jogos maiores.

## Regra de ouro

- Quase tudo que "varre coalicoes" e pelo menos $O(2^n)$.
- Quase tudo que "varre permutacoes" e $O(n!)$.
- Quase tudo que "enumera vertices de politopo" explode com o numero de restricoes e dimensao.

Quando $n$ cresce, a estrategia recomendada e:

- preferir **amostragem** (pontos e/ou permutacoes),
- gerar `analysis.bundle` (tabelas e resumos) em vez de tentar "desenhar o simplex inteiro",
- limitar a geometria exata a $n$ pequeno.

## Tabela rapida (ordem de grandeza)

| Objeto / rotina | Custo tipico | Observacao |
|---|---:|---|
| Scans por coalicao (ex.: excessos) | $O(2^n)$ | depende de ter `v(S)` acessivel / cache |
| Shapley (exato, tabular) | $O(n 2^n)$ | via soma sobre subcoalicoes / DP |
| Banzhaf (exato, tabular) | $O(n 2^n)$ | similar ao Shapley em custo |
| Nucleolus / least-core / balancedness | varios LPs | cada LP pode ter muitas restricoes (coalicoes) |
| Weber set (exato) | $O(n!)$ | so viavel para $n$ pequeno |
| Poliedros (vertices) | exponencial | vertices nao escalam para $n$ grande |
| Hit-and-run (amostragem) | varios passos | requer set **limitado** e um ponto inicial (LP) |

## Recomendacoes praticas por familia

### Solucoes pontuais

- Para $n$ pequeno (ate ~10-12): `shapley_value`, `normalized_banzhaf_value` e afins podem ser usados exatos, desde que o jogo esteja "completo" (tabular).
- Para $n$ maior:
  - prefira aproximar Shapley por amostragem de permutacoes (quando disponivel),
  - evite rotinas com varios LPs (nucleolus/modiclus) sem limites claros.

### Geometria (sets / politopos)

- `PolyhedralSet.extreme_points(...)` e para visualizacao em dimensao baixa.
- Para projecoes quando $n$ cresce, prefira `project(..., approx_n_points=...)` (amostragem + projecao de pontos).

### Jogos simples / indices de poder

- Se voce consegue representar o jogo simples de forma compacta (ex.: weighted voting), indices como Banzhaf/SSI tendem a escalar melhor do que varrer todos os subconjuntos cegamente.
- Em jogos simples tabulares, ainda existe custo $O(2^n)$ para muita coisa.

### Weber set

O Weber set e o fecho convexo de vetores marginais. O gerador exato tem tamanho $n!$, entao:

- para $n$ pequeno: usar `WeberSet.points()` e `WeberSet.poly` (quando `n in {2,3}`),
- para $n$ maior: usar `WeberSet.sample_points(...)` e tratar o resultado como uma nuvem (nao como um politopo exato).

## Backends e dependencias

- Rotinas LP dependem de um backend (recomendado: SciPy/HiGHS). Veja `guides/lp_backends.md`.
- Algumas rotinas de performance usam NumPy quando disponivel (extra `tucoop[fast]`).

## Checklist de "o que fazer quando ficar lento"

1. Verifique se o jogo esta completo/tabular (quando a rotina assume isso).
2. Reduza `max_players` / `max_dim` / `max_points`.
3. Troque vertices por amostragem (`sample_points_*`) e projecao aproximada.
4. Se ha LP, confirme que SciPy esta instalado e sendo usado.
