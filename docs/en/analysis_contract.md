# Contrato de `analysis` (Python -> JSON)

Esta página documenta o significado dos campos em `AnimationSpec.analysis` e os limites/garantias de cada um.

## Princípios

- `analysis` é **opcional** e pensado para cache/explicabilidade no front.
- Para **n<=4**, o Python pode exportar geometrias (vértices) para desenhar.
- Para **n>4**, preferimos exportar **tabelas/listas** (bundle) em vez de objetos geométricos.

## `analysis.meta`

Proveniência e parâmetros usados para gerar o `analysis`:

- `analysis.meta.computed_by`
- `analysis.meta.build_analysis` (flags + `max_players`, `tol`, `diagnostics_top_k`)
- `analysis.meta.computed` (quais seções foram incluídas)
- `analysis.meta.skipped` (razões para pular seções)
- `analysis.meta.limits` (limites/truncamentos aplicados, ex.: `diagnostics_max_list`)
- `analysis.meta.contract_version` (versão do contrato do `analysis`)

## `analysis.solutions`

Pontos (alocações) nomeados:

- Ex.: `shapley`, `normalized_banzhaf`
- Cada entrada tem `{ "allocation": [x1, ..., xn] }`

Limites:
- Pode ser **exponencial** em `n`. Por padrão, `build_analysis` só calcula para `n<=max_players`.
- `analysis.solutions.*.meta` registra `computed_by` e o `method`.

## `analysis.sets`

Representações para visualização (n pequeno):

- `imputation.vertices`: vértices do conjunto de imputações
- `core.vertices`: vértices do núcleo
- `reasonable.vertices`: vértices do conjunto razoável (imputation + limites superiores M)
- `core_cover.vertices`: vértices do core cover (m <= x <= M)
- `weber.points` (opcional)

Limites:
- Produzido apenas para `n<=max_players`.
- Cada entrada pode incluir `meta` com proveniência.
- Listas grandes (`vertices`/`points`) podem ser truncadas por `max_points` e sinalizadas em `meta.truncated`.

## `analysis.blocking_regions`

Cache de regiões de bloqueio (atualmente para `n=3`):

- `coordinate_system`
- `regions[]`: polígonos com `coalition_mask` associado

## `analysis.diagnostics`

Diagnósticos para UI (tooltips/tabelas):

- `analysis.diagnostics.input`: checks do jogo (ex.: `vN`, `sum_singletons`, `essential`)
- `analysis.diagnostics.solutions.<id>.core`: por que um ponto está/não está no núcleo

Truncamentos:
- Listas potencialmente grandes em `analysis.diagnostics.input` podem ser truncadas (ex.: `missing_coalition_masks`).
  O campo `missing_coalition_masks_truncated` indica se houve truncamento e `missing_coalition_mask_count` informa o total.

Consistência (quando aplicável, n pequeno):
- `simple_game`: se todos os valores estão em `{0,1}`
- `monotone_simple_game`: se o jogo simples é monotone (v(S) <= v(S∪{i}))
- `monotone_counterexample`: um contraexemplo quando `monotone_simple_game=false`

## Explicações baseadas em LP (opcional)

Quando `include_lp_explanations=true` e `n<=lp_explanations_max_players`, `build_analysis` pode incluir:

- `analysis.diagnostics.lp.balancedness_check` (Bondareva–Shapley: certificado de core vazio)
- `analysis.diagnostics.lp.least_core` (ε do least-core, coalizões tight e diagnósticos do solver)

## `analysis.bundle` (n>4)

Para `n>max_players`, `build_analysis` inclui um “bundle” leve com resumos:

- `analysis.bundle.game_summary` (ex.: `n_players`, `vN`, `essential`, `provided_coalitions`)
- `analysis.bundle.notes` (como interpretar limites)
- `analysis.bundle.meta` registra proveniência.

Quando `n<=bundle_max_players` e o jogo é **completo** (tem todos os `v(S)`), o bundle também pode incluir tabelas:

- `analysis.bundle.tables.players` (scalars por jogador)
- `analysis.bundle.tables.power_indices` (somente para jogos simples completos)
- `analysis.bundle.tables.tau_vectors` (utopia payoff e minimal rights; requer jogo completo)
- `analysis.bundle.tables.approx_solutions.shapley` (Shapley aproximado via amostragem; inclui `stderr`)
