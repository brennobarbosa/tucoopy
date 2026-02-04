# Depurar core/ε-core via `analysis.diagnostics`

O objetivo de `analysis.diagnostics` é permitir que o front explique “por que” um ponto falha (ou passa) sem chamadas a backend.

## Por que um ponto não está no core?

Para um ponto $x$, o diagnóstico principal é:

- $\text{maxexcess} = \max_S (v(S) - x(S))$
- Se $\text{maxexcess} > 0$, existe uma coalizão que bloqueia.

No JSON, isso aparece como:

- `analysis.diagnostics.solutions.<id>.core.max_excess`
- `analysis.diagnostics.solutions.<id>.core.tight_coalitions`
- `analysis.diagnostics.solutions.<id>.core.violations` (top-k com `vS`, `xS`, `excess`)

## Diagnóstico por frame (tooltip)

Os exemplos também colocam um payload pequeno em cada frame:

- `series[].frames[].highlights.diagnostics.core.blocking_coalition_mask`
- `series[].frames[].highlights.diagnostics.core.blocking_players`

Isso é útil para tooltip seguindo o mouse: ao passar em um ponto/segmento, mostrar a coalizão bloqueadora e as coordenadas.

