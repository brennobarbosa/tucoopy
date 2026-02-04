# How-to: n>4 (bundle em vez de geometria)

Para $n>4$, a UI não desenha um simplex completo; a estratégia recomendada é:

- Python exporta um `analysis.bundle` com tabelas/listas/resumos.
- O front apenas apresenta (tabelas, listas, tooltips), sem exigir backend para “desenhar”.

## Exemplo (Python)

```py
from tucoopy import Game
from tucoopy.io import build_analysis

g = Game.from_coalitions(
    n_players=6,
    values={(): 0.0, (0,1,2,3,4,5): 10.0},
    player_labels=[f"P{i+1}" for i in range(6)],
)

analysis = build_analysis(g, max_players=4, include_bundle=True)
print(analysis["bundle"]["game_summary"])
```

## O que esperar no JSON

- `analysis.sets`/`analysis.solutions` podem ser omitidos quando `n>max_players`.
- `analysis.meta.skipped` explica o motivo (ex.: `n=6 > max_players=4`).
- `analysis.bundle` contém um resumo leve e notas para guiar a UI.

Quando `n<=bundle_max_players` e o jogo é completo, o bundle pode incluir tabelas extras:

- `analysis.bundle.tables.players` (scalars por jogador)
- `analysis.bundle.tables.power_indices` (se for um jogo simples completo)
- `analysis.bundle.tables.tau_vectors` (vetores auxiliares do τ-valor)
- `analysis.bundle.tables.approx_solutions.shapley` (Shapley aproximado via amostragem, com `stderr`)
