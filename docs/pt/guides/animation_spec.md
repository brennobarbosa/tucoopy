# Especificação de animação (contrato Python -> JS)

O `tucoopy` consegue emitir uma “animation spec” em JSON que o renderizador JS pode consumir para desenhar alocações ao longo do tempo.

Arquivo de schema (neste monorepo):
- `schema/tucoopy-animation.schema.json`

## Dataclasses

O modelo de dados do lado Python vive em `tucoopy.io.animation_spec`:

- `AnimationSpec`
- `GameSpec` / `CharacteristicEntry`
- `SeriesSpec` / `FrameSpec`

Funções auxiliares:

- `game_to_spec(game)` converte um `Game` no `GameSpec` (amigável para JSON).
- `series_from_allocations(...)` constrói um `SeriesSpec` a partir de uma sequência de alocações.
- `build_animation_spec(...)` constrói um `AnimationSpec` “completo” (game + analysis + series), com highlights opcionais.

## Exemplo mínimo

```py
from tucoopy import Game, shapley_value
from tucoopy.io import build_animation_spec

g = Game.from_coalitions(
    n_players=3,
    values={
        (): 0.0,
        (0,): 1.0,
        (1,): 1.2,
        (2,): 0.8,
        (0, 1): 2.8,
        (0, 2): 2.2,
        (1, 2): 2.0,
        (0, 1, 2): 4.0,
    },
)

phi = shapley_value(g)
spec = build_animation_spec(
    g,
    schema_version="0.1.0",
    series_id="shapley",
    allocations=[phi] * 60,
    dt=1 / 30,
)
print(spec.to_json())
```

Notas:
- `analysis` é intencionalmente flexível, mas vale manter alinhado com o schema JSON.
- Para visualização, o pacote JS pode renderizar apenas até 4 jogadores (simplex até 3-simplex).

## Highlights por frame (`series[].frames[].highlights`)

Cada frame pode carregar um objeto `highlights` com informação extra para a UI (ex.: tooltip seguindo o mouse).

Convenção atual (opcional) usada pelos exemplos:

- `frame.highlights.diagnostics.core`: contém um payload pequeno com `max_excess` e `blocking_coalition_mask`.

## Proveniência (`analysis.meta`)

O `tucoopy.io.build_analysis(...)` preenche um bloco `analysis.meta` para registrar:

- `analysis.meta.computed_by`: quem gerou (ex.: `tucoopy-py`)
- `analysis.meta.build_analysis`: flags e parâmetros (ex.: `max_players`, `tol`, `diagnostics_top_k`)
- `analysis.meta.computed`: quais seções realmente foram incluídas (`solutions`, `sets`, `diagnostics`, `blocking_regions`)

## Diagnósticos (`analysis.diagnostics`)

Para apoiar a UI (tooltips/tabelas) sem precisar de backend, o Python pode anexar diagnósticos compactos em `analysis.diagnostics`.

Exemplo: para cada ponto em `analysis.solutions`, o `tucoopy.io.build_analysis(...)` pode incluir um resumo de pertinência ao núcleo:

- `analysis.diagnostics.solutions.<id>.core.in_core`
- `analysis.diagnostics.solutions.<id>.core.max_excess`
- `analysis.diagnostics.solutions.<id>.core.tight_coalitions` (coalizões que atingem o `max_excess`)
- `analysis.diagnostics.solutions.<id>.core.violations` (top-k coalizões que bloqueiam)

Também existe `analysis.diagnostics.input` para checks do próprio jogo (ex.: `vN`, `sum_singletons`, `essential`, e se o `characteristic_function` está completo para n pequeno).
