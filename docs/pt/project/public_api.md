# API publica (contrato minimo)

Este documento define o **contrato publico minimo** do `tucoopy` (Python).
Ele existe para reduzir retrabalho: os imports listados aqui devem permanecer estaveis.

## Regra geral

- O top-level `tucoopy` e **pequeno** e serve como "porta de entrada".
- A superficie completa vive nos subpacotes (`tucoopy.geometry`, `tucoopy.solutions`, etc.).
- Se algo nao estiver documentado aqui, pode mudar com mais liberdade (especialmente no `0.x`).

## Imports canônicos (estáveis)

Recomendado para usuarios:

```py
from tucoopy import Game
from tucoopy.games import weighted_voting_game
from tucoopy.solutions import shapley_value, nucleolus
from tucoopy.geometry import Core, EpsilonCore, LeastCore
from tucoopy.power import banzhaf_index, shapley_shubik_index
```

## Top-level (`tucoopy`)

O top-level deve expor apenas alguns itens de alto nivel (conveniencia).
O restante deve ser importado dos subpacotes.

Itens expostos atualmente:

- `Game`
- `mask_from_players`
- `glove_game`, `weighted_voting_game`
- `Core`
- `shapley_value`
- `nucleolus` (requer backend de LP quando chamado)

## Subpacotes (source of truth)

- `tucoopy.base`: primitivas (coalizoes, jogos, config, types/exceptions)
- `tucoopy.games`: geradores de jogos classicos
- `tucoopy.solutions`: solucoes pontuais (vetor payoff)
- `tucoopy.geometry`: conjuntos/poliedros (core, least-core, etc.)
- `tucoopy.diagnostics`: checks e explicacoes (por set e por alocacao)
- `tucoopy.power`: indices de poder para jogos simples/votacao
- `tucoopy.transforms`: transformacoes e representacoes
- `tucoopy.io`: especificacoes JSON e schema
- `tucoopy.backends`: adaptadores para dependencias opcionais

## O que e experimental?

Enquanto o projeto estiver no `0.x`, consideramos mais sujeito a mudanca:

- detalhes de diagnosticos (campos adicionais, estruturas internas);
- implementacoes de LP e explicacoes detalhadas do solver;
- algumas utilidades de visualizacao e sampling.

Quando um recurso migrar de "experimental" para "estavel", ele deve:

- ter docstring completa (Numpy style),
- ter testes unitarios,
- entrar neste documento.

