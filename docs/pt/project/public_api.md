# API publica (contrato minimo)

Este documento define o **contrato publico minimo** do `tucoop` (Python).
Ele existe para reduzir retrabalho: os imports listados aqui devem permanecer estaveis.

## Regra geral

- O top-level `tucoop` e **pequeno** e serve como "porta de entrada".
- A superficie completa vive nos subpacotes (`tucoop.geometry`, `tucoop.solutions`, etc.).
- Se algo nao estiver documentado aqui, pode mudar com mais liberdade (especialmente no `0.x`).

## Imports canônicos (estáveis)

Recomendado para usuarios:

```py
from tucoop import Game
from tucoop.games import weighted_voting_game
from tucoop.solutions import shapley_value, nucleolus
from tucoop.geometry import Core, EpsilonCore, LeastCore
from tucoop.power import banzhaf_index, shapley_shubik_index
```

## Top-level (`tucoop`)

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

- `tucoop.base`: primitivas (coalizoes, jogos, config, types/exceptions)
- `tucoop.games`: geradores de jogos classicos
- `tucoop.solutions`: solucoes pontuais (vetor payoff)
- `tucoop.geometry`: conjuntos/poliedros (core, least-core, etc.)
- `tucoop.diagnostics`: checks e explicacoes (por set e por alocacao)
- `tucoop.power`: indices de poder para jogos simples/votacao
- `tucoop.transforms`: transformacoes e representacoes
- `tucoop.io`: especificacoes JSON e schema
- `tucoop.backends`: adaptadores para dependencias opcionais

## O que e experimental?

Enquanto o projeto estiver no `0.x`, consideramos mais sujeito a mudanca:

- detalhes de diagnosticos (campos adicionais, estruturas internas);
- implementacoes de LP e explicacoes detalhadas do solver;
- algumas utilidades de visualizacao e sampling.

Quando um recurso migrar de "experimental" para "estavel", ele deve:

- ter docstring completa (Numpy style),
- ter testes unitarios,
- entrar neste documento.

