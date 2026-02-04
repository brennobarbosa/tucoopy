# Estrutura do tucoop-py

Este pacote é organizado para manter a API pública explícita, pequena e fácil de navegar.

## Módulos canônicos

- `tucoop.base`
  - estruturas fundamentais (`Coalition` bitmask, `Game`)

- `tucoop.properties`
  - propriedades / reconhecedores

- `tucoop.io`
  - IO JSON + helpers de animation spec (contrato compartilhado com o JS)

- `tucoop.backends`
  - adapters para dependências opcionais (LP, NumPy, ...)

- `tucoop.power`
  - índices de poder (votação / jogos simples)

- `tucoop.solutions`
  - conceitos de solução (Shapley, Banzhaf, ...)

- `tucoop.transforms`
  - representações / transformações (ex.: dividendos de Harsanyi)

- `tucoop.geometry`
  - objetos/operações geométricas usadas para visualização (ex.: vértices do núcleo)

## Regras de importação (restritas)

- Código interno deve importar dos módulos canônicos acima.
- Re-exports públicos ficam em:
  - o `__init__.py` de cada subpacote (ex.: `tucoop.geometry.__init__`)
  - o `tucoop/__init__.py` no topo, por conveniência

Evite introduzir “shims”/atalhos de módulo extras como `tucoop/core.py`: isso deixa o pacote mais difícil de navegar.