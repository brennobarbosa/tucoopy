# Estrutura do tucoopy-py

Este pacote é organizado para manter a API pública explícita, pequena e fácil de navegar.

## Módulos canônicos

- `tucoopy.base`
  - estruturas fundamentais (`Coalition` bitmask, `Game`)

- `tucoopy.properties`
  - propriedades / reconhecedores

- `tucoopy.io`
  - IO JSON + helpers de animation spec (contrato compartilhado com o JS)

- `tucoopy.backends`
  - adapters para dependências opcionais (LP, NumPy, ...)

- `tucoopy.power`
  - índices de poder (votação / jogos simples)

- `tucoopy.solutions`
  - conceitos de solução (Shapley, Banzhaf, ...)

- `tucoopy.transforms`
  - representações / transformações (ex.: dividendos de Harsanyi)

- `tucoopy.geometry`
  - objetos/operações geométricas usadas para visualização (ex.: vértices do núcleo)

## Regras de importação (restritas)

- Código interno deve importar dos módulos canônicos acima.
- Re-exports públicos ficam em:
  - o `__init__.py` de cada subpacote (ex.: `tucoopy.geometry.__init__`)
  - o `tucoopy/__init__.py` no topo, por conveniência

Evite introduzir “shims”/atalhos de módulo extras como `tucoopy/core.py`: isso deixa o pacote mais difícil de navegar.