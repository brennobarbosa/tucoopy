# tucoopy (Python)

<p class="centered-logo">
  <img src="assets/logo.png" width="320" alt="tucoopy">
</p>

`tucoopy` é uma biblioteca Python para **Teoria de Jogos Cooperativos com Utilidade Transferível (TU)**.
Ela oferece:

 - Geradores clássicos de jogos (luvas, votação ponderada, aeroporto, falência, unanimidade, ...)
 - Conceitos clássicos de solução (Shapley, Banzhaf, least-core / nucleolus, kernel / prekernel, valor τ, ...)
 - Utilitários geométricos voltados para **visualização** (core, ε-core, conjunto de imputações, conjunto de Weber, conjunto de barganha)
 - Um gerador de **especificações de animação em JSON**, compatível com o pacote `tucoopyjs`.

## Escopo e objetivos de design

 - Foco em jogos cooperativos TU (jogos em função característica).
 - Manter uma API limpa e bem estruturada: uma superfície pequena no nível superior, com a maior parte da funcionalidade organizada em subpacotes.
 - Priorizar correção e diagnósticos claros em vez de desempenho máximo; muitas rotinas são exponenciais e pensadas para valores pequenos de `n`.
 - Dependências pesadas opcionais:

  - `tucoopy[lp]` habilita métodos baseados em programação linear via `SciPy` (least-core, nucleolus, balanceamento, conjunto de barganha, ...)
  - `tucoopy[lp_alt]` motor alternativo para programação linear via `PuLP` 
  - `tucoopy[fast]` habilita utilitários baseados em `NumPy` (kernel / prekernel)
  - `tucoopy[viz]` habilita visualização simples para jogos com 2 ou 3 jogadores baseada em `Matplotlib`

## Links rápidos

Se você é novo no pacote, comece por aqui:

* [Início Rápido](guides/quickstart.md): instalação + exemplos mínimos.
* [API](reference/index.md): mapa da API pública (nível superior estável vs. subpacotes)
*  [Um pouco de teoria](theory/index.md): conceitos de solução (Shapley, Banzhaf, nucleolus, kernel, ...)
* `geometry.md`: objetos geométricos para visualização (core, ε-core, imputações, Weber, barganha, ...)
* `animation_spec.md`: geração de especificações JSON consumidas pelo renderizador JS
- [Roadmap](project/roadmap.md): checklist de implementação / próximos passos
- [Como contribuir](project/contributions.md): checklist de implementação / próximos passos

## Exemplo mínimo

```py
from tucoopy import Game
from tucoopy.solutions import shapley_value

g = Game.from_value_function(n_players=3, value_fn=lambda S: float(len(S)))
print(shapley_value(g))  # [1.0, 1.0, 1.0]
```
