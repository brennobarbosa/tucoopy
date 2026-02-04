# Jogos simples e votação ponderada

Um *jogo simples* tipicamente tem $v(S) \in \{0,1\}$ para toda coalizão $S$.

Jogos de votação ponderada são uma subclasse comum:

- Dados pesos $w_i$ e uma cota $q$, a coalizão $S$ é vencedora se $\sum_{i \in S} w_i \ge q$.

Índices de poder:

- Índice de Shapley–Shubik
- Índice de Banzhaf (normalizado)

No código, validamos a suposição de “jogo simples” antes de computar esses índices.

## Definição (jogo simples)

!!! note "Definição"
    Um jogo simples satisfaz $v(S) \in \{0,1\}$ para toda coalizão $S$.
    Em geral, $v(S)=1$ significa “vencedora” e $v(S)=0$ significa “perdedora”.

!!! tip "Intuição"
    Só importa o resultado sim/não; utilidades não são cardinais além disso.
