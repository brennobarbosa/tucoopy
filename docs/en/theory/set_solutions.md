# Núcleo e poliedros relacionados

Muitos objetos de solução são definidos como poliedros.

## Núcleo

Alocações no núcleo satisfazem:

- Eficiência: $\sum_i x_i = v(N)$
- Racionalidade coalizional: $x(S) \ge v(S)$ para todo $S$ não-vazio e próprio

!!! note "Definição (núcleo)"
    O núcleo de um jogo TU $(N,v)$ é o conjunto de alocações $x \in \mathbb{R}^n$ tais que:

    $$\sum_{i \in N} x_i = v(N),$$
    e para toda coalizão não-vazia e própria $S$,
    $$x(S) = \sum_{i \in S} x_i \ge v(S).$$

!!! tip "Intuição"
    Nenhuma coalizão consegue desviar com ganho: toda coalizão recebe pelo menos o que consegue garantir sozinha.

## ε-núcleo / least-core

O ε-núcleo relaxa a racionalidade coalizional por $\epsilon \ge 0$:

- $x(S) \ge v(S) - \epsilon$

O least-core escolhe o menor $\epsilon$ possível (calculado via LP quando o SciPy está disponível).

## Conjunto de imputações

- Eficiência + racionalidade individual ($x_i \ge v(\{i\})$)

!!! note "Definição (conjunto de imputações)"
    O conjunto de imputações é:

    $$I(v) = \left\{x \in \mathbb{R}^n : \sum_{i \in N} x_i = v(N),\; x_i \ge v(\{i\})\;\forall i\right\}.$$

Para detalhes voltados à implementação, veja `geometry.md`.
