# Notação e convenções

Esta seção resume a notação usada nas páginas de teoria e na biblioteca.  
Seguimos a notação padrão da **teoria de jogos cooperativos com utilidade transferível (TU)**, com pequenas convenções escolhidas por clareza e facilidade de implementação.

O objetivo não é introduzir novos conceitos, e sim fixar uma linguagem comum para que definições, algoritmos e saídas sejam interpretados de forma consistente.

## Jogadores e coalizões

- **Jogadores** são indexados pelo conjunto finito $N = \{1, \ldots, n\}.$

- Uma **coalizão** é qualquer subconjunto $S \subseteq N$.

- A **grande coalizão** é o conjunto de todos os jogadores, denotado pelo próprio $N$.

No código, os jogadores são indexados de `0` a `n-1`, seguindo a convenção padrão do Python.  
Coalizões são representadas internamente como **máscaras de bits**, mas a maioria das funções voltadas ao usuário aceita iteráveis do Python (listas, tuplas ou conjuntos de índices de jogadores).

!!! tip "Intuição"
    Uma coalizão é simplesmente um grupo de jogadores agindo em conjunto.  
    A grande coalizão representa cooperação total entre todos os jogadores.

## Função característica

Um jogo cooperativo TU é descrito por uma **função característica**

$$
v : 2^N \to \mathbb{R},
$$

que atribui um valor real a cada coalizão, com a normalização

$$
v(\emptyset) = 0.
$$

O valor $v(S)$ representa o valor total que a coalizão $S$ consegue gerar por conta própria, assumindo que seus membros cooperam plenamente e podem transferir utilidade livremente entre si.

!!! tip "Intuição"
    Pense em $v(S)$ como o “*tamanho da torta*” disponível para a coalizão $S$. Como essa torta é dividida vem depois.

## Alocações e eficiência

!!! note "Definição"
    Uma **alocação** é um vetor

    $$
    x = (x_1, \dots, x_n) \in \mathbb{R}^n,
    $$

    onde $x_i$ denota o payoff atribuído ao jogador $i$.

    Uma alocação é **eficiente** se

    $$
    \sum_{i \in N} x_i = v(N).
    $$

!!! tip "Intuição"
    Eficiência significa que todo o valor criado pela cooperação total é distribuído entre os jogadores.  
    Nada é perdido e nada fica sem ser distribuído.

??? example "Exemplo"
    Para um jogo aditivo definido por $v(S) = |S|$ com $n=3$, a grande coalizão tem valor
    
    $$
    v(N) = 3.
    $$

    Uma alocação eficiente natural é
    
    $$
    x = (1, 1, 1),
    $$

    onde cada jogador recebe exatamente sua contribuição isolada.

## Somas coalizionais e excesso

!!! note "Definição"
    Dada uma alocação $x$ e uma coalizão $S \subseteq N$, a **soma coalizional** é
    
    $$
    x(S) = \sum_{i \in S} x_i.
    $$

    O **excesso** da coalizão $S$ na alocação $x$ é definido como
    
    $$
    e(S, x) = v(S) - x(S).
    $$

!!! tip "Intuição"
    O excesso mede o quanto uma coalizão está insatisfeita.
    
    - Se $e(S, x) > 0$, a coalizão $S$ consegue fazer melhor sozinha do que sob a alocação $x$.
    - Se $e(S, x) = 0$, a coalizão está exatamente satisfeita.
    - Se $e(S, x) < 0$, a coalizão recebe mais do que seu valor “sozinha”.

O conceito de excesso é central em muitos conceitos de solução, especialmente os relacionados a **estabilidade**, como o núcleo, o ε-núcleo e o nucleolus.
