# Estabilidade: bloqueio e núcleo (Core)

Até aqui, descrevemos o conjunto das imputações: divisões que usam todo o valor e garantem que ninguém sai pior do que agindo sozinho.

Mas ainda falta uma pergunta essencial.

Mesmo que uma divisão pareça razoável, ela é **sustentável**?

Ou seja:

> Existe algum grupo de jogadores que preferiria sair da grande coalizão e fazer um acordo por conta própria?

Essa é a ideia central por trás da **estabilidade** em jogos cooperativos.

---

## A ideia de “desvio” em jogos cooperativos

Considere uma imputação $x \in \mathbb{R}^n$.
Ela diz quanto cada jogador recebe quando todos cooperam.

Agora pegue uma coalizão $S \subseteq N$. Esse grupo sabe que, se se separar, consegue gerar $v(S)$.

A pergunta é: será que eles conseguem dividir $v(S)$ **entre eles** de um jeito que deixe **todo mundo em $S$ estritamente melhor** do que em $x$?

Se sim, então $x$ não se sustenta: aquele grupo tem incentivo para romper o acordo.

---

## Pagamento total de uma coalizão

Dado um vetor $x$, vamos escrever o total que $x$ dá para uma coalizão $S$ como

$$
x(S) := \sum_{i\in S} x_i.
$$

Essa notação é útil porque a coalizão compara “quanto recebe no acordo atual” com “quanto consegue garantir sozinha”.

---

## Bloqueio

Dizemos que uma coalizão $S$ pode **bloquear** uma imputação $x$ se ela consegue garantir valor suficiente para melhorar a vida de todos dentro de $S$.

Uma condição simples (e muito usada) para isso é:

$$
v(S) > x(S).
$$

!!! tip "Interpretação"
    O grupo $S$ consegue gerar mais do que está recebendo no acordo atual, e portanto tem “margem” para propor um acordo alternativo que beneficie seus membros.

> Se existe algum $S$ que bloqueia $x$, então $x$ não é estável: há um desvio coalicional plausível.

---

## O núcleo (Core)

O **núcleo** é o conjunto de imputações que **não podem ser bloqueadas por nenhuma coalizão**.

Em termos práticos: são as divisões em que *ninguém*, em nenhum grupo, tem incentivo para abandonar o acordo.

Matematicamente, o núcleo é o conjunto de alocações $x$ tais que:

$$
\sum_{i \in N} x_i = v(N)
\quad\text{e}\quad
x(S) \ge v(S)\; \text{para todo } S \subseteq N.
$$

A condição $x(S) \ge v(S)$ diz:

> “qualquer coalizão $S$ já recebe pelo menos o que conseguiria garantir sozinha”.

---

## Lendo isso geometricamente (caso de 3 jogadores)

No caso $n=3$, o conjunto das imputações é um triângulo (como você viu anteriomente).
Cada restrição do tipo

$$
x(S) \ge v(S)
$$

vira um **meio-plano** cortando esse triângulo.

O núcleo é simplesmente a **interseção** de todos esses cortes.

Por isso ele costuma ser um polígono[^1] menor dentro do triângulo — e às vezes… pode nem existir.

[^1]: Em dimensões maiores chamamos de um polítopo, que é a interseção de semiespaços.

---

## Um fato importante: o núcleo pode ser vazio

Mesmo quando a grande coalizão gera muito valor, pode acontecer de não existir nenhuma imputação que satisfaça simultaneamente todas as restrições $x(S) \ge v(S)$.

Intuitivamente, isso ocorre quando as coalizões “parciais” são fortes demais: sempre existe algum grupo capaz de exigir mais do que o acordo atual consegue oferecer sem violar outra restrição.

Quando o núcleo é vazio, precisamos de outras noções de estabilidade — mais flexíveis — como **least-core** e **nucleolus**.

(E é aqui que a teoria começa a ficar realmente interessante.)