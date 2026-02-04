# O jogo cooperativo como uma função de valor das coalizões

Depois da intuição inicial sobre cooperação, precisamos agora dar um passo importante: descrever essa situação de forma precisa.

A ideia central dos jogos cooperativos é surpreendentemente simples.

Não modelamos estratégias.
Não modelamos decisões passo a passo.
Não modelamos sequências de jogadas.

Modelamos apenas isto:

> **quanto valor cada grupo possível de jogadores consegue gerar quando coopera.**

---

## Jogadores e coalizões

Considere um conjunto finito de jogadores

$$
N = \{1,2,\dots,n\}.
$$

Qualquer subconjunto $S \subseteq N$ é chamado de **coalizão**.

Isso inclui:

* jogadores sozinhos, como $\{i\}$,
* grupos intermediários, como $\{1,3,4\}$,
* e o grande grupo $N$, contendo todos.

A pergunta central passa a ser:

> Se exatamente os jogadores de $S$ cooperarem entre si, quanto eles conseguem gerar?

---

## A função característica

Essa pergunta é respondida por uma função

$$
v : 2^N \longrightarrow \mathbb{R},
$$

onde $2^N$ é o conjunto de todas as coalizões possíveis.

Para cada $S \subseteq N$, o número $v(S)$ representa o valor total que os jogadores de $S$ conseguem gerar ao cooperar.

Podemos agora escrever isso de forma precisa.

Um jogo cooperativo (com utilidade transferível) é descrito por um par $(N,v)$, onde

$$
v:2^N \to \mathbb{R}, \qquad v(\emptyset)=0.
$$

Essa função é chamada de **função característica** ou **função de coalizão** do jogo.

---

## O que essa função realmente significa?

Dependendo do contexto, $v(S)$ pode representar:

* lucro gerado por um consórcio de empresas,
* economia de custos ao compartilhar infraestrutura,
* votos obtidos por uma coalizão política,
* capacidade de produção conjunta,
* redução de risco ao atuar em grupo.

O ponto essencial é que $v(S)$ mede o **valor total disponível** para aquele grupo.

Pense em $v(S)$ como o *tamanho da torta* disponível para a coalizão $S$. Como essa torta é dividida vem depois.

---

## Um exemplo concreto

Considere três empresas interessadas em construir um gasoduto:

$$
N = \{1,2,3\}.
$$

Sozinhas, nenhuma consegue viabilizar o projeto:

$$
v(\{1\}) = v(\{2\}) = v(\{3\}) = 0.
$$

Mas juntas em pares, já conseguem extrair algum valor:

$$
v(\{1,2\}) = 80, \quad
v(\{1,3\}) = 60, \quad
v(\{2,3\}) = 70.
$$

E as três juntas conseguem um resultado melhor ainda:

$$
v(\{1,2,3\}) = 100.
$$

Essa tabela **é o jogo inteiro**. Nada mais é necessário.

---

## O que *não* estamos modelando

Note o que ficou de fora:

* quem propôs a aliança,
* quem negociou com quem,
* qual foi a ordem das decisões,
* quais estratégias foram usadas.

Tudo isso desaparece.

Resta apenas a **estrutura de valor da cooperação**.

---

## Por que isso é poderoso?

Porque, a partir apenas dessa função $v$, podemos começar a fazer perguntas profundas:

* Como dividir $v(N)$ entre os jogadores?
* Quem é essencial para gerar valor?
* Existem divisões estáveis?
* O que é uma divisão justa?
* Quem tem mais poder dentro da estrutura de coalizões?

Todos os conceitos clássicos da teoria dos jogos cooperativos nascem exclusivamente dessa função.

---

## Uma mudança de perspectiva importante

Em jogos não cooperativos, o foco está no comportamento dos jogadores.

Aqui, o foco está na **estrutura matemática da cooperação**.

O jogo não é uma sequência de ações.

O jogo é uma **tabela de valores das coalizões**.

E é a partir dessa tabela que toda a teoria será construída.

---

!!! tip "Uma hipótese forte (e talvez invisível)"
    Neste ponto, estamos assumindo algo muito forte.

    Estamos assumindo que, se uma coalizão gera valor $v(S)$, então esse valor pode ser dividido entre seus membros de qualquer maneira que desejarmos.

    Estamos assumindo que o valor é perfeitamente transferível.

    Isso é natural quando falamos de dinheiro, lucro, custos ou energia.

    Mas nem toda cooperação gera algo que possa ser redistribuído dessa forma.

    Às vezes, cooperar não produz “uma torta para dividir”, mas sim um conjunto de resultados possíveis, e alguns simplesmente não podem ser transformados uns nos outros por compensações.

    Quando isso acontece, o modelo que estamos usando deixa de ser adequado.

    Entramos então no mundo dos jogos com utilidade não-transferível.

    Eles existem. São importantes. E são matematicamente mais sutis.

    Mas, para quase tudo o que queremos estudar aqui — justiça, estabilidade e poder quando o valor pode ser redistribuído — o modelo que estamos usando, chamado de utilidade transferível (TU), é não apenas suficiente, mas extraordinariamente expressivo.
