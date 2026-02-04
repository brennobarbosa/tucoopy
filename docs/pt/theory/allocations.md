# Dividindo o valor: alocações e imputações

Agora que entendemos que um jogo cooperativo é descrito por uma função $v(S)$, uma pergunta surge naturalmente:

> Como dividir o valor gerado pela grande coalizão $N$?

Sabemos quanto cada grupo consegue gerar.
Mas ainda não falamos sobre **como esse valor é distribuído entre os jogadores**.

É aqui que surge um novo objeto central da teoria.

---

## Alocações

Uma forma de dividir o valor entre os jogadores é por meio de um vetor

$$
x = (x_1, x_2, \dots, x_n) \in \mathbb{R}^n,
$$

onde $x_i$ representa quanto o jogador $i$ recebe.

Esse vetor é chamado de **alocação**.

Ele não depende mais de coalizões.
Ele descreve apenas **o resultado final da cooperação**.

---

## A primeira exigência natural: usar todo o valor disponível

Se os jogadores juntos conseguem gerar $v(N)$, parece razoável exigir que tudo isso seja distribuído:

$$
\sum_{i \in N} x_i = v(N).
$$

Se a soma for menor, estamos “jogando fora” valor.
Se for maior, estamos distribuindo algo que não existe.

Essa condição é chamada de **eficiência**.

---

## A segunda exigência natural: ninguém aceitar receber menos do que sozinho

Cada jogador sabe quanto consegue gerar por conta própria: $v(\{i\})$.

Portanto, uma divisão só faz sentido se

$$
x_i \ge v(\{i\}), \quad \text{para todo } i \in N.
$$

Caso contrário, o jogador preferiria abandonar a cooperação.

Essa condição é chamada de **racionalidade individual**.

---

## Imputações

Quando uma alocação satisfaz essas duas propriedades — eficiência e racionalidade individual — ela recebe um nome especial.

Chamamos essa alocação de **imputação**.

Em termos matemáticos, o conjunto de todas as imputações é

$$
\left\{ x \in \mathbb{R}^n \,:\, \sum_{i \in N} x_i = v(N) \text{ e } x_i \ge v({i}) \ \forall i \right\}.
$$

Esse conjunto é chamado de **conjunto das imputações**.

---

## O que isso representa intuitivamente?

Uma imputação é simplesmente uma forma de dividir o valor total que:

* usa exatamente tudo o que foi gerado, e
* garante que ninguém sai prejudicado em relação a agir sozinho.

Nada foi dito ainda sobre justiça.
Nada foi dito ainda sobre estabilidade.
Nada foi dito ainda sobre poder.

Estamos apenas descrevendo **as divisões que fazem sentido considerar**.

---

## Visualizando isso

Geometricamente, as imputações formam um subconjunto de um hiperplano em $\mathbb{R}^n$: todas as divisões possíveis do valor total, limitadas pelo fato de que cada jogador precisa receber pelo menos o que conseguiria sozinho.

Esse é o “espaço” onde todas as soluções clássicas da teoria vão viver.

<div class="cg-figure">
  <svg
    class="cg-svg"
    viewBox="0 0 600 450"
    role="img"
    aria-label="Imputation set para 3 jogadores (v(N)=100, v({i})=0)"
  >
    <!-- Triângulo: x1+x2+x3=100 com xi>=0 -->
    <path
      id="imputation-set"
      class="region imputation"
      d="M 300 60 L 100 420 L 500 420 Z"
      data-tex=""
      fill="currentColor"
      fill-opacity="0.4"
    ></path>

    <path
      class="outline"
      d="M 300 60 L 100 420 L 500 420 Z"
      fill="none"
      stroke="currentColor"
    ></path>

    <!-- Vértices / rótulos (opcional) -->
    <circle class="vertex" cx="300" cy="60" r="2"></circle>
    <circle class="vertex" cx="100" cy="420" r="2"></circle>
    <circle class="vertex" cx="500" cy="420" r="2"></circle>

    <text class="label" x="300" y="40" text-anchor="middle">P1</text>
    <text class="label" x="85" y="445" text-anchor="end">P2</text>
    <text class="label" x="515" y="445" text-anchor="start">P3</text>
  </svg>
  <div class="cg-caption">
    Conjunto das imputações do exemplo do gasoduto: <span class="arithmatex">$ \{(x_1,x_2,x_3) \, : \, x_1+x_2+x_3 = 100,\; x_i \ge 0\} $</span>
  </div>
  <!-- Tooltip bonito (opcional; usado pelo JS abaixo) -->
  <div class="cg-tooltip" id="cgTip"></div>
</div>


---

## Por que isso é um passo tão importante?

Porque agora podemos reformular todas as perguntas da teoria de forma muito clara:

* Entre todas as imputações, quais são estáveis?
* Entre todas as imputações, quais são justas?
* Entre todas as imputações, quais refletem melhor o poder de cada jogador?

Os conceitos que vêm a seguir — núcleo, valor de Shapley, núcleo mínimo, kernel, índices de poder — nada mais são do que diferentes maneiras de escolher pontos dentro desse conjunto.

A teoria dos jogos cooperativos, a partir daqui, passa a ser a arte de selecionar boas imputações.
