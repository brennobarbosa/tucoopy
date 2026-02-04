# Critica (bem direta) do projeto `tucoopy`

Este documento e propositalmente critico. A ideia e registrar:

- pontos fracos reais (tecnicos e de produto),
- riscos de manutencao,
- lacunas de teoria/estado-da-arte,
- oportunidades de performance/robustez,

e sugerir a menor lista de mudancas que aumenta muito a previsibilidade do projeto.

## 0) Resumo executivo

O `tucoopy` ja esta em um bom ponto como **MVP** (TU games, solucoes e conjuntos principais, contrato JSON + renderer JS).
O maior risco hoje nao e "falta de feature", e sim:

1) **robustez numerica + LP** (degenerescencia, tolerancias, solver differences),
2) **contrato publico** (evitar drift e refactors repetidos),
3) **performance previsivel** (limites claros e caminhos aproximados).

## 1) Pontos fortes (onde o projeto esta acima da media)

- **Arquitetura por camadas** razoavelmente clara:
  - base (game/coalition/config),
  - solutions (single-valued),
  - geometry (set-valued / poliedros),
  - diagnostics (explicacoes),
  - io/schema (contrato),
  - JS renderer separado.
- **Contrato JSON** como interface entre Python e JS: isso e uma decisao correta e escalavel.
- **Dependencias opcionais** bem encaminhadas: `lp`, `fast`, `viz` (sem forcar SciPy/NumPy/Matplotlib no core).
- **Testes** ja existem e cobrem fluxos reais (isso e raro em libs matematicas pequenas).

## 2) Onde esta fragil (pontos fracos importantes)

### 2.1 Robustez numerica (LP e poliedros)

Problema: teoria de jogos cooperativos vira LP/poliedros rapidamente, e isso e uma area onde:

- degenerescencia e comum,
- tolerancias mudam resultado (principalmente "tight" constraints),
- metodos diferentes de solver (SciPy/HiGHS vs PuLP/CBC) podem divergir em detalhes,
- enumeracao de vertices pode explodir ou falhar silenciosamente.

Melhorias que valem muito:

- padronizar *sempre* o uso de `tol` (em um lugar) e documentar o que significa (feasibility vs dedup),
- garantir que diagnosticos retornem "por que deu estranho" (slacks, residuals, affine dimension, boundedness),
- tornar mais explicito o "contrato" do backend de LP (o que e esperado e o que pode vir `None`).

### 2.2 Performance: riscos de uso ingenuo

O pacote tem rotinas intrinsicamente explosivas:

- coalizoes: $2^n$,
- Weber set: $n!$,
- vertices/projecoes: combinatorial,
- kernel/prekernel: iterativo.

Risco: usuario chama uma funcao em `n=20` e acha que "travou".

Melhorias:

- manter limites default conservadores,
- mensagens de erro sempre explicarem o custo e a alternativa (sampling/approx),
- oferecer caminhos "approx por amostragem" sempre que fizer sentido (principalmente geometry/project).

### 2.3 API/contrato: muita mudanca estrutural custa caro

Se o usuario nao sabe quais imports sao estaveis, ele vira refem do repo.

O projeto ja comecou a melhorar isso com "API publica minima", mas ainda falta:

- aplicar de verdade uma politica de deprecacao (mesmo no 0.x),
- evitar "arquivos fantasmas" (principalmente em Windows/OneDrive),
- manter docs como parte do contrato (CI ja ajuda bastante).

## 3) Funcoes ausentes / gaps (conteudo e estado da arte)

### 3.1 Solucoes/sets (teoria de jogos cooperativos)

Mesmo ignorando "public goods" e indices raros, ha lacunas naturais:

- **nucleolus**: existem algoritmos de "constraint generation" e criterios de Kohlberg que sao estado da arte para escalar melhor do que enumerar tudo de uma vez (depende do escopo desejado).
- **kernel/prekernel**: normalmente exigem cuidados com estabilidade numerica, criterios de parada, e diagnosticos (na pratica, "convergiu" e "qualidade do ponto" importam).
- **bargaining set** e variantes: computacao exata e pesada; sampling e ok, mas precisa diagnostico bom (falso positivo/negativo).
- **vertex enumeration**: para ir alem de `n=3`, o caminho "correto" geralmente usa ferramentas como `cddlib`/`pycddlib` ou `polymake` (provavelmente opcional, mas vale documentar).

### 3.2 Propriedades e reconhecedores

Comparando com toolboxes maduras (MATLAB TuGames, R CoopGame), normalmente existe:

- catalogo maior de "sanity checks" e exemplos canonicos,
- funcoes auxiliares para normalizacao e transformacoes padrao,
- mais testes "teoricos" (nao so de software).

Aqui o projeto ja evoluiu, mas ainda pode melhorar:

- checks de convexidade/balancedness com diagnosticos melhores,
- exemplos pequenos para cada propriedade (contraexemplos incluidos).

### 3.3 Indices de poder

O pacote tem um bom conjunto de indices classicos, mas sempre vai existir "mais um indice".
O maior risco aqui nao e faltar indice, e sim:

- definicoes inconsistentes (normalizacao, dominio: simple vs TU),
- documentacao confusa do que e suportado.

## 4) Comparacao com outros sistemas (onde o `tucoopy` se posiciona)

### MATLAB (TuGames toolbox, etc.)

Normalmente: muito conteudo, funcoes prontas e exemplos; mas:

- depende de MATLAB,
- integracao web/renderer nao e foco,
- arquitetura nem sempre e modular.

`tucoopy` pode competir bem se priorizar:

- contrato claro,
- docs boas,
- instalacao simples + extras opcionais.

### R (CoopGame e afins)

R e forte em analise e visualizacao rapida, mas:

- performance/engenharia pode ser desigual,
- integracao com um renderer JS customizado e rara.

### Bibliotecas de poliedros (cddlib/polymake)

Essas sao "estado da arte" para V/H-rep e vertices.
O `tucoopy` nao precisa competir diretamente: ele pode integrar opcionalmente, ou usar como referencia.

## 5) Oportunidades concretas de performance

Coisas que normalmente dao salto de performance sem reescrever o mundo:

- **evitar loops Python** onde houver soma sobre coalizoes repetida: utilitarios (ex.: `coalition_sum`) ajudam.
- **cache** onde o custo e alto mas o input se repete (value function + coalition iterators).
- **NumPy opcional** para rotinas de algebra linear pequenas (ja existe `fast`, mas pode ser expandido com cuidado).
- **transformadas** em $O(n 2^n)$ (Mobius) ja e um passo certo; aplicar o mesmo raciocinio onde houver $3^n$ escondido.

O que NAO vale para MVP (custo alto):

- implementar vertex enumeration generica para `n>3` sem backend especializado,
- otimizar nucleolus/kernels para `n` grande sem decidir claramente o publico-alvo.

## 6) Recomendacao de prioridades (ordem sugerida)

1) **Estabilizar contrato e docs** (ja avancou muito com CI + public_api + deprecation).
2) **Melhorar diagnosticos numericos** (explicar "por que" e nao so "deu erro").
3) **Mais testes teoricos** (canonicos e contraexemplos).
4) **Performance pontual** (hotspots reais medidos, nao "otimizacao por intuicao").

## 7) Conclusao

O projeto esta bem encaminhado e tem potencial real de ser referencia leve/educacional + utilitaria.
Para isso, o foco tem que ser previsibilidade:

- o que e estavel,
- o que e caro,
- o que e aproximado,
- e como depurar quando o solver/numero "nao bate".
