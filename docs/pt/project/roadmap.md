# Roadmap (do zero)

Este roadmap foi resetado para iniciar uma nova fase do projeto (lancamento do produto).
Use as caixinhas para acompanhar o andamento.

## Legenda
- `[x]` feito
- `[ ]` pendente
- `(*)` opcional / depois

---

## 1) MVP (minimo teorico + estabilidade)

- [ ] Definir escopo do MVP (o que entra / o que fica fora)
- [ ] Fixar "API publica minima" e imports canonicos
- [ ] Revisar numerica / tolerancias (documentar `tol` e limites)
- [ ] Revisar mensagens de erro (padrao + extras opcionais)
- [ ] Revisar contrato JSON (schema + exemplos)

## 2) Documentacao PT (release)

- [ ] Revisar paginas de teoria (minimo teorico)
- [ ] Revisar guias (quickstart, diagnostics, large games, performance)
- [ ] Revisar pagina de exemplos (mais exemplos curtos e canonicos)
- [ ] Revisar referencia (mkdocstrings): docstrings com exemplos onde fizer sentido
- [ ] Checar navegacao do MkDocs (nav completo e sem paginas orfas)

## 3) Qualidade (CI + testes)

- [ ] Rodar `pytest` em Windows e Linux (CI)
- [ ] Rodar `mypy` em Windows e Linux (CI)
- [ ] Rodar `mkdocs build` (PT) como parte do CI
- [ ] Aumentar cobertura em pontos com maior risco (LP/geometry/diagnostics)
- [ ] Avaliar possibilidade de paralelizacao (hotspots em loops de coalizoes, sampling e scans de excessos)

## 4) Produto (empacotamento)

- [ ] Revisar `pyproject.toml` (extras, classifiers, python_requires)
- [ ] Revisar README do pacote Python (curto, com tabela "feature -> extra")
- [ ] Revisar `CHANGELOG.md` (minimo para release)
- [ ] Preparar checklist de release (tag, build, publish)

## 5) Depois (nao bloquear o lancamento)

- [ ] (*) Traduzir docs para EN
- [ ] (*) Integracao opcional com backend de poliedros (cddlib/polymake) para `n>3`
- [ ] (*) Mais solucoes/sets (conforme necessidade)
- [ ] (*) Paralelizacao: implementar via `concurrent.futures` (process) em tarefas naturalmente independentes (permutacoes, coalizoes, amostragem), com controle por `max_workers`
