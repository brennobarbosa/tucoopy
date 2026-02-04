<!--
Este guia é curto por design. A documentação de teoria/API vive em outras páginas.
-->

# Contribuições

Este documento descreve um fluxo simples para contribuir com o `tucoop` (foco: `tucoop-py`).

## Setup (dev)

Recomendado: criar um ambiente virtual na raiz de `packages/tucoop-py` e instalar o pacote em modo editável.

```bash
cd packages/tucoop-py
python -m venv .venv
```

Ative o ambiente e instale com extras:

```bash
pip install -e ".[dev,docs]"
```

Extras opcionais:

- `lp`: `pip install -e ".[lp]"` (SciPy; habilita métodos baseados em LP)
- `lp_alt`: `pip install -e ".[lp_alt]"` (PuLP; fallback)
- `fast`: `pip install -e ".[fast]"` (NumPy; speedups)
- `viz`: `pip install -e ".[viz]"` (Matplotlib; visualização estática)

## Rodar checks locais

### Testes

```bash
pytest -q
```

### Type checking

```bash
mypy src/tucoop
```

### Docs (MkDocs)

Build da documentação em português:

```bash
mkdocs build -f mkdocs.pt.yml --dirty
```

## Regras de qualidade (projeto)

- Evitar duplicação: manter um “source of truth” por conceito/módulo.
- Não usar shims/aliases para “compatibilidade”: prefira imports canônicos.
- Preferir docstrings em **Numpy style** (é o estilo configurado no mkdocstrings).

## Windows/OneDrive

Se estiver desenvolvendo dentro do OneDrive, veja:

- `guides/dev_windows_onedrive.md`

## Pull requests

Antes de abrir PR:

- rode `pytest` e `mypy`;
- valide `mkdocs build` (PT) se mexer em docs/docstrings;
- descreva impacto de API e mudanças em extras/dependências opcionais quando aplicável.
