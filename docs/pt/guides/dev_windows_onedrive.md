# Desenvolvimento no Windows/OneDrive

Este guia foca em problemas comuns ao desenvolver o `tucoopy-py` no Windows quando o repo esta dentro do OneDrive.
O objetivo e evitar perda de tempo com IO lento, arquivos travados e caches inconsistentes.

## Sintomas comuns

- `pytest` fica muito mais lento do que o esperado.
- Arquivos `.pyc` e caches aparecem/desaparecem e geram diffs estranhos.
- O OneDrive "segura" arquivos em uso (lock) e algumas operacoes falham intermitentemente.
- `mkdocs build` pode ficar pesado ao reprocessar muitos arquivos.

## Recomendacoes (praticas)

### 1) Preferir desenvolver fora do OneDrive (recomendado)

Se der, mova o repo para um caminho que nao esteja sendo sincronizado (ex.: `C:\\dev\\tucoopy`).
Isso costuma resolver 80% dos problemas de performance/lock.

### 2) Se precisar ficar no OneDrive, reduza escrita de bytecode

Para evitar spam de `.pyc` e problemas de lock/cache:

- Configure `PYTHONPYCACHEPREFIX` para um diretorio fora do OneDrive.
- Ou, para sessoes curtas, use `PYTHONDONTWRITEBYTECODE=1`.

Exemplo (PowerShell, sessao atual):

```powershell
$env:PYTHONPYCACHEPREFIX = "C:\\temp\\pycache"
# ou
$env:PYTHONDONTWRITEBYTECODE = "1"
```

### 3) Pytest: manter cache sob controle

Em ambientes lentos, vale:

- Usar `-q` para reduzir output.
- Rodar testes mais especificos primeiro (um arquivo/um teste).
- Se houver instabilidade com cache, limpar `pytest` cache quando necessario:

```powershell
Remove-Item -Recurse -Force .pytest_cache -ErrorAction SilentlyContinue
```

### 4) MkDocs: builds incrementais

Durante escrita de docs, prefira builds incrementais (`--dirty`) quando possivel:

```powershell
python -m mkdocs build -f mkdocs.pt.yml --dirty
```

## Nota sobre reproducibilidade

Quando reportar bugs de performance/IO:

- informe se o repo esta no OneDrive,
- informe a versao do Python,
- e informe se esta usando `scipy` e/ou `numpy` (backends opcionais).

