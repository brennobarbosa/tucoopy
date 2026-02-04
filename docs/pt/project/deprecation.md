# Politica de deprecacao

Este documento define como o `tucoopy` (Python) lida com mudancas de API.

## Objetivo

- Evitar "drift" e refactors repetidos.
- Dar previsibilidade para usuarios e para a documentacao.

## Versoes `0.x`

O projeto esta em fase Alpha (`0.x`):

- Mudancas quebrando API ainda podem acontecer.
- Mesmo assim, tentamos **depreciar antes de remover** quando isso nao custar caro.

## Processo recomendado (quando possivel)

1. **Marcar como deprecated**
   - Documentar no changelog.
   - Atualizar docs (pagina de referencia, exemplos).
2. **Emitir aviso**
   - Usar `DeprecationWarning` (ou excecao especifica) apenas quando fizer sentido.
3. **Remover**
   - Preferencia: remover no proximo "minor" (ex.: `0.2.0`), ou no maximo em `+2` releases.

## Arquivos "fantasma" e duplicados

Arquivos vazios (ou com `# apagar`) nao devem permanecer:

- Preferir **remover de fato**.
- Se nao for possivel remover (restricoes de ambiente), manter um modulo que:
  - falhe no import com erro explicito, e
  - explique o replacement (novo caminho/camada source-of-truth).

## Compatibilidade com JS

Compatibilidade deve ser mantida no **contrato JSON** (`tucoopy.io.schema`), nao por shims/aliases de modulo.

