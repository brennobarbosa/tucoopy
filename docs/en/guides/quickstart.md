# Começando

## Instalação

Instalação básica (sem dependências pesadas opcionais):

```bash
pip install tucoopy
```

Extras opcionais:

- Métodos baseados em LP (least-core / nucleolus / balancedness / conjunto de barganha):
  ```bash
  pip install "tucoopy[lp]"
  ```
- Backend alternativo de LP (PuLP):
  ```bash
  pip install "tucoopy[lp_alt]"
  ```
- Acelerações baseadas em NumPy (kernel / prekernel e alguns utilitários):
  ```bash
  pip install "tucoopy[fast]"
  ```
- Visualizações simples para 2 ou 3 jogadores em Matplotlib:
  ```bash
  pip install "tucoopy[viz]"
  ```

## Construir um jogo TU

Coalizões são armazenadas internamente como máscaras de bits, mas você pode definir jogos com chaves de coalizão “pythonicas”:

```py
from tucoopy import Game

g = Game.from_coalitions(
    n_players=3,
    values={
        (): 0.0,
        (0,): 1.0,
        (1,): 1.2,
        (2,): 0.8,
        (0, 1): 2.8,
        (0, 2): 2.2,
        (1, 2): 2.0,
        (0, 1, 2): 4.0,
    },
)
```

## Calcular uma solução

```py
from tucoopy.solutions import shapley_value

phi = shapley_value(g)
print(phi)
```

## Gerar uma especificação de animação (contrato Python -> JS)

```py
from tucoopy.io.animation_spec import build_animation_spec

spec = build_animation_spec(
    g,
    series_id="shapley",
    allocations=[phi] * 60,
    dt=1 / 30,
    series_description="Valor de Shapley (estático).",
    include_analysis=True,
)
print(spec.to_json())
```

Mais scripts executáveis ficam em `packages/tucoopy-py/examples/`.
