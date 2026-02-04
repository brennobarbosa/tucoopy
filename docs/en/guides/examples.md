# Exemplos

Os exemplos executaveis ficam em `packages/tucoopy-py/examples/`.

A partir de `packages/tucoopy-py/`.

## Basicos (sem deps opcionais)

```bash
python examples/01_shapley_and_core.py
python examples/02_classic_games.py
python examples/07_weighted_values.py
python examples/08_power_indices_weighted_voting.py
python examples/15_transforms_mobius_and_harsanyi.py
python examples/16_io_roundtrip_animation_spec.py
python examples/17_diagnostics_core_checks.py
python examples/18_blocking_regions_ternary.py
python examples/19_power_indices_more.py
```

## IO / contrato JSON

```bash
python examples/05_animation_spec_to_file.py
python examples/06_generate_specs_for_js_demo.py
python examples/16_io_roundtrip_animation_spec.py
```

## Exemplos com dependencias opcionais

!!! warning
    Alguns exemplos exigem extras em runtime.

    - LP (SciPy): `pip install "tucoopy[lp]"`
    - Fast (NumPy): `pip install "tucoopy[fast]"`
    - Viz (Matplotlib): `pip install "tucoopy[viz]"`

```bash
python examples/03_least_core_and_nucleolus.py        # requer: tucoopy[lp]
python examples/04_kernel_and_prekernel.py            # requer: tucoopy[fast]
python examples/09_modiclus.py                        # requer: tucoopy[lp]
python examples/10_kernel_set_and_bargaining_set.py   # parte requer: tucoopy[lp]
python examples/11_static_viz_from_spec.py            # requer: tucoopy[viz]
python examples/12_static_viz_direct.py               # requer: tucoopy[viz]
python examples/13_mpl2_segment.py                    # requer: tucoopy[viz]
python examples/14_mpl3_ternary.py                    # requer: tucoopy[viz]
python examples/20_geometry_sets_with_lp_backend.py   # requer: tucoopy[lp]
```


## Flags

Alguns exemplos aceitam --out DIR (e em alguns casos --spec-dir DIR) para controlar onde os arquivos sao escritos/lidos.

