# Examples

Runnable examples live in `examples/`.

From the repo root:

## Basics (no optional deps)

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

## IO / JSON contract

```bash
python examples/05_animation_spec_to_file.py
python examples/06_generate_specs_for_js_demo.py
python examples/16_io_roundtrip_animation_spec.py
```

## Examples with optional dependencies

!!! warning
    Some examples require extras at runtime.

    - LP (SciPy): `pip install "tucoopy[lp]"`
    - Fast (NumPy): `pip install "tucoopy[fast]"`
    - Viz (Matplotlib): `pip install "tucoopy[viz]"`

```bash
python examples/03_least_core_and_nucleolus.py        # requires: tucoopy[lp]
python examples/04_kernel_and_prekernel.py            # requires: tucoopy[fast]
python examples/09_modiclus.py                        # requires: tucoopy[lp]
python examples/10_kernel_set_and_bargaining_set.py   # partly requires: tucoopy[lp]
python examples/11_static_viz_from_spec.py            # requires: tucoopy[viz]
python examples/12_static_viz_direct.py               # requires: tucoopy[viz]
python examples/13_mpl2_segment.py                    # requires: tucoopy[viz]
python examples/14_mpl3_ternary.py                    # requires: tucoopy[viz]
python examples/20_geometry_sets_with_lp_backend.py   # requires: tucoopy[lp]
```

## Flags

Some examples accept `--out DIR` (and in some cases `--spec-dir DIR`) to control where files are written/read.
