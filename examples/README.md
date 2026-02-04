# Examples

These scripts are small, runnable examples for the `tucoop` Python package.

Run them from `packages/tucoop-py/`:

```bash
python examples/01_shapley_and_core.py
python examples/02_classic_games.py
python examples/03_least_core_and_nucleolus.py   # requires: pip install "tucoop[lp]"
python examples/04_kernel_and_prekernel.py       # requires: pip install "tucoop[fast]"
python examples/05_animation_spec_to_file.py
python examples/06_generate_specs_for_js_demo.py
python examples/07_weighted_values.py
python examples/08_power_indices_weighted_voting.py
python examples/09_modiclus.py                   # requires: pip install "tucoop[lp]"
python examples/10_kernel_set_and_bargaining_set.py
python examples/11_static_viz_from_spec.py       # requires: pip install "tucoop[viz]"
python examples/12_static_viz_direct.py          # requires: pip install "tucoop[viz]"
python examples/13_mpl2_segment.py               # requires: pip install "tucoop[viz]"
python examples/14_mpl3_ternary.py               # requires: pip install "tucoop[viz]"
python examples/15_transforms_mobius_and_harsanyi.py
python examples/16_io_roundtrip_animation_spec.py
python examples/17_diagnostics_core_checks.py
python examples/18_blocking_regions_ternary.py
python examples/19_power_indices_more.py
python examples/20_geometry_sets_with_lp_backend.py  # requires: pip install "tucoop[lp]"
```

Notes:
- Examples add `packages/tucoop-py/src` to `sys.path` so you can run them without installing.
- Optional examples exit with a friendly message if the extra dependency is missing.

Flags
-----
Some examples accept `--out DIR` (and in a few cases `--spec-dir DIR`) to
control where output files are written/read. Paths can be absolute, or relative
to the example file.
