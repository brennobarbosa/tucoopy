"""
# Lightweight visualization helpers (optional).

This subpackage is intentionally dependency-light and only requires Matplotlib
when actually used (install with: `pip install "tucoop[viz]"`).

Scope
-----
- Only 2-player (segment) and 3-player (ternary) static plots are supported.
- The plotting helpers are designed to work either from a `Game` object or from
  the JSON contract used by the JS demo (when applicable).

Warnings
--------
- This subpackage depends on Matplotlib at runtime.
  Install it with `pip install "tucoop[viz]"`.
- If Matplotlib is not installed, calling any plotting function will raise a
  `MissingOptionalDependencyError`.
"""

from .mpl2 import plot_segment, plot_spec_segment
from .mpl3 import plot_spec_ternary, plot_ternary

__all__ = ["plot_segment", "plot_ternary", "plot_spec_segment", "plot_spec_ternary"]
