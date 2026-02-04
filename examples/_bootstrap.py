from __future__ import annotations

import sys
from pathlib import Path


def add_src_to_path() -> None:
    """
    Allow running examples without installing the package.
    """
    pkg_root = Path(__file__).resolve().parents[1]
    src = str(pkg_root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
