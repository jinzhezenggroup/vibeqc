"""Compatibility entry point for compiler-owned Libxc lowering."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from vibeqc_compiler.xc import libxc_c_metadata as _implementation

sys.modules[__name__] = _implementation
