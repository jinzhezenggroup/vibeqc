"""Compatibility forwarding only; see docs/compiler_architecture.md.

Remove after downstream callers have migrated for one release and the legacy
import compatibility tests are the only repository users. No duplicate IR,
class definitions or cache implementation belongs here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from importlib import import_module

_target = import_module("vibeqc_compiler.integral")
__all__ = list(
    getattr(_target, "__all__", [n for n in vars(_target) if not n.startswith("_")])
)


def __getattr__(name):
    return getattr(_target, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_target)))
