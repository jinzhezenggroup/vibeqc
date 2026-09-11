"""Compatibility forwarding only; see docs/compiler_architecture.md.

Remove after downstream callers have migrated for one release and the legacy
import compatibility tests are the only repository users. No duplicate IR,
class definitions or cache implementation belongs here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from importlib import import_module

_target = import_module("vibeqc_compiler.integral.expr")

if __name__ == "__main__":
    if hasattr(_target, "main"):
        raise SystemExit(_target.main())
else:
    sys.modules[__name__] = _target
