"""Compatibility forwarding only; see docs/compiler_architecture.md.

Remove after downstream callers have migrated for one release and the legacy
import compatibility tests are the only repository users. No duplicate IR,
class definitions or cache implementation belongs here.
"""

from importlib import import_module

_target = import_module("vibeqc_compiler.common.cuda_adapter")
import sys

if __name__ == "__main__":
    _target.main()
else:
    sys.modules[__name__] = _target
