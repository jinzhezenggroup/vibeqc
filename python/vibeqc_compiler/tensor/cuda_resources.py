"""Compatibility forwarding only; see docs/compiler_architecture.md.

Remove after downstream callers have migrated for one release and the legacy
import compatibility tests are the only repository users. No duplicate IR,
class definitions or cache implementation belongs here.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibeqc_compiler.common.cuda_resources import (
        KernelResources as KernelResources,
    )
    from vibeqc_compiler.common.cuda_resources import (
        parse_resources as parse_resources,
    )
else:
    import sys
    from importlib import import_module

    sys.modules[__name__] = import_module("vibeqc_compiler.common.cuda_resources")
