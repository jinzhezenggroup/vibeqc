"""Compatibility forwarding only; see docs/compiler_architecture.md.

Remove after downstream callers have migrated for one release and the legacy
import compatibility tests are the only repository users. No duplicate IR,
class definitions or cache implementation belongs here.
"""

from typing import TYPE_CHECKING

__all__ = (
    "BenchmarkExecutor",
    "CompilerAdapter",
    "DeviceProbe",
    "RegistryEmitter",
    "ResourceParser",
    "SourceEmitter",
    "TargetInfo",
    "TargetScheduleShape",
)

if TYPE_CHECKING:
    from vibeqc_compiler.common.backend import (
        BenchmarkExecutor,
        CompilerAdapter,
        DeviceProbe,
        RegistryEmitter,
        ResourceParser,
        SourceEmitter,
        TargetInfo,
        TargetScheduleShape,
    )
else:
    import sys
    from importlib import import_module

    sys.modules[__name__] = import_module("vibeqc_compiler.common.backend")
