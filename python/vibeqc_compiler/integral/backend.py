"""Compatibility forwarding only; see docs/compiler_architecture.md.

Remove after downstream callers have migrated for one release and the legacy
import compatibility tests are the only repository users. No duplicate IR,
class definitions or cache implementation belongs here.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibeqc_compiler.common import backend as _backend

    BenchmarkExecutor = _backend.BenchmarkExecutor
    CompilerAdapter = _backend.CompilerAdapter
    DeviceProbe = _backend.DeviceProbe
    RegistryEmitter = _backend.RegistryEmitter
    ResourceParser = _backend.ResourceParser
    SourceEmitter = _backend.SourceEmitter
    TargetInfo = _backend.TargetInfo
    TargetScheduleShape = _backend.TargetScheduleShape
else:
    import sys
    from importlib import import_module

    sys.modules[__name__] = import_module("vibeqc_compiler.common.backend")
