"""Compatibility forwarding only; see docs/compiler_architecture.md.

Remove after downstream callers have migrated for one release and the legacy
import compatibility tests are the only repository users. No duplicate IR,
class definitions or cache implementation belongs here.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibeqc_compiler.common.backend import (
        BenchmarkExecutor as BenchmarkExecutor,
    )
    from vibeqc_compiler.common.backend import (
        CompilerAdapter as CompilerAdapter,
    )
    from vibeqc_compiler.common.backend import (
        DeviceProbe as DeviceProbe,
    )
    from vibeqc_compiler.common.backend import (
        RegistryEmitter as RegistryEmitter,
    )
    from vibeqc_compiler.common.backend import (
        ResourceParser as ResourceParser,
    )
    from vibeqc_compiler.common.backend import (
        SourceEmitter as SourceEmitter,
    )
    from vibeqc_compiler.common.backend import (
        TargetInfo as TargetInfo,
    )
    from vibeqc_compiler.common.backend import (
        TargetScheduleShape as TargetScheduleShape,
    )
else:
    import sys
    from importlib import import_module

    sys.modules[__name__] = import_module("vibeqc_compiler.common.backend")
