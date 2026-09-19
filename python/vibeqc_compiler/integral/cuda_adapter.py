"""Compatibility forwarding only; see docs/compiler_architecture.md.

Remove after downstream callers have migrated for one release and the legacy
import compatibility tests are the only repository users. No duplicate IR,
class definitions or cache implementation belongs here.
"""

from typing import TYPE_CHECKING

__all__ = (
    "CudaBenchmarkExecutor",
    "CudaCompileResult",
    "CudaCompilerAdapter",
    "CudaExecutionProfile",
    "resolve_cuda_execution_profile",
)

if TYPE_CHECKING:
    from vibeqc_compiler.common.cuda_adapter import (
        CudaBenchmarkExecutor,
        CudaCompilerAdapter,
        CudaCompileResult,
        CudaExecutionProfile,
        resolve_cuda_execution_profile,
    )
else:
    import sys
    from importlib import import_module

    sys.modules[__name__] = import_module("vibeqc_compiler.common.cuda_adapter")
