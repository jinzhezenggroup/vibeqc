"""Compatibility forwarding only; see docs/compiler_architecture.md.

Remove after downstream callers have migrated for one release and the legacy
import compatibility tests are the only repository users. No duplicate IR,
class definitions or cache implementation belongs here.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibeqc_compiler.common.cuda_adapter import (
        CudaBenchmarkExecutor as CudaBenchmarkExecutor,
    )
    from vibeqc_compiler.common.cuda_adapter import (
        CudaCompilerAdapter as CudaCompilerAdapter,
    )
    from vibeqc_compiler.common.cuda_adapter import (
        CudaCompileResult as CudaCompileResult,
    )
    from vibeqc_compiler.common.cuda_adapter import (
        CudaExecutionProfile as CudaExecutionProfile,
    )
    from vibeqc_compiler.common.cuda_adapter import (
        resolve_cuda_execution_profile as resolve_cuda_execution_profile,
    )
else:
    import sys
    from importlib import import_module

    sys.modules[__name__] = import_module("vibeqc_compiler.common.cuda_adapter")
