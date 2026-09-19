"""Compatibility forwarding only; see docs/compiler_architecture.md.

Remove after downstream callers have migrated for one release and the legacy
import compatibility tests are the only repository users. No duplicate IR,
class definitions or cache implementation belongs here.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibeqc_compiler.common import cuda_adapter as _cuda_adapter

    CudaBenchmarkExecutor = _cuda_adapter.CudaBenchmarkExecutor
    CudaCompilerAdapter = _cuda_adapter.CudaCompilerAdapter
    CudaCompileResult = _cuda_adapter.CudaCompileResult
    CudaExecutionProfile = _cuda_adapter.CudaExecutionProfile
    resolve_cuda_execution_profile = _cuda_adapter.resolve_cuda_execution_profile
else:
    import sys
    from importlib import import_module

    sys.modules[__name__] = import_module("vibeqc_compiler.common.cuda_adapter")
