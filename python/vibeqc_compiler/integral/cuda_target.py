"""Compatibility forwarding only; see docs/compiler_architecture.md.

Remove after downstream callers have migrated for one release and the legacy
import compatibility tests are the only repository users. No duplicate IR,
class definitions or cache implementation belongs here.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibeqc_compiler.common import cuda_target as _cuda_target

    CUDA_TARGETS = _cuda_target.CUDA_TARGETS
    DEFAULT_CUDA_TARGET = _cuda_target.DEFAULT_CUDA_TARGET
    CudaArchitecture = _cuda_target.CudaArchitecture
    CudaTargetInfo = _cuda_target.CudaTargetInfo
    cuda_architecture = _cuda_target.cuda_architecture
    cuda_target_info = _cuda_target.cuda_target_info
    normalize_cuda_architecture = _cuda_target.normalize_cuda_architecture
    normalize_cuda_compile_architecture = (
        _cuda_target.normalize_cuda_compile_architecture
    )
else:
    import sys
    from importlib import import_module

    sys.modules[__name__] = import_module("vibeqc_compiler.common.cuda_target")
