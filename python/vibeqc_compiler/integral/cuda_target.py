"""Compatibility forwarding only; see docs/compiler_architecture.md.

Remove after downstream callers have migrated for one release and the legacy
import compatibility tests are the only repository users. No duplicate IR,
class definitions or cache implementation belongs here.
"""

from typing import TYPE_CHECKING

__all__ = (
    "CUDA_TARGETS",
    "DEFAULT_CUDA_TARGET",
    "CudaArchitecture",
    "CudaTargetInfo",
    "cuda_architecture",
    "cuda_target_info",
    "normalize_cuda_architecture",
    "normalize_cuda_compile_architecture",
)

if TYPE_CHECKING:
    from vibeqc_compiler.common.cuda_target import (
        CUDA_TARGETS,
        DEFAULT_CUDA_TARGET,
        CudaArchitecture,
        CudaTargetInfo,
        cuda_architecture,
        cuda_target_info,
        normalize_cuda_architecture,
        normalize_cuda_compile_architecture,
    )
else:
    import sys
    from importlib import import_module

    sys.modules[__name__] = import_module("vibeqc_compiler.common.cuda_target")
