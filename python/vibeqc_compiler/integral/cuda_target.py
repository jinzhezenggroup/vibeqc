"""Compatibility forwarding only; see docs/compiler_architecture.md.

Remove after downstream callers have migrated for one release and the legacy
import compatibility tests are the only repository users. No duplicate IR,
class definitions or cache implementation belongs here.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibeqc_compiler.common.cuda_target import (
        CUDA_TARGETS as CUDA_TARGETS,
    )
    from vibeqc_compiler.common.cuda_target import (
        DEFAULT_CUDA_TARGET as DEFAULT_CUDA_TARGET,
    )
    from vibeqc_compiler.common.cuda_target import (
        CudaArchitecture as CudaArchitecture,
    )
    from vibeqc_compiler.common.cuda_target import (
        CudaTargetInfo as CudaTargetInfo,
    )
    from vibeqc_compiler.common.cuda_target import (
        cuda_architecture as cuda_architecture,
    )
    from vibeqc_compiler.common.cuda_target import (
        cuda_target_info as cuda_target_info,
    )
    from vibeqc_compiler.common.cuda_target import (
        normalize_cuda_architecture as normalize_cuda_architecture,
    )
    from vibeqc_compiler.common.cuda_target import (
        normalize_cuda_compile_architecture as normalize_cuda_compile_architecture,
    )
else:
    import sys
    from importlib import import_module

    sys.modules[__name__] = import_module("vibeqc_compiler.common.cuda_target")
