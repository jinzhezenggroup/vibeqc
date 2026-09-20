"""CUDA lowering composition for the bounded stationary RKS diagnostic.

Primitive recurrences, Becke local AD and AO bilinear AD remain their existing
compiler programs. Native code owns only allocation, traversal and reduction.
Generation is host-only and does not import the public runtime or probe CUDA.
"""

import os
import typing
from pathlib import Path

from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.native_runtime import compile_runtime
from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.common.source_cache import cache_source
from vibeqc_compiler.xc.geometry_cuda import emit_geometry_cuda


def emit_stationary_cuda(
    primitive_source: typing.Any, *, pbe: typing.Any, iterations: typing.Any = 3
) -> typing.Any:
    """Compose explicit primitive lowering and shared XC geometric lowering.

    The integral subsystem supplies a finite device dispatcher. Method lowering
    consumes its source, without reaching into recurrence or scheduling policy.
    """
    return (
        primitive_source
        + emit_geometry_cuda(pbe=pbe, iterations=iterations)
        + '#include "dft/stationary_gradient_cuda.cuh"\n'
    )


def compile_stationary_cuda(
    primitive_source: typing.Any,
    *,
    pbe: typing.Any,
    iterations: typing.Any,
    compiler: typing.Any,
    cache: typing.Any,
) -> typing.Any:
    """Compile a finite strict-FP64 artifact with transitive header identities."""
    if not isinstance(compiler, CudaCompilerAdapter):
        raise TypeError("stationary CUDA requires an explicit CUDA compiler adapter")
    if os.environ.get("NVCC_PREPEND_FLAGS") or os.environ.get("NVCC_APPEND_FLAGS"):
        raise ValueError("stationary strict CUDA rejects NVCC flag overrides")
    source = emit_stationary_cuda(primitive_source, pbe=pbe, iterations=iterations)
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (canonical_hash(source) + ".cu")
    cache_source(path, source)
    header = asset_path("src/dft/stationary_gradient_cuda.cuh")
    return compile_runtime(
        compiler,
        cache,
        path,
        headers=tuple(
            asset_path(name)
            for name in (
                "src/dft/stationary_gradient_cuda.cuh",
                "src/dft/grid_task_view.cuh",
                "src/dft/xc_point.hpp",
                "src/integrals/eri_geometry.hpp",
                "src/integrals/range_moments.hpp",
                "src/tensor/cuda_runtime.cuh",
                "src/runtime/bounded_workspace.hpp",
                "src/runtime/cuda_resources.cuh",
                "src/runtime/resource_cuda.cuh",
                "src/runtime/resource_ledger.hpp",
                "src/tensor/cuda_error.hpp",
                "src/tensor/metrics.hpp",
                "src/runtime/allocation_measurement.hpp",
            )
        ),
        libraries=("cublas",),
        options=("--fmad=false", "--expt-relaxed-constexpr", f"-I{header.parents[1]}"),
    )
