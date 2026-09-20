"""CUDA lowering composition for the bounded stationary RKS diagnostic.

Primitive recurrences, Becke local AD and AO bilinear AD remain their existing
compiler programs; their bounded primitive/geometry contractions are emitted here.
Native code owns allocation, validation, transfers, launches and ABI only.
Generation is host-only and does not import the public runtime or probe CUDA.

Rationale: .agents/notes/implemented/architecture/2026-09-20-stationary-cuda-emitted-contractions.md
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

_STATIONARY_SCIENTIFIC_KERNELS = r"""namespace vibeqc_stationary_cuda {
__global__ void primitive_kernel(unsigned kind, const double* records, size_t count, double* output,
                                 int* error) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
    const double* r = records + record_stride * i;
    double v[12]{};
    for (size_t j = 0; j < record_stride; ++j)
      if (!isfinite(r[j])) {
        atomicExch(error, 1);
        return;
      }
    for (size_t j = 0; j < 4; ++j)
      if (!(r[j] > 0)) {
        atomicExch(error, 1);
        return;
      }
    if (!first_derivative(kind, r, r + 4, v)) {
      atomicExch(error, 1);
      return;
    }
    // Normalization and the plan's already generated source weight are
    // contracted on device, once per primitive; no host derivative tensor.
    double weight = r[24] * r[25];
    for (size_t j = 0; j < 4; ++j) weight *= r[16 + j] * r[20 + j];
    for (size_t j = 0; j < 12; ++j) output[12 * i + j] = finite(weight * v[j], error, 0);
  }
}
__global__ void primitive_reduce(const double* input, const int64_t* maps, size_t count, size_t na,
                                 double* output, int* error) {
  // A failed primitive leaves later output records unwritten. Never read
  // that storage after the producer has reported an error on this stream.
  if (*error) return;
  const size_t coord = blockIdx.x * blockDim.x + threadIdx.x;
  if (coord >= 3 * na) return;
  double sum = 0;
  for (size_t i = 0; i < count; ++i)
    for (size_t center = 0; center < 4; ++center) {
      const auto atom = maps[4 * i + center];
      if (atom < -1 || atom >= int64_t(na)) {
        atomicExch(error, 1);
        return;
      }
      if (atom == int64_t(coord / 3)) sum += input[12 * i + 3 * center + coord % 3];
    }
  output[coord] = finite(output[coord] + sum, error, 0);
}
__global__ void validate_centers(const double* centers, size_t na, double tolerance, int* error) {
  bool valid = true;
  for (size_t a = 0; a < na; ++a) {
    for (size_t k = 0; k < 3; ++k)
      if (!isfinite(centers[3 * a + k])) valid = false;
    for (size_t b = 0; b < a; ++b)
      if (vibeqc_grid_adjoint::distance(centers + 3 * a, centers + 3 * b, local_norm, valid)[0] <=
          tolerance)
        valid = false;
  }
  if (!valid) atomicExch(error, 1);
}
__global__ void geometry_kernel(vibeqc::dft::GridTaskView view, const double* work,
                                const int64_t* ao_atoms, const int64_t* owners,
                                const double* centers, size_t na, const double* weights,
                                const double* raw, double* partial, double* scratch, int* error) {
  const size_t lane = threadIdx.x;
  const size_t np = view.npoint, n = view.nactive, stride = np * n;
  double* grad = partial + lane * 9 * na;
  for (size_t k = 0; k < 9 * na; ++k) grad[k] = 0;
  double* ws = scratch + lane * 9 * na;
  auto* distances = reinterpret_cast<std::array<double, 4>*>(ws + 5 * na);
  auto* zeros = reinterpret_cast<size_t*>(ws + 4 * na);
  for (size_t p = lane; p < np; p += workers) {
    if (owners[p] < 0 || owners[p] >= int64_t(na) || !isfinite(weights[p]) || !isfinite(raw[p])) {
      atomicExch(error, 1);
      return;
    }
    double rho[2]{view.features[p], view.features[5 * np + p]}, g[2][3]{}, tau[2]{};
    if (stationary_functional != 0)
      for (size_t s = 0; s < 2; ++s)
        for (size_t k = 0; k < 3; ++k) g[s][k] = view.features[(5 * s + k + 1) * np + p];
    if (stationary_functional == 2)
      for (size_t s = 0; s < 2; ++s) tau[s] = view.features[(5 * s + 4) * np + p];
    // The exact shared SCF point model, including vacuum/spin boundaries.
    const auto xc = stationary_evaluate_point(rho, g, tau);
    if (!xc.valid) {
      atomicExch(error, 1);
      return;
    }
    for (size_t mu = 0; mu < n; ++mu) {
      if (view.ao_ids[mu] >= view.nao) {
        atomicExch(error, 1);
        return;
      }
      const auto atom = ao_atoms[view.ao_ids[mu]];
      if (atom < 0 || atom >= int64_t(na)) {
        atomicExch(error, 1);
        return;
      }
      double pullback[4]{};
      for (size_t s = 0; s < 2; ++s) {
        double c[5]{weights[p] * xc.rho[s]}, w[4]{};
        for (size_t j = 0; j < stationary_jets; ++j) {
          w[j] = work[(4 * s + j) * stride + p * n + mu];
          if (j) c[j] = weights[p] * xc.gradient[s][j - 1];
        }
        if (stationary_functional == 2) c[4] = weights[p] * xc.kinetic[s];
        double local[4]{};
        ao_pullback(c, w, local);
        for (size_t j = 0; j < stationary_jets; ++j) pullback[j] += local[j];
      }
      for (size_t k = 0; k < 3; ++k) {
        double value = 0;
        for (size_t j = 0; j < stationary_jets; ++j)
          value += pullback[j] * view.ao[stationary_shift[j][k] * stride + p * n + mu];
        grad[3 * atom + k] -= value;
        grad[3 * na + 3 * owners[p] + k] += value;
      }
    }
    if (!vibeqc_grid_adjoint::contract_point(view.points + 3 * p, centers, na, owners[p],
                                             xc.energy * raw[p], grad + 6 * na, ws, ws + na,
                                             ws + 2 * na, ws + 3 * na, zeros, distances, local_norm,
                                             local_ratio, local_log, local_becke)) {
      atomicExch(error, 1);
      return;
    }
  }
  for (size_t k = 0; k < 9 * na; ++k) finite(grad[k], error, 0);
}
__global__ void geometry_reduce(const double* partial, size_t na, double* output, int* error) {
  if (*error) return;
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 9 * na) return;
  double sum = 0;
  for (size_t lane = 0; lane < workers; ++lane) sum += partial[lane * 9 * na + i];
  output[i] = finite(output[i] + sum, error, 0);
}
}  // namespace vibeqc_stationary_cuda
"""


def emit_stationary_scientific_kernels() -> str:
    """Emit bounded primitive and XC geometry contractions for the runtime owner."""
    return _STATIONARY_SCIENTIFIC_KERNELS


def emit_stationary_cuda(
    primitive_source: typing.Any,
    *,
    functional: typing.Any = None,
    pbe: typing.Any = None,
    iterations: typing.Any = 3,
) -> typing.Any:
    """Compose explicit primitive lowering and shared XC geometric lowering.

    ``pbe`` remains a compatibility spelling for historical LDA/PBE callers.
    New method-owned lowering passes 0=LDA, 1=PBE, or 2=r2SCAN explicitly.
    """
    return (
        primitive_source
        + emit_geometry_cuda(functional=functional, pbe=pbe, iterations=iterations)
        + '#include "dft/stationary_gradient_cuda.cuh"\n'
        + emit_stationary_scientific_kernels()
    )


def compile_stationary_cuda(
    primitive_source: typing.Any,
    *,
    functional: typing.Any = None,
    pbe: typing.Any = None,
    iterations: typing.Any,
    compiler: typing.Any,
    cache: typing.Any,
) -> typing.Any:
    """Compile a finite strict-FP64 artifact with transitive header identities."""
    if not isinstance(compiler, CudaCompilerAdapter):
        raise TypeError("stationary CUDA requires an explicit CUDA compiler adapter")
    if os.environ.get("NVCC_PREPEND_FLAGS") or os.environ.get("NVCC_APPEND_FLAGS"):
        raise ValueError("stationary strict CUDA rejects NVCC flag overrides")
    source = emit_stationary_cuda(
        primitive_source, functional=functional, pbe=pbe, iterations=iterations
    )
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
