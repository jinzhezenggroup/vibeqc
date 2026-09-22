"""CUDA setup lowering for shared symmetric overlap and occupied densities.

Both reconstructions instantiate the existing SCF weighted-projector TensorIR:
inverse-square-root spectral weights for X and occupation weights for D. The
matrix-function compiler remains the inverse-square-root scalar owner. Runtime
code supplies eigenframes, borrowed storage and execution ordering.
"""

from vibeqc_compiler.tensor.scf import density_program


def emit_mean_field_setup_cuda() -> str:
    """Lower the admitted weighted-projector topology to dynamic column storage."""
    program = density_program(1, 2)
    root = program.outputs["density"]
    if (
        root.op != "einsum"
        or root.attrs["labels"] != ((0, 1, 2, 3), (0, 1, 3), (0, 1, 4, 3))
        or root.attrs["output"] != (0, 1, 2, 4)
        or root.attrs["coefficient"] != (1, 1)
        or root.inputs[0] is not root.inputs[2]
    ):
        raise ValueError(
            "CUDA setup requires the shared SCF weighted-projector topology"
        )
    return (
        "// SCF weighted-projector prototype: "
        + program.logical_hash
        + "\n"
        + r"""
#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include "generated_symmetric_matrix_function.cuh"

namespace vibeqc::scf::generated {
// Overlap uses the historical absolute cutoff: equality at 1e-10 is allowed.
// The full symmetric representation is retained; no rank truncation occurs.
__global__ void overlap_spectral_weights(size_t n, double* values, int* invalid) {
  for (size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x; i < n;
       i += size_t(blockDim.x)*gridDim.x) {
    const double value = values[i];
    if (!isfinite(value) || value < 1e-10) {
      atomicExch(invalid, 1);
      values[i] = 0.0;
    } else {
      values[i] = vibeqc::tensor::detail::matrix_function_value(value, true, 0);
    }
  }
}

// One output cell reduces the TensorIR orbital index in canonical order.
// Column-major eigenframes match cuSOLVER; symmetric outputs also satisfy the
// public row-major density convention without a transposition/staging pass.
__global__ void weighted_projector(size_t n, unsigned spins, const double* vectors,
                                    const double* weights, double* output) {
  for (size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x; i < spins*n*n;
       i += size_t(blockDim.x)*gridDim.x) {
    const size_t spin = i/(n*n), row = i%n, column = (i/n)%n;
    const double* c = vectors + spin*n*n;
    const double* w = weights + spin*n;
    double value = 0.0;
    for (size_t k = 0; k < n; ++k)
      value += w[k] * c[row+k*n] * c[column+k*n];
    output[i] = value;
  }
}

__global__ void occupation_weights(size_t n, unsigned spins, int alpha, int beta,
                                    double* weights) {
  for (size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x; i < spins*n;
       i += size_t(blockDim.x)*gridDim.x) {
    const int count = i/n == 0 ? alpha : beta;
    weights[i] = i%n < size_t(count) ? (spins == 1 ? 2.0 : 1.0) : 0.0;
  }
}

__global__ void check_overlap_identity(size_t n, const double* metric, int* invalid) {
  for (size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x; i < n*n;
       i += size_t(blockDim.x)*gridDim.x) {
    const double value = metric[i];
    if (!isfinite(value) || fabs(value - (i%n == i/n ? 1.0 : 0.0)) > 1e-8)
      atomicExch(invalid, 1);
  }
}
} // namespace vibeqc::scf::generated
"""
    )
