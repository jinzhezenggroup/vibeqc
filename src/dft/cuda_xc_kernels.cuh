#pragma once

// Included only by the native generated_grid_policy.cu, after cuda_grid.cu.
// Reuse compiler-emitted AO traversal, XC point algebra and dense contractions;
// this header keeps only validation and launch/runtime glue.
#include "dft/cuda_xc.hpp"

namespace vibeqc::dft::cuda_xc_detail {
namespace {
using vibeqc_tensor::blocks;
using vibeqc_tensor::cuda_check;
using vibeqc_tensor::I;

__global__ void validate_density(const double* density, I n, I spins, int* error) {
  for (I i = I(blockIdx.x) * blockDim.x + threadIdx.x; i < spins * n * n;
       i += I(blockDim.x) * gridDim.x) {
    const I transpose = (i / (n * n) * n + i % n) * n + i / n % n;
    const double a = density[i], b = density[transpose];
    if (!isfinite(a) || fabs(a - b) > 1e-12 + 1e-10 * fmax(fabs(a), fabs(b)))
      atomicCAS(error, 0, 5);
  }
}

}  // namespace

void enqueue(const CudaXcLayout& l, CudaXcPointLauncher point_launcher, cudaStream_t stream,
             const double* basis, const double* points, const double* weights,
             const double* density, double* ao, double* work, double* features,
             double* coefficients, double* point_totals, double* potential, double* totals,
             int* error, CudaXcDensityPrecision precision, const double* direction,
             double* delta_features) {
  const I matrices = l.spins * l.nao * l.nao;
  cuda_check(cudaMemsetAsync(error, 0, sizeof(int), stream));
  cuda_check(cudaMemsetAsync(totals, 0, 3 * sizeof(double), stream));
  cuda_check(cudaMemsetAsync(potential, 0, matrices * sizeof(double), stream));
  validate_density<<<blocks(matrices, 128), 128, 0, stream>>>(density, l.nao, l.spins, error);
  cuda_check(cudaGetLastError());
  if (direction) {
    validate_density<<<blocks(matrices, 128), 128, 0, stream>>>(direction, l.nao, l.spins, error);
    cuda_check(cudaGetLastError());
  }
  for (std::size_t begin = 0; begin < l.npoint; begin += l.tile_points) {
    const I count = std::min(l.tile_points, l.npoint - begin);
    // Route B changes only AO arithmetic. The AO panel and all downstream
    // density/XC reductions stay FP64 so this is a clean precision ablation.
    if (l.ao_precision == CudaXcAoPrecision::Fp32ComputeFp64Storage)
      ao_kernel_fp32<<<blocks(l.jets * count * l.nao, 128), 128, 0, stream>>>(
          basis, l.natom, l.nprimitive, l.nao, points + 3 * begin, count, l.jets, ao, error,
          nullptr);
    else
      ao_kernel<<<blocks(l.jets * count * l.nao, 128), 128, 0, stream>>>(
          basis, l.natom, l.nprimitive, l.nao, points + 3 * begin, count, l.jets, ao, error,
          nullptr);
    cuda_check(cudaGetLastError());
    scheduled_density_product(
        stream, density, ao, l.nao, count, l.spins, l.work_jets,
        precision == CudaXcDensityPrecision::Fp32ComputeFp64Accumulate, work, error);
    cuda_check(cudaGetLastError());
    scheduled_density_features(stream, ao, work, l.nao, count, l.spins, l.jets, l.work_jets,
                               l.feature_terms, l.functional, features, error);
    cuda_check(cudaGetLastError());
    if (direction) {
      // AO panels are shared; work is scratch and can be reused after the
      // reference features are retained. No host AO/feature staging occurs.
      scheduled_density_product(
          stream, direction, ao, l.nao, count, l.spins, l.work_jets,
          precision == CudaXcDensityPrecision::Fp32ComputeFp64Accumulate, work, error);
      cuda_check(cudaGetLastError());
      scheduled_density_features(stream, ao, work, l.nao, count, l.spins, l.jets, l.work_jets,
                                 l.feature_terms, l.functional, delta_features, error);
      cuda_check(cudaGetLastError());
    }
    point_launcher(stream, features, weights + begin, count, l.spins, coefficients, point_totals,
                   error, delta_features);
    cuda_check(cudaGetLastError());
    // Feature/response consumers have finished reading work. The compiler may
    // reuse those same panels for weighted symmetric potential assembly.
    scheduled_potential(stream, ao, coefficients, weights + begin, l.nao, count, l.spins,
                        l.feature_terms, l.work_jets, work, potential, error);
    cuda_check(cudaGetLastError());
    accumulate_totals<<<1, 32, 0, stream>>>(point_totals, count, totals, error);
    cuda_check(cudaGetLastError());
  }
}
}  // namespace vibeqc::dft::cuda_xc_detail
