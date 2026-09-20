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

void enqueue(const CudaXcLayout& l, cudaStream_t stream, const double* basis, const double* points,
             const double* weights, const double* density, double* ao, double* work,
             double* features, double* coefficients, double* point_totals, double* potential,
             double* totals, int* error, const double* direction, double* delta_features) {
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
    // The existing through-f AO kernel is compiled earlier in this same TU.
    ao_kernel<<<blocks(l.jets * count * l.nao, 128), 128, 0, stream>>>(
        basis, l.natom, l.nprimitive, l.nao, points + 3 * begin, count, l.jets, ao, error, nullptr);
    cuda_check(cudaGetLastError());
    density_product<<<blocks(l.spins * l.work_jets * count * l.nao, 128), 128, 0, stream>>>(
        density, ao, l.nao, count, l.spins, l.work_jets, work, error);
    cuda_check(cudaGetLastError());
    density_features<<<blocks(l.spins * count, 128), 128, 0, stream>>>(
        ao, work, l.nao, count, l.spins, l.jets, l.work_jets, l.feature_terms, l.functional,
        features, error);
    cuda_check(cudaGetLastError());
    if (direction) {
      // AO panels are shared; work is scratch and can be reused after the
      // reference features are retained. No host AO/feature staging occurs.
      density_product<<<blocks(l.spins * count * l.nao, 128), 128, 0, stream>>>(
          direction, ao, l.nao, count, l.spins, l.work_jets, work, error);
      cuda_check(cudaGetLastError());
      density_features<<<blocks(l.spins * count, 128), 128, 0, stream>>>(
          ao, work, l.nao, count, l.spins, l.jets, l.work_jets, l.feature_terms, l.functional,
          delta_features, error);
      cuda_check(cudaGetLastError());
    }
    evaluate_points<<<blocks(count, 128), 128, 0, stream>>>(
        features, weights + begin, count, l.spins, l.feature_terms, l.functional, coefficients,
        point_totals, error, delta_features);
    cuda_check(cudaGetLastError());
    assemble_potential<<<blocks(matrices, 128), 128, 0, stream>>>(
        ao, coefficients, weights + begin, l.nao, count, l.spins, l.feature_terms, potential,
        error);
    cuda_check(cudaGetLastError());
    accumulate_totals<<<1, 32, 0, stream>>>(point_totals, count, totals, error);
    cuda_check(cudaGetLastError());
  }
}
}  // namespace vibeqc::dft::cuda_xc_detail
