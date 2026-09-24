#include <math_constants.h>

#include <cmath>

#include "generated_scf_density_cuda.cuh"
#include "scf/cuda/matrix_index.cuh"
#include "scf/cuda/scf_constants.hpp"
#include "scf/cuda/scf_density_kernels.hpp"

namespace vibeqc::scf::cuda_execution {

__global__ void mix_open_shell_guess_kernel(std::int32_t batch_size, std::int32_t nbf,
                                            const std::int32_t* occupied,
                                            const std::uint8_t* active, double* coefficients) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch_size) * n) return;
  const std::size_t system = element / n;
  if (active != nullptr && active[system] == 0) return;
  const std::size_t row = element % n;
  const std::int32_t alpha_occupied = occupied[system * 2];
  const std::int32_t beta_occupied = occupied[system * 2 + 1];
  if (alpha_occupied == beta_occupied || beta_occupied <= 0 || beta_occupied >= nbf) {
    return;
  }

  // Match the CPU open-shell cold guess: preserve the beta orbital metric
  // while breaking exact spatial symmetry between its frontier orbitals.
  constexpr double cosine = 0.7071067811865476;
  constexpr double sine = 0.7071067811865476;
  const std::size_t matrix_size = n * n;
  const std::size_t offset = (system * 2 + 1) * matrix_size;
  const std::size_t occupied_orbital = static_cast<std::size_t>(beta_occupied - 1);
  const std::size_t virtual_orbital = static_cast<std::size_t>(beta_occupied);
  const double occupied_value = coefficients[offset + matrix_index(row, occupied_orbital, n)];
  const double virtual_value = coefficients[offset + matrix_index(row, virtual_orbital, n)];
  coefficients[offset + matrix_index(row, occupied_orbital, n)] =
      cosine * occupied_value + sine * virtual_value;
  coefficients[offset + matrix_index(row, virtual_orbital, n)] =
      -sine * occupied_value + cosine * virtual_value;
}

template <unsigned BlockThreads>
__device__ double warm_density_block_sum(double value, double* warp_sums) {
  static_assert(BlockThreads % 32 == 0);
  constexpr unsigned kWarpWidth = 32;
  const unsigned lane = threadIdx.x % kWarpWidth;
  const unsigned warp = threadIdx.x / kWarpWidth;
  for (unsigned delta = kWarpWidth / 2; delta != 0; delta >>= 1) {
    value += __shfl_down_sync(0xffffffffU, value, delta);
  }
  if (lane == 0) warp_sums[warp] = value;
  __syncthreads();

  // The first warp reduces the block's partial sums. All lanes participate in
  // the shuffle so the full mask remains valid; unused lanes contribute 0.
  value = warp == 0 && lane < BlockThreads / kWarpWidth ? warp_sums[lane] : 0.0;
  if (warp == 0) {
    for (unsigned delta = kWarpWidth / 2; delta != 0; delta >>= 1) {
      value += __shfl_down_sync(0xffffffffU, value, delta);
    }
  }
  if (threadIdx.x == 0) warp_sums[0] = value;
  __syncthreads();
  return warp_sums[0];
}

__global__ void apply_warm_density_kernel(std::int32_t batch_size, std::int32_t nbf,
                                          const std::int32_t* occupied,
                                          const std::uint8_t* warm_mask, const double* warm_density,
                                          const double* overlap, double* density,
                                          std::uint8_t* warm_invalid) {
  const std::int32_t system = static_cast<std::int32_t>(blockIdx.x);
  if (system >= batch_size || warm_mask[system] == 0) return;
  __shared__ double warp_sums[kWarmDensityThreads / 32];
  __shared__ double scale;
  __shared__ int valid_trace;
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t offset = static_cast<std::size_t>(system) * matrix_size;
  double trace = 0.0;
  for (std::size_t element = threadIdx.x; element < matrix_size; element += blockDim.x) {
    const std::size_t row = element % n;
    const std::size_t column = element / n;
    const std::size_t transpose = matrix_index(column, row, n);
    const double symmetric =
        0.5 * (warm_density[offset + element] + warm_density[offset + transpose]);
    density[offset + element] = symmetric;
    trace += symmetric * overlap[offset + transpose];
  }
  trace = warm_density_block_sum<kWarmDensityThreads>(trace, warp_sums);
  if (threadIdx.x == 0) {
    valid_trace = isfinite(trace) && trace > 0.0;
    warm_invalid[system] = valid_trace ? 0 : 1;
    scale = valid_trace ? 2.0 * occupied[system] / trace : 0.0;
  }
  __syncthreads();
  if (valid_trace != 0) {
    for (std::size_t element = threadIdx.x; element < matrix_size; element += blockDim.x) {
      density[offset + element] *= scale;
    }
  }
}

__global__ void apply_uhf_warm_density_kernel(std::int32_t batch_size, std::int32_t nbf,
                                              const std::int32_t* occupied,
                                              const std::uint8_t* warm_mask,
                                              const double* warm_density, const double* overlap,
                                              double* density, std::uint8_t* warm_invalid) {
  const std::int32_t system = static_cast<std::int32_t>(blockIdx.x);
  if (system >= batch_size || warm_mask[system] == 0) return;
  __shared__ double warp_sums[kWarmDensityThreads / 32];
  __shared__ double scale;
  __shared__ int valid_trace;
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t overlap_offset = static_cast<std::size_t>(system) * matrix_size;
  if (threadIdx.x == 0) warm_invalid[system] = 0;
  __syncthreads();
  for (std::int32_t spin = 0; spin < 2; ++spin) {
    const std::size_t state = static_cast<std::size_t>(system) * 2 + spin;
    const std::size_t offset = state * matrix_size;
    const double target = static_cast<double>(occupied[state]);
    if (target == 0.0) {
      for (std::size_t element = threadIdx.x; element < matrix_size; element += blockDim.x) {
        density[offset + element] = 0.0;
      }
      __syncthreads();
      continue;
    }
    double trace = 0.0;
    for (std::size_t element = threadIdx.x; element < matrix_size; element += blockDim.x) {
      const std::size_t row = element % n;
      const std::size_t column = element / n;
      const std::size_t transpose = matrix_index(column, row, n);
      const double symmetric =
          0.5 * (warm_density[offset + element] + warm_density[offset + transpose]);
      density[offset + element] = symmetric;
      trace += symmetric * overlap[overlap_offset + transpose];
    }
    trace = warm_density_block_sum<kWarmDensityThreads>(trace, warp_sums);
    if (threadIdx.x == 0) {
      valid_trace = isfinite(trace) && trace > 0.0;
      scale = valid_trace ? target / trace : 0.0;
      if (valid_trace == 0) warm_invalid[system] = 1;
    }
    __syncthreads();
    if (valid_trace != 0) {
      for (std::size_t element = threadIdx.x; element < matrix_size; element += blockDim.x) {
        density[offset + element] *= scale;
      }
    }
    // Both spin passes reuse the same reduction and scalar slots.
    __syncthreads();
  }
}

__global__ void build_weighted_density_kernel(std::int32_t batch_size, std::int32_t nbf,
                                              const std::int32_t* occupied,
                                              const double* coefficients,
                                              const double* orbital_energies,
                                              const std::uint8_t* active,
                                              double* weighted_density) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch_size) * matrix_size) return;
  const std::int32_t system = static_cast<std::int32_t>(element / matrix_size);
  if (active[system] == 0) return;
  const std::size_t local = element % matrix_size;
  const std::size_t row = local % n;
  const std::size_t column = local / n;
  const std::size_t offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t eigen_offset = static_cast<std::size_t>(system) * n;
  double value = 0.0;
  for (std::int32_t orbital = 0; orbital < occupied[system]; ++orbital) {
    value += 2.0 * orbital_energies[eigen_offset + orbital] *
             coefficients[offset + matrix_index(row, orbital, n)] *
             coefficients[offset + matrix_index(column, orbital, n)];
  }
  weighted_density[element] = value;
}

__global__ void build_spin_weighted_density_kernel(std::int32_t batch_size, std::int32_t nbf,
                                                   const std::int32_t* occupied,
                                                   const double* coefficients,
                                                   const double* orbital_energies,
                                                   const std::uint8_t* active,
                                                   double* weighted_density) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t state_count = static_cast<std::size_t>(batch_size) * 2;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= state_count * matrix_size) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system = state / 2;
  if (active[system] == 0) return;
  const std::size_t local = element % matrix_size;
  const std::size_t row = local % n;
  const std::size_t column = local / n;
  const std::size_t offset = state * matrix_size;
  const std::size_t eigen_offset = state * n;
  double value = 0.0;
  for (std::int32_t orbital = 0; orbital < occupied[state]; ++orbital) {
    value += orbital_energies[eigen_offset + orbital] *
             coefficients[offset + matrix_index(row, orbital, n)] *
             coefficients[offset + matrix_index(column, orbital, n)];
  }
  weighted_density[element] = value;
}

__global__ void sum_uhf_spin_matrices_kernel(std::int32_t batch_size, std::int32_t nbf,
                                             const double* spin_matrices,
                                             const std::uint8_t* active, double* total_matrices) {
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * nbf;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch_size) * matrix_size) return;
  const std::size_t system = element / matrix_size;
  if (active[system] == 0) return;
  const std::size_t local = element % matrix_size;
  const std::size_t alpha_offset = system * 2 * matrix_size;
  total_matrices[element] =
      spin_matrices[alpha_offset + local] + spin_matrices[alpha_offset + matrix_size + local];
}

void launch_build_density_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                 cudaStream_t stream, std::int32_t batch_size, std::int32_t nbf,
                                 const std::int32_t* occupied, const double* coefficients,
                                 const std::uint8_t* active, double* density) {
  generated::occupied_density_kernel<2><<<grid, block, shared_bytes, stream>>>(
      batch_size, 1, nbf, occupied, coefficients, active, density);
}

void launch_build_spin_density_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                      cudaStream_t stream, std::int32_t batch_size,
                                      std::int32_t spin_count, std::int32_t nbf,
                                      const std::int32_t* occupied, const double* coefficients,
                                      const std::uint8_t* active, double* density) {
  generated::occupied_density_kernel<1><<<grid, block, shared_bytes, stream>>>(
      batch_size, spin_count, nbf, occupied, coefficients, active, density);
}

void launch_mix_open_shell_guess_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                        cudaStream_t stream, std::int32_t batch_size,
                                        std::int32_t nbf, const std::int32_t* occupied,
                                        const std::uint8_t* active, double* coefficients) {
  mix_open_shell_guess_kernel<<<grid, block, shared_bytes, stream>>>(batch_size, nbf, occupied,
                                                                     active, coefficients);
}

void launch_apply_warm_density_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                      cudaStream_t stream, std::int32_t batch_size,
                                      std::int32_t nbf, const std::int32_t* occupied,
                                      const std::uint8_t* warm_mask, const double* warm_density,
                                      const double* overlap, double* density,
                                      std::uint8_t* warm_invalid) {
  apply_warm_density_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, nbf, occupied, warm_mask, warm_density, overlap, density, warm_invalid);
}

void launch_apply_uhf_warm_density_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                          cudaStream_t stream, std::int32_t batch_size,
                                          std::int32_t nbf, const std::int32_t* occupied,
                                          const std::uint8_t* warm_mask, const double* warm_density,
                                          const double* overlap, double* density,
                                          std::uint8_t* warm_invalid) {
  apply_uhf_warm_density_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, nbf, occupied, warm_mask, warm_density, overlap, density, warm_invalid);
}

void launch_build_weighted_density_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                          cudaStream_t stream, std::int32_t batch_size,
                                          std::int32_t nbf, const std::int32_t* occupied,
                                          const double* coefficients,
                                          const double* orbital_energies,
                                          const std::uint8_t* active, double* weighted_density) {
  build_weighted_density_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, nbf, occupied, coefficients, orbital_energies, active, weighted_density);
}

void launch_build_spin_weighted_density_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, std::int32_t batch_size,
    std::int32_t nbf, const std::int32_t* occupied, const double* coefficients,
    const double* orbital_energies, const std::uint8_t* active, double* weighted_density) {
  build_spin_weighted_density_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, nbf, occupied, coefficients, orbital_energies, active, weighted_density);
}

void launch_sum_uhf_spin_matrices_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                         cudaStream_t stream, std::int32_t batch_size,
                                         std::int32_t nbf, const double* spin_matrices,
                                         const std::uint8_t* active, double* total_matrices) {
  sum_uhf_spin_matrices_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, nbf, spin_matrices, active, total_matrices);
}

}  // namespace vibeqc::scf::cuda_execution
