#include <math_constants.h>

#include <cmath>

#include "scf/cuda/matrix_index.cuh"
#include "scf/cuda/scf_state_kernels.hpp"

namespace vibeqc::scf::cuda_execution {

__global__ void initialize_state_kernel(std::int32_t batch_size, bool reuse_previous_energy,
                                        const double* energy, std::uint8_t* active,
                                        std::uint8_t* converged, std::uint8_t* failed,
                                        std::uint32_t* iterations, double* previous_energy,
                                        double* energy_change, double* density_rms,
                                        std::uint32_t* diis_count, std::uint32_t* diis_head) {
  const std::int32_t system =
      static_cast<std::int32_t>(static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  if (system >= batch_size) return;
  active[system] = 1;
  converged[system] = 0;
  failed[system] = 0;
  iterations[system] = 0;
  previous_energy[system] = reuse_previous_energy ? energy[system] : CUDART_INF;
  energy_change[system] = CUDART_INF;
  density_rms[system] = CUDART_INF;
  diis_count[system] = 0;
  diis_head[system] = 0;
}

__global__ void inspect_solver_kernel(std::int32_t batch_size, const int* info,
                                      std::uint8_t* active, std::uint8_t* failed,
                                      std::uint8_t* converged) {
  const std::int32_t system =
      static_cast<std::int32_t>(static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  // An inactive state has already converged or failed. Provider writes to its
  // fixed-batch info slot must never overwrite that terminal status.
  if (system >= batch_size || active[system] == 0 || info[system] == 0) return;
  active[system] = 0;
  failed[system] = 1;
  converged[system] = 0;
}

__global__ void expand_spin_active_kernel(std::int32_t batch_size, std::int32_t spin_count,
                                          const std::uint8_t* active, std::uint8_t* spin_active) {
  const std::int32_t state =
      static_cast<std::int32_t>(static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  const std::int32_t state_count = batch_size * spin_count;
  if (state < state_count) spin_active[state] = active[state / spin_count];
}

__global__ void inspect_spin_solver_kernel(std::int32_t batch_size, std::int32_t spin_count,
                                           const int* info, std::uint8_t* active,
                                           std::uint8_t* failed, std::uint8_t* converged) {
  const std::int32_t system =
      static_cast<std::int32_t>(static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  if (system >= batch_size || active[system] == 0) return;
  for (std::int32_t spin = 0; spin < spin_count; ++spin) {
    if (info[system * spin_count + spin] != 0) {
      active[system] = 0;
      failed[system] = 1;
      converged[system] = 0;
      return;
    }
  }
}

void launch_initialize_state_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, std::int32_t batch_size,
    bool reuse_previous_energy, const double* energy, std::uint8_t* active, std::uint8_t* converged,
    std::uint8_t* failed, std::uint32_t* iterations, double* previous_energy, double* energy_change,
    double* density_rms, std::uint32_t* diis_count, std::uint32_t* diis_head) {
  initialize_state_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, reuse_previous_energy, energy, active, converged, failed, iterations,
      previous_energy, energy_change, density_rms, diis_count, diis_head);
}

void launch_inspect_solver_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                  cudaStream_t stream, std::int32_t batch_size, const int* info,
                                  std::uint8_t* active, std::uint8_t* failed,
                                  std::uint8_t* converged) {
  inspect_solver_kernel<<<grid, block, shared_bytes, stream>>>(batch_size, info, active, failed,
                                                               converged);
}

void launch_expand_spin_active_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                      cudaStream_t stream, std::int32_t batch_size,
                                      std::int32_t spin_count, const std::uint8_t* active,
                                      std::uint8_t* spin_active) {
  expand_spin_active_kernel<<<grid, block, shared_bytes, stream>>>(batch_size, spin_count, active,
                                                                   spin_active);
}

void launch_inspect_spin_solver_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                       cudaStream_t stream, std::int32_t batch_size,
                                       std::int32_t spin_count, const int* info,
                                       std::uint8_t* active, std::uint8_t* failed,
                                       std::uint8_t* converged) {
  inspect_spin_solver_kernel<<<grid, block, shared_bytes, stream>>>(batch_size, spin_count, info,
                                                                    active, failed, converged);
}

}  // namespace vibeqc::scf::cuda_execution
