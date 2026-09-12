#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "scf/cuda/df_scf_kernels.hpp"

namespace vibeqc::scf::cuda_df {

__global__ void store_device_occupied_kernel(std::size_t nbf, std::size_t maximum_rank,
                                             const std::int32_t* occupied,
                                             const double* coefficients, const std::uint8_t* active,
                                             const std::uint32_t* iterations, double* factors,
                                             std::uint32_t* generations) {
  const auto system = static_cast<std::size_t>(blockIdx.x);
  if (!active[system]) return;
  const auto stride = nbf * maximum_rank;
  for (std::size_t element = threadIdx.x; element < stride; element += blockDim.x)
    factors[system * stride + element] = element / nbf < static_cast<std::size_t>(occupied[system])
                                             ? coefficients[system * nbf * nbf + element]
                                             : 0.0;
  if (threadIdx.x == 0) generations[system] = iterations[system] + 1;
}

__global__ void validate_device_occupied_kernel(std::size_t batch_size,
                                                const std::uint32_t* iterations,
                                                const std::uint32_t* alpha_generations,
                                                const std::uint32_t* beta_generations, int* error) {
  const auto system = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system >= batch_size) return;
  if (!iterations[system] || alpha_generations[system] != iterations[system] ||
      (beta_generations && beta_generations[system] != iterations[system]))
    atomicExch(error, 1);
}

void launch_store_device_occupied_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                         cudaStream_t stream, std::size_t nbf,
                                         std::size_t maximum_rank, const std::int32_t* occupied,
                                         const double* coefficients, const std::uint8_t* active,
                                         const std::uint32_t* iterations, double* factors,
                                         std::uint32_t* generations) {
  store_device_occupied_kernel<<<grid, block, shared_bytes, stream>>>(
      nbf, maximum_rank, occupied, coefficients, active, iterations, factors, generations);
}

void launch_validate_device_occupied_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                            cudaStream_t stream, std::size_t batch_size,
                                            const std::uint32_t* iterations,
                                            const std::uint32_t* alpha_generations,
                                            const std::uint32_t* beta_generations, int* error) {
  validate_device_occupied_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, iterations, alpha_generations, beta_generations, error);
}

// Existing DF arithmetic and reduction order; host orchestration compiles separately.
__global__ void assemble_rhf_fock_kernel(std::size_t elements, const double* hcore,
                                         const double* coulomb, const double* exchange,
                                         double* fock) {
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= elements) return;
  fock[element] = hcore[element] + coulomb[element] - 0.5 * exchange[element];
}

__global__ void assemble_uhf_fock_kernel(std::size_t elements, const double* hcore,
                                         const double* coulomb, const double* alpha_exchange,
                                         const double* beta_exchange, double* alpha_fock,
                                         double* beta_fock) {
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= elements) return;
  alpha_fock[element] = hcore[element] + coulomb[element] - alpha_exchange[element];
  beta_fock[element] = hcore[element] + coulomb[element] - beta_exchange[element];
}

__global__ void build_device_density_kernel(std::size_t batch_size, std::size_t nbf,
                                            const std::int32_t* occupied,
                                            const double* coefficients, double occupation_weight,
                                            double* density) {
  const std::size_t matrix_elements = nbf * nbf;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= batch_size * matrix_elements) return;
  const std::size_t system = element / matrix_elements;
  const std::size_t local = element % matrix_elements;
  const std::size_t row = local % nbf;
  const std::size_t column = local / nbf;
  const std::size_t offset = system * matrix_elements;
  double value = 0.0;
  for (std::int32_t orbital = 0; orbital < occupied[system]; ++orbital) {
    value += occupation_weight * coefficients[offset + row + orbital * nbf] *
             coefficients[offset + column + orbital * nbf];
  }
  density[element] = value;
}

__global__ void compute_device_energy_kernel(std::size_t batch_size, std::size_t nbf,
                                             const double* density, const double* hcore,
                                             const double* fock, const double* nuclear_repulsion,
                                             double* energy) {
  const std::size_t system = blockIdx.x;
  if (system >= batch_size) return;
  const std::size_t matrix_elements = nbf * nbf;
  const std::size_t offset = system * matrix_elements;
  double value = 0.0;
  for (std::size_t element = threadIdx.x; element < matrix_elements; element += blockDim.x) {
    value += 0.5 * density[offset + element] * (hcore[offset + element] + fock[offset + element]);
  }
  for (unsigned delta = warpSize / 2; delta != 0; delta >>= 1) {
    value += __shfl_down_sync(0xffffffffU, value, delta);
  }
  if (threadIdx.x == 0) energy[system] = value + nuclear_repulsion[system];
}

__global__ void compute_device_uhf_energy_kernel(std::size_t batch_size, std::size_t nbf,
                                                 const double* alpha_density,
                                                 const double* beta_density, const double* hcore,
                                                 const double* alpha_fock, const double* beta_fock,
                                                 const double* nuclear_repulsion, double* energy) {
  const std::size_t system = blockIdx.x;
  if (system >= batch_size) return;
  const std::size_t matrix_elements = nbf * nbf;
  const std::size_t physical_offset = system * matrix_elements;
  const std::size_t spin_offset = system * matrix_elements;
  double value = 0.0;
  for (std::size_t element = threadIdx.x; element < matrix_elements; element += blockDim.x) {
    value += 0.5 * alpha_density[spin_offset + element] *
             (hcore[physical_offset + element] + alpha_fock[spin_offset + element]);
    value += 0.5 * beta_density[spin_offset + element] *
             (hcore[physical_offset + element] + beta_fock[spin_offset + element]);
  }
  for (unsigned delta = warpSize / 2; delta != 0; delta >>= 1) {
    value += __shfl_down_sync(0xffffffffU, value, delta);
  }
  if (threadIdx.x == 0) energy[system] = value + nuclear_repulsion[system];
}

__global__ void update_device_convergence_kernel(std::size_t batch_size, std::size_t nbf,
                                                 double energy_tolerance, double density_tolerance,
                                                 const double* energy, double* previous_energy,
                                                 const double* next_density, double* density,
                                                 std::uint8_t* active, std::uint8_t* converged,
                                                 std::uint32_t* iterations, double* energy_change,
                                                 double* density_rms) {
  const std::size_t system = blockIdx.x;
  if (system >= batch_size || active[system] == 0) return;
  const std::size_t matrix_elements = nbf * nbf;
  const std::size_t offset = system * matrix_elements;
  double square = 0.0;
  for (std::size_t element = threadIdx.x; element < matrix_elements; element += blockDim.x) {
    const double delta = next_density[offset + element] - density[offset + element];
    square += delta * delta;
  }
  for (unsigned delta = warpSize / 2; delta != 0; delta >>= 1) {
    square += __shfl_down_sync(0xffffffffU, square, delta);
  }
  if (threadIdx.x == 0) {
    const std::uint32_t iteration = iterations[system] + 1;
    const bool has_baseline = isfinite(previous_energy[system]);
    const double change =
        has_baseline ? fabs(energy[system] - previous_energy[system]) : CUDART_INF;
    const double rms = sqrt(square / static_cast<double>(matrix_elements));
    iterations[system] = iteration;
    energy_change[system] = change;
    density_rms[system] = rms;
    if ((iteration > 1 || has_baseline) && change < energy_tolerance && rms < density_tolerance) {
      converged[system] = 1;
      active[system] = 0;
    } else {
      previous_energy[system] = energy[system];
    }
  }
  __syncwarp();
  // Every iteration advances the resident density, including the converged
  // one.  This makes the final host copy the density associated with the
  // convergence test and avoids a second device-to-device staging pass.
  for (std::size_t element = threadIdx.x; element < matrix_elements; element += blockDim.x) {
    density[offset + element] = next_density[offset + element];
  }
}

__global__ void tail_cuda_density_fitting_scf_graph_kernel(std::int32_t batch_size,
                                                           std::uint32_t maximum_iterations,
                                                           const std::uint8_t* active,
                                                           const std::uint32_t* iterations) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  bool continue_loop = false;
  for (std::int32_t system = 0; system < batch_size; ++system) {
    continue_loop =
        continue_loop || (active[system] != 0 && iterations[system] < maximum_iterations);
  }
  if (!continue_loop) return;
  const cudaGraphExec_t current = cudaGetCurrentGraphExec();
  if (current != nullptr) {
    (void)cudaGraphLaunch(current, cudaStreamGraphTailLaunch);
  }
}

__global__ void update_device_uhf_convergence_kernel(
    std::size_t batch_size, std::size_t nbf, double energy_tolerance, double density_tolerance,
    const double* energy, double* previous_energy, const double* next_alpha,
    const double* next_beta, double* alpha_density, double* beta_density, std::uint8_t* active,
    std::uint8_t* converged, std::uint32_t* iterations, double* energy_change,
    double* density_rms) {
  const std::size_t system = blockIdx.x;
  if (system >= batch_size || active[system] == 0) return;
  const std::size_t matrix_elements = nbf * nbf;
  const std::size_t offset = system * matrix_elements;
  double square = 0.0;
  for (std::size_t element = threadIdx.x; element < matrix_elements; element += blockDim.x) {
    const double da = next_alpha[offset + element] - alpha_density[offset + element];
    const double db = next_beta[offset + element] - beta_density[offset + element];
    square += da * da + db * db;
  }
  for (unsigned delta = warpSize / 2; delta != 0; delta >>= 1) {
    square += __shfl_down_sync(0xffffffffU, square, delta);
  }
  if (threadIdx.x == 0) {
    const std::uint32_t iteration = iterations[system] + 1;
    const bool has_baseline = isfinite(previous_energy[system]);
    const double change =
        has_baseline ? fabs(energy[system] - previous_energy[system]) : CUDART_INF;
    const double rms = sqrt(square / static_cast<double>(2 * matrix_elements));
    iterations[system] = iteration;
    energy_change[system] = change;
    density_rms[system] = rms;
    if ((iteration > 1 || has_baseline) && change < energy_tolerance && rms < density_tolerance) {
      converged[system] = 1;
      active[system] = 0;
    } else {
      previous_energy[system] = energy[system];
    }
  }
  __syncwarp();
  for (std::size_t element = threadIdx.x; element < matrix_elements; element += blockDim.x) {
    alpha_density[offset + element] = next_alpha[offset + element];
    beta_density[offset + element] = next_beta[offset + element];
  }
}

void launch_assemble_rhf_fock_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                     cudaStream_t stream, std::size_t elements, const double* hcore,
                                     const double* coulomb, const double* exchange, double* fock) {
  assemble_rhf_fock_kernel<<<grid, block, shared_bytes, stream>>>(elements, hcore, coulomb,
                                                                  exchange, fock);
}
void launch_assemble_uhf_fock_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                     cudaStream_t stream, std::size_t elements, const double* hcore,
                                     const double* coulomb, const double* alpha_exchange,
                                     const double* beta_exchange, double* alpha_fock,
                                     double* beta_fock) {
  assemble_uhf_fock_kernel<<<grid, block, shared_bytes, stream>>>(
      elements, hcore, coulomb, alpha_exchange, beta_exchange, alpha_fock, beta_fock);
}
void launch_build_device_density_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                        cudaStream_t stream, std::size_t batch_size,
                                        std::size_t nbf, const std::int32_t* occupied,
                                        const double* coefficients, double occupation_weight,
                                        double* density) {
  build_device_density_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, nbf, occupied, coefficients, occupation_weight, density);
}
void launch_compute_device_energy_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                         cudaStream_t stream, std::size_t batch_size,
                                         std::size_t nbf, const double* density,
                                         const double* hcore, const double* fock,
                                         const double* nuclear_repulsion, double* energy) {
  compute_device_energy_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, nbf, density, hcore, fock, nuclear_repulsion, energy);
}
void launch_compute_device_uhf_energy_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                             cudaStream_t stream, std::size_t batch_size,
                                             std::size_t nbf, const double* alpha_density,
                                             const double* beta_density, const double* hcore,
                                             const double* alpha_fock, const double* beta_fock,
                                             const double* nuclear_repulsion, double* energy) {
  compute_device_uhf_energy_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, nbf, alpha_density, beta_density, hcore, alpha_fock, beta_fock, nuclear_repulsion,
      energy);
}
void launch_update_device_convergence_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                             cudaStream_t stream, std::size_t batch_size,
                                             std::size_t nbf, double energy_tolerance,
                                             double density_tolerance, const double* energy,
                                             double* previous_energy, const double* next_density,
                                             double* density, std::uint8_t* active,
                                             std::uint8_t* converged, std::uint32_t* iterations,
                                             double* energy_change, double* density_rms) {
  update_device_convergence_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, nbf, energy_tolerance, density_tolerance, energy, previous_energy, next_density,
      density, active, converged, iterations, energy_change, density_rms);
}
void launch_tail_cuda_density_fitting_scf_graph_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, std::int32_t batch_size,
    std::uint32_t maximum_iterations, const std::uint8_t* active, const std::uint32_t* iterations) {
  tail_cuda_density_fitting_scf_graph_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, maximum_iterations, active, iterations);
}
void launch_update_device_uhf_convergence_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, std::size_t batch_size,
    std::size_t nbf, double energy_tolerance, double density_tolerance, const double* energy,
    double* previous_energy, const double* next_alpha, const double* next_beta,
    double* alpha_density, double* beta_density, std::uint8_t* active, std::uint8_t* converged,
    std::uint32_t* iterations, double* energy_change, double* density_rms) {
  update_device_uhf_convergence_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, nbf, energy_tolerance, density_tolerance, energy, previous_energy, next_alpha,
      next_beta, alpha_density, beta_density, active, converged, iterations, energy_change,
      density_rms);
}
}  // namespace vibeqc::scf::cuda_df
