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
#include "scf/cuda/scf_convergence_policy.cuh"

namespace vibeqc::scf::cuda_df {
using cuda_execution::hf_iteration_converged;
using cuda_execution::maximum_physical_residual;

__global__ void density_exchange_factor_kernel(std::size_t nbf, std::size_t rank,
                                               const double* vectors, const double* values,
                                               double* factor) {
  const std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k < nbf * rank) {
    const std::size_t offset = nbf - rank;
    factor[k] = vectors[offset * nbf + k] * sqrt(values[offset + k / nbf]);
  }
}

__global__ void density_exchange_error_kernel(std::size_t elements, const double* density,
                                              const double* reconstructed, double* errors) {
  __shared__ double maxima[256], squares[256];
  double maximum = 0, sum = 0;
  for (std::size_t k = threadIdx.x; k < elements; k += blockDim.x) {
    const double error = fabs(density[k] - reconstructed[k]);
    // fmax alone would hide NaNs. Reject nonfinite reconstruction explicitly.
    maximum = isfinite(error) ? fmax(maximum, error) : CUDART_INF;
    sum += error * error;
  }
  maxima[threadIdx.x] = maximum;
  squares[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned width = 128; width; width >>= 1) {
    if (threadIdx.x < width) {
      maxima[threadIdx.x] = fmax(maxima[threadIdx.x], maxima[threadIdx.x + width]);
      squares[threadIdx.x] += squares[threadIdx.x + width];
    }
    __syncthreads();
  }
  if (!threadIdx.x) {
    errors[0] = maxima[0];
    errors[1] = sqrt(squares[0] / elements);
  }
}

void launch_density_exchange_factor(cudaStream_t stream, std::size_t nbf, std::size_t rank,
                                    const double* vectors, const double* values, double* factor) {
  if (rank)
    density_exchange_factor_kernel<<<(nbf * rank + 255) / 256, 256, 0, stream>>>(nbf, rank, vectors,
                                                                                 values, factor);
}

void launch_density_exchange_error(cudaStream_t stream, std::size_t elements, const double* density,
                                   const double* reconstructed, double* errors) {
  density_exchange_error_kernel<<<1, 256, 0, stream>>>(elements, density, reconstructed, errors);
}

__global__ void store_device_final_frame_kernel(
    std::size_t nbf, const double* coefficients, const double* eigenvalues, const int* info,
    const std::uint8_t* active, const std::uint32_t* iterations, double* retained_coefficients,
    double* retained_values, std::uint64_t* generations, int* retained_info) {
  const auto item = static_cast<std::size_t>(blockIdx.x);
  if (!active[item]) return;
  for (std::size_t k = threadIdx.x; k < nbf * nbf; k += blockDim.x)
    retained_coefficients[item * nbf * nbf + k] = coefficients[item * nbf * nbf + k];
  for (std::size_t k = threadIdx.x; k < nbf; k += blockDim.x)
    retained_values[item * nbf + k] = eigenvalues[item * nbf + k];
  if (threadIdx.x == 0) {
    // Imported D is generation one. The pending projection will commit the
    // next generation, including an item that converges in this iteration.
    generations[item] = static_cast<std::uint64_t>(iterations[item]) + 2;
    retained_info[item] = info[item];
  }
}

void launch_store_device_final_frame_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                            cudaStream_t stream, std::size_t nbf,
                                            const double* coefficients, const double* eigenvalues,
                                            const int* info, const std::uint8_t* active,
                                            const std::uint32_t* iterations,
                                            double* retained_coefficients, double* retained_values,
                                            std::uint64_t* generations, int* retained_info) {
  store_device_final_frame_kernel<<<grid, block, shared_bytes, stream>>>(
      nbf, coefficients, eigenvalues, info, active, iterations, retained_coefficients,
      retained_values, generations, retained_info);
}

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
                                                const std::uint32_t* beta_generations, int* error,
                                                bool retained_seed) {
  const auto system = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system >= batch_size) return;
  if ((!iterations[system] && !retained_seed) || alpha_generations[system] != iterations[system] ||
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
                                            const std::uint32_t* beta_generations, int* error,
                                            bool retained_seed) {
  validate_device_occupied_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, iterations, alpha_generations, beta_generations, error, retained_seed);
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
  // Production RHF/UHF use exact power-of-two occupation weights (2 or 1).
  // In that domain the density is symmetric, so contract each AO pair once.
  // Mirroring can only replace a lower-triangle worker when the complete
  // flat domain is launched. A clipped launch must retain every original write.
  const auto launched = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  const bool symmetric_density = (occupation_weight == 1.0 || occupation_weight == 2.0) &&
                                 launched >= batch_size * matrix_elements;
  if (symmetric_density && row > column) return;
  const std::size_t offset = system * matrix_elements;
  double value = 0.0;
  for (std::int32_t orbital = 0; orbital < occupied[system]; ++orbital) {
    value += occupation_weight * coefficients[offset + row + orbital * nbf] *
             coefficients[offset + column + orbital * nbf];
  }
  density[element] = value;
  if (symmetric_density && row != column) density[offset + column + row * nbf] = value;
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

__global__ void update_device_convergence_kernel(
    std::size_t batch_size, std::size_t nbf, double energy_tolerance, double density_tolerance,
    const double* energy, double* previous_energy, const double* next_density, double* density,
    std::uint8_t* active, std::uint8_t* converged, std::uint32_t* iterations, double* energy_change,
    double* density_rms, const double* physical_residual) {
  const std::size_t system = blockIdx.x;
  if (system >= batch_size || active[system] == 0) return;
  const std::size_t matrix_elements = nbf * nbf;
  const std::size_t offset = system * matrix_elements;
  const double physical_maximum = maximum_physical_residual(
      physical_residual ? physical_residual + offset : nullptr, matrix_elements);
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
    if (hf_iteration_converged(energy[system], previous_energy[system], energy_tolerance,
                               density_tolerance, rms, physical_maximum)) {
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
    std::uint8_t* converged, std::uint32_t* iterations, double* energy_change, double* density_rms,
    const double* physical_residual) {
  const std::size_t system = blockIdx.x;
  if (system >= batch_size || active[system] == 0) return;
  const std::size_t matrix_elements = nbf * nbf;
  const std::size_t offset = system * matrix_elements;
  // DF DIIS stores the two physical spin residuals adjacent within each item.
  const double physical_maximum = maximum_physical_residual(
      physical_residual ? physical_residual + 2 * offset : nullptr, 2 * matrix_elements);
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
    if (hf_iteration_converged(energy[system], previous_energy[system], energy_tolerance,
                               density_tolerance, rms, physical_maximum)) {
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
void launch_update_device_convergence_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, std::size_t batch_size,
    std::size_t nbf, double energy_tolerance, double density_tolerance, const double* energy,
    double* previous_energy, const double* next_density, double* density, std::uint8_t* active,
    std::uint8_t* converged, std::uint32_t* iterations, double* energy_change, double* density_rms,
    const double* physical_residual) {
  update_device_convergence_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, nbf, energy_tolerance, density_tolerance, energy, previous_energy, next_density,
      density, active, converged, iterations, energy_change, density_rms, physical_residual);
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
    std::uint32_t* iterations, double* energy_change, double* density_rms,
    const double* physical_residual) {
  update_device_uhf_convergence_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, nbf, energy_tolerance, density_tolerance, energy, previous_energy, next_alpha,
      next_beta, alpha_density, beta_density, active, converged, iterations, energy_change,
      density_rms, physical_residual);
}
}  // namespace vibeqc::scf::cuda_df
