#include <math_constants.h>

#include <cmath>

#include "scf/cuda/matrix_index.cuh"
#include "scf/cuda/scf_constants.hpp"
#include "scf/cuda/scf_convergence_kernels.hpp"

namespace vibeqc::scf::cuda_execution {

__global__ void compute_energy_kernel(std::int32_t batch_size, std::int32_t nbf,
                                      const double* density, const double* hcore,
                                      const double* fock, const double* nuclear_repulsion,
                                      const std::uint8_t* active, double* energy) {
  // One warp owns one system.  The previous one-thread-per-system mapping
  // made the N^2 contraction and its global-memory latency completely serial
  // at large AO counts; all callers launch exactly one 32-thread block per
  // system, which also keeps this graph-capture-safe.
  const std::int32_t system = static_cast<std::int32_t>(blockIdx.x);
  if (system >= batch_size || (active != nullptr && active[system] == 0)) return;
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * nbf;
  const std::size_t offset = static_cast<std::size_t>(system) * matrix_size;
  double value = 0.0;
  for (std::size_t element = threadIdx.x; element < matrix_size; element += blockDim.x) {
    value += 0.5 * density[offset + element] * (hcore[offset + element] + fock[offset + element]);
  }
  for (unsigned delta = warpSize / 2; delta != 0; delta >>= 1) {
    value += __shfl_down_sync(0xffffffffU, value, delta);
  }
  if (threadIdx.x == 0) energy[system] = nuclear_repulsion[system] + value;
}

__global__ void compute_uhf_energy_kernel(std::int32_t batch_size, std::int32_t nbf,
                                          const double* density, const double* hcore,
                                          const double* fock, const double* nuclear_repulsion,
                                          const std::uint8_t* active, double* energy) {
  const std::int32_t system = static_cast<std::int32_t>(blockIdx.x);
  if (system >= batch_size || (active != nullptr && active[system] == 0)) return;
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * nbf;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t alpha_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t beta_offset = alpha_offset + matrix_size;
  double value = 0.0;
  for (std::size_t element = threadIdx.x; element < matrix_size; element += blockDim.x) {
    value += 0.5 * density[alpha_offset + element] *
             (hcore[physical_offset + element] + fock[alpha_offset + element]);
    value += 0.5 * density[beta_offset + element] *
             (hcore[physical_offset + element] + fock[beta_offset + element]);
  }
  for (unsigned delta = warpSize / 2; delta != 0; delta >>= 1) {
    value += __shfl_down_sync(0xffffffffU, value, delta);
  }
  if (threadIdx.x == 0) energy[system] = nuclear_repulsion[system] + value;
}

/** Comparison guard for the nondeterministic FP64 direct-Fock reduction. */
__device__ __forceinline__ double direct_fock_energy_roundoff_guard(bool enabled, double energy,
                                                                    double previous_energy) {
  if (!enabled || !isfinite(previous_energy)) return 0.0;
  const double energy_scale = fmax(1.0, fmax(fabs(energy), fabs(previous_energy)));
  return kDirectFockEnergyRoundoffFactor * kDoubleMachineEpsilon * energy_scale;
}

/** One-warp physical maximum, with nonfinite entries failing closed. The same
 * absolute criterion is used by the shared strict final-state contract. */
__device__ double maximum_physical_residual(const double* values, std::size_t size) {
  if (values == nullptr) return 0.0;
  double maximum = 0.0;
  for (std::size_t element = threadIdx.x; element < size; element += blockDim.x) {
    maximum = isfinite(values[element]) ? fmax(maximum, fabs(values[element])) : CUDART_INF;
  }
  for (unsigned delta = warpSize / 2; delta != 0; delta >>= 1) {
    maximum = fmax(maximum, __shfl_down_sync(0xffffffffU, maximum, delta));
  }
  return __shfl_sync(0xffffffffU, maximum, 0);
}

template <bool RetainConvergedDensity>
__global__ void update_convergence_kernel(
    std::int32_t batch_size, std::int32_t nbf, double energy_tolerance, double density_tolerance,
    bool guard_direct_fock_roundoff, const double* energy, double* previous_energy,
    const double* next_density, double* density, std::uint8_t* active, std::uint8_t* converged,
    std::uint32_t* iterations, double* energy_change, double* density_rms,
    const double* physical_residual, const std::uint32_t* approximate_item_census) {
  // A warp owns one system.  This is intentionally a one-warp block because
  // all scalar state transitions are performed by lane zero after the warp
  // reduction; the matrix walk itself is spread over the 32 lanes.
  const std::int32_t system = static_cast<std::int32_t>(blockIdx.x);
  if (system >= batch_size || active[system] == 0) return;
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * nbf;
  const std::size_t offset = static_cast<std::size_t>(system) * matrix_size;
  const bool target_operator =
      approximate_item_census == nullptr || approximate_item_census[system] == 0;
  const double physical_maximum = maximum_physical_residual(
      physical_residual && target_operator ? physical_residual + offset : nullptr, matrix_size);
  double square = 0.0;
  for (std::size_t element = threadIdx.x; element < matrix_size; element += blockDim.x) {
    const double delta = next_density[offset + element] - density[offset + element];
    square += delta * delta;
    if constexpr (!RetainConvergedDensity) {
      density[offset + element] = next_density[offset + element];
    }
  }
  for (unsigned delta = warpSize / 2; delta != 0; delta >>= 1) {
    square += __shfl_down_sync(0xffffffffU, square, delta);
  }
  int copy_next_density = 0;
  if (threadIdx.x == 0) {
    const std::uint32_t iteration = iterations[system] + 1;
    const bool has_energy_baseline = isfinite(previous_energy[system]);
    const double change =
        has_energy_baseline ? fabs(energy[system] - previous_energy[system]) : CUDART_INF;
    const double roundoff_guard = direct_fock_energy_roundoff_guard(
        guard_direct_fock_roundoff, energy[system], previous_energy[system]);
    const double rms = sqrt(square / static_cast<double>(matrix_size));
    iterations[system] = iteration;
    energy_change[system] = change;
    density_rms[system] = rms;
    const bool did_converge =
        (iteration > 1 || has_energy_baseline) && change < energy_tolerance + roundoff_guard &&
        rms < density_tolerance && physical_maximum <= fmin(1e-8, density_tolerance);
    if (did_converge) {
      converged[system] = 1;
      active[system] = 0;
    } else {
      previous_energy[system] = energy[system];
      copy_next_density = 1;
    }
  }
  copy_next_density = __shfl_sync(0xffffffffU, copy_next_density, 0);
  if constexpr (RetainConvergedDensity) {
    // The raw Fock matrix still corresponds to P_n. Advance to P_{n+1} only
    // when another SCF iteration is required, so finalization can reuse the
    // already computed F(P_n) after convergence instead of rebuilding it.
    if (copy_next_density != 0) {
      for (std::size_t element = threadIdx.x; element < matrix_size; element += blockDim.x) {
        density[offset + element] = next_density[offset + element];
      }
    }
  }
}

template <bool RetainConvergedDensity>
__global__ void update_uhf_convergence_kernel(
    std::int32_t batch_size, std::int32_t nbf, double energy_tolerance, double density_tolerance,
    bool guard_direct_fock_roundoff, const double* energy, double* previous_energy,
    const double* next_density, double* density, std::uint8_t* active, std::uint8_t* converged,
    std::uint32_t* iterations, double* energy_change, double* density_rms,
    const double* physical_residual, const std::uint32_t* approximate_item_census) {
  // Keep UHF's two spin matrices under one warp so the convergence reduction
  // and scalar state transition have the same ordering as RHF.
  const std::int32_t system = static_cast<std::int32_t>(blockIdx.x);
  if (system >= batch_size || active[system] == 0) return;
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * nbf;
  const std::size_t vector_size = 2 * matrix_size;
  const std::size_t offset = static_cast<std::size_t>(system) * vector_size;
  const bool target_operator =
      approximate_item_census == nullptr || approximate_item_census[system] == 0;
  const double physical_maximum = maximum_physical_residual(
      physical_residual && target_operator ? physical_residual + offset : nullptr, vector_size);
  double square = 0.0;
  for (std::size_t element = threadIdx.x; element < vector_size; element += blockDim.x) {
    const double delta = next_density[offset + element] - density[offset + element];
    square += delta * delta;
    if constexpr (!RetainConvergedDensity) {
      density[offset + element] = next_density[offset + element];
    }
  }
  for (unsigned delta = warpSize / 2; delta != 0; delta >>= 1) {
    square += __shfl_down_sync(0xffffffffU, square, delta);
  }
  int copy_next_density = 0;
  if (threadIdx.x == 0) {
    const bool has_energy_baseline = isfinite(previous_energy[system]);
    const double change =
        has_energy_baseline ? fabs(energy[system] - previous_energy[system]) : CUDART_INF;
    const double roundoff_guard = direct_fock_energy_roundoff_guard(
        guard_direct_fock_roundoff, energy[system], previous_energy[system]);
    const double rms = sqrt(square / static_cast<double>(vector_size));
    // Preserve the existing UHF baseline update semantics, including the
    // converged iteration, because it is observable by the next warm replay.
    previous_energy[system] = energy[system];
    energy_change[system] = change;
    density_rms[system] = rms;
    const std::uint32_t iteration = ++iterations[system];
    const bool did_converge =
        (iteration > 1 || has_energy_baseline) && change < energy_tolerance + roundoff_guard &&
        rms < density_tolerance && physical_maximum <= fmin(1e-8, density_tolerance);
    if (did_converge) {
      converged[system] = 1;
      active[system] = 0;
    } else {
      copy_next_density = 1;
    }
  }
  copy_next_density = __shfl_sync(0xffffffffU, copy_next_density, 0);
  if constexpr (RetainConvergedDensity) {
    // Preserve each system's spin densities paired with its raw alpha/beta
    // Fock matrices until per-system finalization selects reuse or rebuild.
    if (copy_next_density != 0) {
      for (std::size_t element = threadIdx.x; element < vector_size; element += blockDim.x) {
        density[offset + element] = next_density[offset + element];
      }
    }
  }
}

__global__ void validate_force_residual_kernel(std::int32_t batch_size, std::int32_t spin_count,
                                               std::int32_t nbf, double density_tolerance,
                                               const double* residual, std::uint8_t* active,
                                               std::uint8_t* converged,
                                               std::uint32_t* tested_count) {
  const auto system = static_cast<std::int32_t>(blockIdx.x);
  if (system >= batch_size || active[system] == 0) return;
  const std::size_t size = static_cast<std::size_t>(spin_count) * nbf * nbf;
  const double maximum = maximum_physical_residual(residual + system * size, size);
  if (threadIdx.x == 0) {
    atomicAdd(tested_count, 1U);
    if (maximum > fmin(1e-8, density_tolerance)) {
      // The canonical projection can change the physical residual. An accepted
      // iterative state alone never licenses forces from a worse final state.
      active[system] = 0;
      converged[system] = 0;
    }
  }
}

__global__ void tail_rhf_loop_kernel(std::int32_t batch_size, std::uint32_t maximum_iterations,
                                     const std::uint8_t* active, const std::uint32_t* iterations) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  bool continue_loop = false;
  for (std::int32_t system = 0; system < batch_size; ++system) {
    continue_loop =
        continue_loop || (active[system] == 1 && iterations[system] < maximum_iterations);
  }
  if (!continue_loop) return;

  // Re-launch the currently executing one-iteration Graph on its tail stream.
  // This is the same device-resident early-stop pattern used by xTBloom: the
  // host submits one Graph and never polls convergence between iterations.
  const cudaGraphExec_t current = cudaGetCurrentGraphExec();
  if (current != nullptr) {
    (void)cudaGraphLaunch(current, cudaStreamGraphTailLaunch);
  }
}

__global__ void select_converged_kernel(std::int32_t batch_size, const std::uint8_t* converged,
                                        const std::uint8_t* failed, std::uint8_t* active) {
  const std::int32_t system =
      static_cast<std::int32_t>(static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  if (system < batch_size) {
    active[system] = converged[system] == 1 && failed[system] == 0 ? 1 : 0;
  }
}

/**
 * Enter the exact target-precision refinement for the items that used the mixed
 * iterative operator: their energy baseline and DIIS history were built from a
 * different operator, so both are cleared and the item continues from the mixed
 * density in exact FP64. Items that never used mixed precision keep their own
 * verdict, and a failed item is never revived.
 */
__global__ void enter_target_refinement_kernel(
    std::int32_t batch_size, const std::uint32_t* item_census, std::uint8_t* active,
    std::uint8_t* converged, const std::uint8_t* failed, std::uint32_t* iterations,
    double* previous_energy, double* energy_change, double* density_rms, std::uint32_t* diis_count,
    std::uint32_t* diis_head) {
  const std::int32_t system =
      static_cast<std::int32_t>(static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  if (system >= batch_size) return;
  if (failed[system] != 0) return;
  if (item_census == nullptr || item_census[system] == 0U) {
    active[system] = 0;
    return;
  }
  active[system] = 1;
  converged[system] = 0;
  iterations[system] = 0;
  previous_energy[system] = CUDART_INF;
  energy_change[system] = CUDART_INF;
  density_rms[system] = CUDART_INF;
  diis_count[system] = 0;
  diis_head[system] = 0;
}

/**
 * Partition converged systems between retained-Fock reuse and exact rebuild.
 *
 * A converged system still owns P_n/F(P_n) because the templated convergence
 * kernel did not advance its density. Only a looser final step restores
 * P_{n+1} and becomes active for the legacy Fock builder. The reuse mask is
 * retained until forces finish so the accepted P_{n+1} warm state can then be
 * restored independently for every system in the bucket.
 */
__global__ void select_final_fock_rebuild_kernel(std::int32_t batch_size, double reuse_density_rms,
                                                 const double* density_rms,
                                                 const std::uint8_t* converged,
                                                 const std::uint8_t* failed,
                                                 std::uint8_t* reuse_mask, std::uint8_t* active,
                                                 std::uint32_t* rebuild_count) {
  const std::int32_t system =
      static_cast<std::int32_t>(static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  if (system >= batch_size) return;
  const bool valid = converged[system] == 1 && failed[system] == 0;
  const bool reuse = valid && density_rms[system] <= reuse_density_rms;
  reuse_mask[system] = reuse ? 1 : 0;
  active[system] = valid && !reuse ? 1 : 0;
  if (active[system] != 0) atomicAdd(rebuild_count, 1U);
}

void launch_compute_energy_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                  cudaStream_t stream, std::int32_t batch_size, std::int32_t nbf,
                                  const double* density, const double* hcore, const double* fock,
                                  const double* nuclear_repulsion, const std::uint8_t* active,
                                  double* energy) {
  compute_energy_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, nbf, density, hcore, fock, nuclear_repulsion, active, energy);
}

void launch_compute_uhf_energy_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                      cudaStream_t stream, std::int32_t batch_size,
                                      std::int32_t nbf, const double* density, const double* hcore,
                                      const double* fock, const double* nuclear_repulsion,
                                      const std::uint8_t* active, double* energy) {
  compute_uhf_energy_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, nbf, density, hcore, fock, nuclear_repulsion, active, energy);
}

void launch_update_convergence_kernel(
    bool retain_converged_density, dim3 grid, dim3 block, std::size_t shared_bytes,
    cudaStream_t stream, std::int32_t batch_size, std::int32_t nbf, double energy_tolerance,
    double density_tolerance, bool guard_direct_fock_roundoff, const double* energy,
    double* previous_energy, const double* next_density, double* density, std::uint8_t* active,
    std::uint8_t* converged, std::uint32_t* iterations, double* energy_change, double* density_rms,
    const double* physical_residual, const std::uint32_t* approximate_item_census) {
  if (retain_converged_density) {
    update_convergence_kernel<true><<<grid, block, shared_bytes, stream>>>(
        batch_size, nbf, energy_tolerance, density_tolerance, guard_direct_fock_roundoff, energy,
        previous_energy, next_density, density, active, converged, iterations, energy_change,
        density_rms, physical_residual, approximate_item_census);
  } else {
    update_convergence_kernel<false><<<grid, block, shared_bytes, stream>>>(
        batch_size, nbf, energy_tolerance, density_tolerance, guard_direct_fock_roundoff, energy,
        previous_energy, next_density, density, active, converged, iterations, energy_change,
        density_rms, physical_residual, approximate_item_census);
  }
}

void launch_update_uhf_convergence_kernel(
    bool retain_converged_density, dim3 grid, dim3 block, std::size_t shared_bytes,
    cudaStream_t stream, std::int32_t batch_size, std::int32_t nbf, double energy_tolerance,
    double density_tolerance, bool guard_direct_fock_roundoff, const double* energy,
    double* previous_energy, const double* next_density, double* density, std::uint8_t* active,
    std::uint8_t* converged, std::uint32_t* iterations, double* energy_change, double* density_rms,
    const double* physical_residual, const std::uint32_t* approximate_item_census) {
  if (retain_converged_density) {
    update_uhf_convergence_kernel<true><<<grid, block, shared_bytes, stream>>>(
        batch_size, nbf, energy_tolerance, density_tolerance, guard_direct_fock_roundoff, energy,
        previous_energy, next_density, density, active, converged, iterations, energy_change,
        density_rms, physical_residual, approximate_item_census);
  } else {
    update_uhf_convergence_kernel<false><<<grid, block, shared_bytes, stream>>>(
        batch_size, nbf, energy_tolerance, density_tolerance, guard_direct_fock_roundoff, energy,
        previous_energy, next_density, density, active, converged, iterations, energy_change,
        density_rms, physical_residual, approximate_item_census);
  }
}

void launch_validate_force_residual_kernel(cudaStream_t stream, std::int32_t batch_size,
                                           std::int32_t spin_count, std::int32_t nbf,
                                           double density_tolerance,
                                           const double* physical_residual, std::uint8_t* active,
                                           std::uint8_t* converged, std::uint32_t* tested_count) {
  validate_force_residual_kernel<<<batch_size, 32, 0, stream>>>(
      batch_size, spin_count, nbf, density_tolerance, physical_residual, active, converged,
      tested_count);
}

void launch_tail_rhf_loop_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                 cudaStream_t stream, std::int32_t batch_size,
                                 std::uint32_t maximum_iterations, const std::uint8_t* active,
                                 const std::uint32_t* iterations) {
  tail_rhf_loop_kernel<<<grid, block, shared_bytes, stream>>>(batch_size, maximum_iterations,
                                                              active, iterations);
}

void launch_select_converged_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                    cudaStream_t stream, std::int32_t batch_size,
                                    const std::uint8_t* converged, const std::uint8_t* failed,
                                    std::uint8_t* active) {
  select_converged_kernel<<<grid, block, shared_bytes, stream>>>(batch_size, converged, failed,
                                                                 active);
}

void launch_enter_target_refinement_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                           cudaStream_t stream, std::int32_t batch_size,
                                           const std::uint32_t* item_census, std::uint8_t* active,
                                           std::uint8_t* converged, const std::uint8_t* failed,
                                           std::uint32_t* iterations, double* previous_energy,
                                           double* energy_change, double* density_rms,
                                           std::uint32_t* diis_count, std::uint32_t* diis_head) {
  enter_target_refinement_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, item_census, active, converged, failed, iterations, previous_energy,
      energy_change, density_rms, diis_count, diis_head);
}

void launch_select_final_fock_rebuild_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                             cudaStream_t stream, std::int32_t batch_size,
                                             double reuse_density_rms, const double* density_rms,
                                             const std::uint8_t* converged,
                                             const std::uint8_t* failed, std::uint8_t* reuse_mask,
                                             std::uint8_t* active, std::uint32_t* rebuild_count) {
  select_final_fock_rebuild_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, reuse_density_rms, density_rms, converged, failed, reuse_mask, active,
      rebuild_count);
}

}  // namespace vibeqc::scf::cuda_execution
