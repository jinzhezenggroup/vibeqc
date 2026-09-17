#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace vibeqc::scf::cuda_execution {

/** Preserve launch geometry, stream and per-item state routing. */
void launch_compute_energy_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                  cudaStream_t stream, std::int32_t batch_size, std::int32_t nbf,
                                  const double* density, const double* hcore, const double* fock,
                                  const double* nuclear_repulsion, const std::uint8_t* active,
                                  double* energy);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_compute_uhf_energy_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                      cudaStream_t stream, std::int32_t batch_size,
                                      std::int32_t nbf, const double* density, const double* hcore,
                                      const double* fock, const double* nuclear_repulsion,
                                      const std::uint8_t* active, double* energy);

/** A supplied physical residual adds max|FPS-SPF| <= min(1e-8, density_tolerance)
 * to the iterative stop. Null preserves energy-only/reference behavior. Each
 * block must contain exactly one warp; residuals precede DIIS extrapolation.
 * A nonzero per-item approximate census defers this test to target refinement;
 * exact neighbors retain the gate. Refinement must pass a null census. */
void launch_update_convergence_kernel(
    bool retain_converged_density, dim3 grid, dim3 block, std::size_t shared_bytes,
    cudaStream_t stream, std::int32_t batch_size, std::int32_t nbf, double energy_tolerance,
    double density_tolerance, bool guard_direct_fock_roundoff, const double* energy,
    double* previous_energy, const double* next_density, double* density, std::uint8_t* active,
    std::uint8_t* converged, std::uint32_t* iterations, double* energy_change, double* density_rms,
    const double* physical_residual = nullptr,
    const std::uint32_t* approximate_item_census = nullptr);

/** UHF counterpart; the physical maximum covers both spin channels. */
void launch_update_uhf_convergence_kernel(
    bool retain_converged_density, dim3 grid, dim3 block, std::size_t shared_bytes,
    cudaStream_t stream, std::int32_t batch_size, std::int32_t nbf, double energy_tolerance,
    double density_tolerance, bool guard_direct_fock_roundoff, const double* energy,
    double* previous_energy, const double* next_density, double* density, std::uint8_t* active,
    std::uint8_t* converged, std::uint32_t* iterations, double* energy_change, double* density_rms,
    const double* physical_residual = nullptr,
    const std::uint32_t* approximate_item_census = nullptr);

/** Validate the rebuilt force determinant, clearing convergence on a failed
 * physical residual. One warp per item; tested_count counts all active items,
 * including rejected ones, and must be zeroed by the caller. No host products. */
void launch_validate_force_residual_kernel(cudaStream_t stream, std::int32_t batch_size,
                                           std::int32_t spin_count, std::int32_t nbf,
                                           double density_tolerance,
                                           const double* physical_residual, std::uint8_t* active,
                                           std::uint8_t* converged, std::uint32_t* tested_count);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_tail_rhf_loop_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                 cudaStream_t stream, std::int32_t batch_size,
                                 std::uint32_t maximum_iterations, const std::uint8_t* active,
                                 const std::uint32_t* iterations);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_select_converged_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                    cudaStream_t stream, std::int32_t batch_size,
                                    const std::uint8_t* converged, const std::uint8_t* failed,
                                    std::uint8_t* active);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_enter_target_refinement_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                           cudaStream_t stream, std::int32_t batch_size,
                                           const std::uint32_t* item_census, std::uint8_t* active,
                                           std::uint8_t* converged, const std::uint8_t* failed,
                                           std::uint32_t* iterations, double* previous_energy,
                                           double* energy_change, double* density_rms,
                                           std::uint32_t* diis_count, std::uint32_t* diis_head);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_select_final_fock_rebuild_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                             cudaStream_t stream, std::int32_t batch_size,
                                             double reuse_density_rms, const double* density_rms,
                                             const std::uint8_t* converged,
                                             const std::uint8_t* failed, std::uint8_t* reuse_mask,
                                             std::uint8_t* active, std::uint32_t* rebuild_count);

}  // namespace vibeqc::scf::cuda_execution
