#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace vibeqc::scf::cuda_df {

/** Preserve the exact C used for next D while the old active mask still holds.
 * The factor generation advances with the subsequent density commit, including
 * the converged iteration. Rank-zero channels still receive a generation tag.
 */
void launch_store_device_occupied_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                         cudaStream_t stream, std::size_t nbf,
                                         std::size_t maximum_rank, const std::int32_t* occupied,
                                         const double* coefficients, const std::uint8_t* active,
                                         const std::uint32_t* iterations, double* factors,
                                         std::uint32_t* generations);

/** A sticky validation failure rejects the device solve at its existing host
 * readback boundary. No result built from stale factors can escape to SCF or
 * force consumers; the caller's established numerical recovery remains valid.
 */
void launch_validate_device_occupied_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                            cudaStream_t stream, std::size_t batch_size,
                                            const std::uint32_t* iterations,
                                            const std::uint32_t* alpha_generations,
                                            const std::uint32_t* beta_generations, int* error);

/** Forward the caller's exact launch configuration on its existing stream. */
void launch_assemble_rhf_fock_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                     cudaStream_t stream, std::size_t elements, const double* hcore,
                                     const double* coulomb, const double* exchange, double* fock);

/** Forward the caller's exact launch configuration on its existing stream. */
void launch_assemble_uhf_fock_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                     cudaStream_t stream, std::size_t elements, const double* hcore,
                                     const double* coulomb, const double* alpha_exchange,
                                     const double* beta_exchange, double* alpha_fock,
                                     double* beta_fock);

/** Forward the caller's exact launch configuration on its existing stream. */
void launch_build_device_density_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                        cudaStream_t stream, std::size_t batch_size,
                                        std::size_t nbf, const std::int32_t* occupied,
                                        const double* coefficients, double occupation_weight,
                                        double* density);

/** Forward the caller's exact launch configuration on its existing stream. */
void launch_compute_device_energy_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                         cudaStream_t stream, std::size_t batch_size,
                                         std::size_t nbf, const double* density,
                                         const double* hcore, const double* fock,
                                         const double* nuclear_repulsion, double* energy);

/** Forward the caller's exact launch configuration on its existing stream. */
void launch_compute_device_uhf_energy_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                             cudaStream_t stream, std::size_t batch_size,
                                             std::size_t nbf, const double* alpha_density,
                                             const double* beta_density, const double* hcore,
                                             const double* alpha_fock, const double* beta_fock,
                                             const double* nuclear_repulsion, double* energy);

/** Forward the caller's exact launch configuration on its existing stream. */
void launch_update_device_convergence_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, std::size_t batch_size,
    std::size_t nbf, double energy_tolerance, double density_tolerance, const double* energy,
    double* previous_energy, const double* next_density, double* density, std::uint8_t* active,
    std::uint8_t* converged, std::uint32_t* iterations, double* energy_change, double* density_rms);

/** Forward the caller's exact launch configuration on its existing stream. */
void launch_tail_cuda_density_fitting_scf_graph_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, std::int32_t batch_size,
    std::uint32_t maximum_iterations, const std::uint8_t* active, const std::uint32_t* iterations);

/** Forward the caller's exact launch configuration on its existing stream. */
void launch_update_device_uhf_convergence_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, std::size_t batch_size,
    std::size_t nbf, double energy_tolerance, double density_tolerance, const double* energy,
    double* previous_energy, const double* next_alpha, const double* next_beta,
    double* alpha_density, double* beta_density, std::uint8_t* active, std::uint8_t* converged,
    std::uint32_t* iterations, double* energy_change, double* density_rms);

}  // namespace vibeqc::scf::cuda_df
