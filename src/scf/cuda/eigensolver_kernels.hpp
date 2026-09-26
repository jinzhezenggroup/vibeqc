#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "scf/cuda/eigensolver_types.hpp"

namespace vibeqc::scf::cuda_execution {

/** Narrow host-callable launch boundary; wrappers leave last-error inspection and synchronization
 * to their caller. */
/** Submit one kernel using the caller's validated launch geometry. */
void launch_begin_inactive_eigensolver_profile_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    std::int32_t physical_batch_size, std::int32_t solver_batch_size, std::uint32_t family,
    bool provider_invoked, bool cublas_transformed_inactive, const std::uint8_t* physical_active,
    const std::uint8_t* solver_active, std::uint32_t capacity, std::uint32_t* count,
    DeviceInactiveEigensolverProfileEntry* entries);

/** Submit one kernel using the caller's validated launch geometry. */
void launch_finish_inactive_eigensolver_profile_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    std::int32_t solver_batch_size, const std::uint8_t* solver_active, const int* info,
    std::uint32_t capacity, const std::uint32_t* count,
    DeviceInactiveEigensolverProfileEntry* entries);

/** Submit one kernel using the caller's validated launch geometry. */
void launch_sanitize_inactive_solver_input_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    std::int32_t solver_batch_size, std::int32_t nbf, const std::uint8_t* solver_active,
    double* matrices, int* info, std::uint32_t profile_capacity, const std::uint32_t* profile_count,
    DeviceInactiveEigensolverProfileEntry* profile_entries);

/** Submit one kernel using the caller's validated launch geometry. */
void launch_start_inactive_eigensolver_timer_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                                    cudaStream_t stream, std::uint32_t capacity,
                                                    const std::uint32_t* count,
                                                    DeviceInactiveEigensolverProfileEntry* entries);

/** Submit one kernel using the caller's validated launch geometry. */
void launch_symmetric_eigen_graph_maximum_pivot_kernel(dim3 grid, dim3 block,
                                                       std::size_t shared_bytes,
                                                       cudaStream_t stream, std::int32_t batch_size,
                                                       std::int32_t nbf, double* matrices,
                                                       double* eigenvectors, double* eigenvalues,
                                                       int* info, const std::uint8_t* active);

/** Submit one kernel using the caller's validated launch geometry. */
void launch_symmetric_eigen_small_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                         cudaStream_t stream, std::int32_t batch_size,
                                         std::int32_t nbf, double* matrices, double* eigenvalues,
                                         int* info, const std::uint8_t* active);

}  // namespace vibeqc::scf::cuda_execution
