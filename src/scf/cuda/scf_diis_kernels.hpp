#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace vibeqc::scf::cuda_execution {

/** Form deterministic block partials for the history including the current
 * residual, before its circular slot is overwritten. The caller lends
 * batch*history^2*parts doubles; parts=ceil(nbf^2*spins/4096). The kernel
 * computes each symmetric history pair once and mirrors its identical partial
 * into the full-square layout consumed by the update kernel. Only active
 * systems and populated slots are read. No atomics or history mutation occur.
 */
void launch_diis_dot_partials(cudaStream_t stream, std::int32_t batch_size, std::int32_t nbf,
                              std::int32_t spins, std::uint32_t history, const double* residual,
                              const double* residual_history, const std::uint8_t* active,
                              const std::uint32_t* counts, const std::uint32_t* heads,
                              std::size_t parts, double* partials);

/** Preserve launch geometry, stream and per-item state routing.
 * cooperative_dots uses one complete 32-lane warp per system, as submitted
 * by compact DF SCF. It computes only unique symmetric Gram entries, mirrors
 * them into the dense solve, and otherwise changes only reduction order while
 * keeping the normalized metric, pivot gate, chronological history retirement
 * and Fock proposal unchanged.
 */
void launch_update_diis_kernel(dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
                               std::int32_t batch_size, std::int32_t nbf,
                               std::int32_t matrices_per_system, std::uint32_t history_capacity,
                               const double* fock, const double* residual,
                               const std::uint8_t* active, double* fock_history,
                               double* residual_history, double* linear_system,
                               double* coefficients, std::uint32_t* history_count,
                               std::uint32_t* history_head, double* effective_fock,
                               bool normalize_metric = false, bool cooperative_dots = false,
                               const double* dot_partials = nullptr, std::size_t parts = 0);

}  // namespace vibeqc::scf::cuda_execution
