#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "molecule/basis.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

/** Shell-local public AO expansion into the existing normalized Cartesian ABI.
 * One record per public AO and per batch item; indices remain in Cartesian
 * order so contraction summation is unchanged. Cartesian public AOs have
 * exactly one term. The normalized basis owner defines the finite s--f bound.
 */
struct DfPublicAoExpansion {
  std::uint32_t count{};
  std::int32_t cartesian[molecule::kMaximumAoExpansionTerms]{};
  double coefficients[molecule::kMaximumAoExpansionTerms]{};
};

/** Host-callable DF generation boundary. Callers own valid ranges, device buffers, stream ordering
 * and last-error inspection. */
/** Submit the selected value/coordinate-response specialization with the caller's launch geometry.
 */
void launch_build_cuda_df_integrals_kernel(
    bool derivative, dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    DeviceBatch batch, std::size_t orbital_count, std::size_t auxiliary_count,
    std::size_t dummy_index, std::size_t metric_elements, std::size_t three_center_elements,
    std::size_t system_base, std::size_t launch_batch_size, std::int64_t derivative_coordinate,
    double* metric, double* three_center, unsigned math = 0U, unsigned lanes = 1U);

/** Submit the selected value/coordinate-response specialization with the caller's launch geometry.
 */
void launch_build_cuda_df_metric_source_kernel(
    bool derivative, dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    DeviceBatch batch, std::size_t cartesian_orbital_count, std::size_t cartesian_auxiliary_count,
    std::size_t public_naux, std::size_t dummy_index, std::size_t system,
    std::size_t auxiliary_row_begin, std::size_t auxiliary_row_count,
    std::int64_t derivative_coordinate, const DfPublicAoExpansion* auxiliary_to_cartesian,
    double* output, unsigned mapping = 0U);

/** Submit the selected value/coordinate-response specialization with the caller's launch geometry.
 */
void launch_build_cuda_df_transformed_tile_kernel(
    bool derivative, dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    DeviceBatch batch, std::size_t cartesian_orbital_count, std::size_t cartesian_auxiliary_count,
    std::size_t public_nbf, std::size_t public_naux, std::size_t dummy_index, std::size_t system,
    std::size_t pair_begin, std::size_t pair_count, std::size_t auxiliary_begin,
    std::size_t auxiliary_count, std::int64_t derivative_coordinate,
    const DfPublicAoExpansion* orbital_to_cartesian,
    const DfPublicAoExpansion* auxiliary_to_cartesian, const double* inverse_square_root,
    bool apply_metric_transform, double* output, unsigned mapping = 0U, unsigned math = 0U);

}  // namespace vibeqc::scf::cuda_execution
