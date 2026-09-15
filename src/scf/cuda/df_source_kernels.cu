#include <cuda_runtime.h>

#include <type_traits>

#include "generated_df_derivative_policy.cuh"
#include "generated_df_policy.cuh"
#include "molecule/basis.hpp"
#include "runtime/cuda_gaussian_products.cuh"
#include "scf/cuda/df_source_kernels.hpp"

namespace vibeqc::scf::cuda_execution {

/** Generated-policy basis traversal and bounded public-layout contractions, extracted from direct
 * HF without introducing a second scientific implementation. */
namespace {
constexpr std::size_t kMaximumAoExpansionTerms = molecule::kMaximumAoExpansionTerms;

/** Borrow the packed integral metadata through the common normalized-basis ABI. */
__device__ runtime::cuda_gaussian_products::BasisView df_basis_view(const DeviceBatch& batch) {
  return {
      static_cast<std::size_t>(batch.nbf), batch.shell_atoms,         batch.ao_shells,
      batch.shell_primitive_offsets,       batch.ao_term_counts,      batch.ao_term_angular,
      batch.ao_term_coefficients,          batch.primitive_exponents, batch.primitive_coefficients};
}

/** Values and coordinate responses instantiate the same generic basis traversal.
 * Metric uses two real factors; three-center uses three. Mathematical center
 * channels are projected onto physical atoms only after primitive contraction.
 * The packed dummy remains an ABI detail and never enters a scientific policy.
 * Let the compiler inline this metadata adapter. Forcing a device call spills
 * the surrounding transformed-tile state across each Cartesian component;
 * scalar mathematical evaluators retain their own independent call boundaries.
 */
template <bool Derivative, bool Metric>
__device__ double contracted_df(const DeviceBatch& batch, std::int32_t system, std::int32_t first,
                                std::int32_t second, std::int32_t auxiliary, std::int32_t dummy,
                                std::int64_t coordinate, unsigned lane = 0U, unsigned lanes = 1U) {
  (void)dummy;
  namespace products = runtime::cuda_gaussian_products;
  using Policy =
      std::conditional_t<Derivative, generated_df_policy::Derivative, generated_df_policy::Value>;
  constexpr unsigned rank = Metric ? 2 : 3;
  const auto basis = df_basis_view(batch);
  const std::int64_t base = static_cast<std::int64_t>(system) * batch.nbf;
  products::Factor factors[rank]{{basis, base + first}, {basis, base + auxiliary}};
  if constexpr (!Metric) {
    factors[1] = {basis, base + second};
    factors[2] = {basis, base + auxiliary};
  }
  if constexpr (Derivative) {
    bool affected = false;
    for (unsigned slot = 0; slot < rank; ++slot)
      affected = affected || basis.shell_atoms[basis.ao_shells[factors[slot].ao]] == coordinate / 3;
    if (!affected) return 0.0;
    const auto result =
        products::contract<Policy, kMaximumAoExpansionTerms>(factors, batch.positions, lane, lanes);
    return products::coordinate(factors, result, coordinate);
  } else {
    return products::contract<Policy, kMaximumAoExpansionTerms>(factors, batch.positions, lane,
                                                                lanes);
  }
}

/** Evaluate raw Cartesian M[P,Q] and A[mu,nu,P], without pair compression. */
template <bool Derivative>
__global__ void build_cuda_df_integrals_kernel(
    DeviceBatch batch, std::size_t orbital_count, std::size_t auxiliary_count,
    std::size_t dummy_index, std::size_t metric_elements, std::size_t three_center_elements,
    std::size_t system_base, std::size_t launch_batch_size, std::int64_t derivative_coordinate,
    double* metric, double* three_center) {
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t per_system = metric_elements + three_center_elements;
  const std::size_t total = launch_batch_size * per_system;
  if (element >= total) return;
  const std::size_t local_system = element / per_system;
  const std::size_t system = system_base + local_system;
  const std::size_t system_local = element % per_system;
  const std::int64_t system_derivative_coordinate =
      derivative_coordinate < 0 ? derivative_coordinate
                                : derivative_coordinate + batch.atom_offsets[system] * 3;

  if (system_local < metric_elements) {
    const std::size_t first_aux = system_local / auxiliary_count;
    const std::size_t second_aux = system_local % auxiliary_count;
    const auto value = contracted_df<Derivative, true>(
        batch, static_cast<std::int32_t>(system),
        static_cast<std::int32_t>(orbital_count + first_aux),
        static_cast<std::int32_t>(dummy_index),
        static_cast<std::int32_t>(orbital_count + second_aux),
        static_cast<std::int32_t>(dummy_index), system_derivative_coordinate);
    metric[local_system * metric_elements + system_local] = value;
    return;
  }

  const std::size_t local = system_local - metric_elements;
  const std::size_t orbital_pair = local / auxiliary_count;
  const std::size_t auxiliary = local % auxiliary_count;
  const std::size_t first_orbital = orbital_pair / orbital_count;
  const std::size_t second_orbital = orbital_pair % orbital_count;
  const auto value = contracted_df<Derivative, false>(
      batch, static_cast<std::int32_t>(system), static_cast<std::int32_t>(first_orbital),
      static_cast<std::int32_t>(second_orbital),
      static_cast<std::int32_t>(orbital_count + auxiliary), static_cast<std::int32_t>(dummy_index),
      system_derivative_coordinate);
  three_center[local_system * three_center_elements + local] = value;
}

/**
 * Generate one public-basis three-center tile without materializing the raw
 * Cartesian tensor.  The source recurrence is evaluated directly for each
 * requested AO/auxiliary element and contracted with the already prepared
 * metric inverse square root.  This intentionally trades redundant arithmetic
 * for a strict O(pair_tile*aux_tile) device footprint in budgeted plans.
 */
template <bool Derivative>
__global__ void build_cuda_df_transformed_tile_kernel(
    DeviceBatch batch, std::size_t cartesian_orbital_count, std::size_t cartesian_auxiliary_count,
    std::size_t public_nbf, std::size_t public_naux, std::size_t dummy_index, std::size_t system,
    std::size_t pair_begin, std::size_t pair_count, std::size_t auxiliary_begin,
    std::size_t auxiliary_count, std::int64_t derivative_coordinate,
    const DfPublicAoExpansion* orbital_to_cartesian,
    const DfPublicAoExpansion* auxiliary_to_cartesian, const double* inverse_square_root,
    bool apply_metric_transform, double* output, unsigned mapping = 0U) {
  const unsigned lanes = !Derivative && mapping == 2U ? 32U : 1U;
  const unsigned lane = threadIdx.x % lanes;
  const std::size_t element =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / lanes;
  const std::size_t tile_elements = pair_count * auxiliary_count;
  if (element >= tile_elements) return;
  const bool components_contiguous = !Derivative && mapping == 1U;
  const std::size_t pair =
      pair_begin + (components_contiguous ? element % pair_count : element / auxiliary_count);
  const std::size_t auxiliary =
      auxiliary_begin + (components_contiguous ? element / pair_count : element % auxiliary_count);
  const std::size_t public_first = pair / public_nbf;
  const std::size_t public_second = pair % public_nbf;
  // The source stores transforms for every system contiguously.  A fleet can
  // legitimately mix different shell layouts while keeping the same public
  // AO dimensions, so never reuse system zero's transform for later items.
  const auto& first_expansion = orbital_to_cartesian[system * public_nbf + public_first];
  const auto& second_expansion = orbital_to_cartesian[system * public_nbf + public_second];
  const auto* system_auxiliary_to_cartesian = auxiliary_to_cartesian + system * public_naux;
  double value = 0.0;
  const std::size_t source_begin = apply_metric_transform ? 0U : auxiliary;
  const std::size_t source_end = apply_metric_transform ? public_naux : source_begin + 1U;
  // A transformed output reduces over both source auxiliaries and primitive
  // products. The generated schedule splits those independent extents so short
  // contractions do not leave most of the warp idle. Raw tiles have only one
  // source term and keep their full primitive-product partition. Every lane
  // still reaches the final output reduction, including ragged source tails.
  constexpr unsigned source_primitive_lanes =
      generated_df_policy::ValueSourceSchedule::primitive_lanes;
  static_assert(source_primitive_lanes && source_primitive_lanes <= 32U &&
                !(source_primitive_lanes & (source_primitive_lanes - 1U)));
  const unsigned primitive_lanes =
      lanes == 32U && apply_metric_transform ? source_primitive_lanes : lanes;
  const unsigned source_lanes = lanes / primitive_lanes;
  for (std::size_t source = source_begin + lane / primitive_lanes; source < source_end;
       source += source_lanes) {
    double transformed_raw = 0.0;
    const auto& auxiliary_expansion = system_auxiliary_to_cartesian[source];
    for (unsigned i = 0; i < first_expansion.count; ++i) {
      const auto first = first_expansion.cartesian[i];
      const double first_coefficient = first_expansion.coefficients[i];
      for (unsigned j = 0; j < second_expansion.count; ++j) {
        const auto second = second_expansion.cartesian[j];
        const double second_coefficient = second_expansion.coefficients[j];
        for (unsigned k = 0; k < auxiliary_expansion.count; ++k) {
          const auto cartesian_auxiliary = auxiliary_expansion.cartesian[k];
          const double auxiliary_coefficient = auxiliary_expansion.coefficients[k];
          const double raw = contracted_df<Derivative, false>(
              batch, static_cast<std::int32_t>(system), static_cast<std::int32_t>(first),
              static_cast<std::int32_t>(second),
              static_cast<std::int32_t>(cartesian_orbital_count + cartesian_auxiliary),
              static_cast<std::int32_t>(dummy_index), derivative_coordinate, lane % primitive_lanes,
              primitive_lanes);
          transformed_raw += first_coefficient * second_coefficient * auxiliary_coefficient * raw;
        }
      }
    }
    value += apply_metric_transform
                 ? transformed_raw * inverse_square_root[auxiliary * public_naux + source]
                 : transformed_raw;
  }
  if (lanes == 32U) {
    // Each warp owns a complete output, including ragged tails; no lane can
    // exit independently before this full-mask deterministic reduction.
    for (unsigned offset = 16U; offset != 0U; offset /= 2U) {
      value += __shfl_down_sync(0xffffffffU, value, offset);
    }
  }
  if (lane == 0U)
    output[(pair - pair_begin) * auxiliary_count + (auxiliary - auxiliary_begin)] = value;
}

/** Generate one public-basis auxiliary metric (or its derivative). */
template <bool Derivative>
__global__ void build_cuda_df_metric_source_kernel(
    DeviceBatch batch, std::size_t cartesian_orbital_count, std::size_t cartesian_auxiliary_count,
    std::size_t public_naux, std::size_t dummy_index, std::size_t system,
    std::size_t auxiliary_row_begin, std::size_t auxiliary_row_count,
    std::int64_t derivative_coordinate, const DfPublicAoExpansion* auxiliary_to_cartesian,
    double* output, unsigned mapping = 0U) {
  const unsigned lanes = !Derivative && mapping == 2U ? 32U : 1U;
  const unsigned lane = threadIdx.x % lanes;
  const std::size_t element =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / lanes;
  const std::size_t total = auxiliary_row_count * public_naux;
  if (element >= total) return;
  const bool components_contiguous = !Derivative && mapping == 1U;
  const std::size_t first =
      auxiliary_row_begin +
      (components_contiguous ? element % auxiliary_row_count : element / public_naux);
  const std::size_t second =
      components_contiguous ? element / auxiliary_row_count : element % public_naux;
  const auto& first_expansion = auxiliary_to_cartesian[system * public_naux + first];
  const auto& second_expansion = auxiliary_to_cartesian[system * public_naux + second];
  double value = 0.0;
  for (unsigned i = 0; i < first_expansion.count; ++i) {
    const auto cartesian_first = first_expansion.cartesian[i];
    const double first_coefficient = first_expansion.coefficients[i];
    for (unsigned j = 0; j < second_expansion.count; ++j) {
      const auto cartesian_second = second_expansion.cartesian[j];
      const double second_coefficient = second_expansion.coefficients[j];
      const double raw = contracted_df<Derivative, true>(
          batch, static_cast<std::int32_t>(system),
          static_cast<std::int32_t>(cartesian_orbital_count + cartesian_first),
          static_cast<std::int32_t>(dummy_index),
          static_cast<std::int32_t>(cartesian_orbital_count + cartesian_second),
          static_cast<std::int32_t>(dummy_index), derivative_coordinate, lane, lanes);
      value += first_coefficient * second_coefficient * raw;
    }
  }
  if (lanes == 32U) {
    for (unsigned offset = 16U; offset != 0U; offset /= 2U) {
      value += __shfl_down_sync(0xffffffffU, value, offset);
    }
  }
  if (lane == 0U) output[(first - auxiliary_row_begin) * public_naux + second] = value;
}

}  // namespace

void launch_build_cuda_df_integrals_kernel(
    bool derivative, dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    DeviceBatch batch, std::size_t orbital_count, std::size_t auxiliary_count,
    std::size_t dummy_index, std::size_t metric_elements, std::size_t three_center_elements,
    std::size_t system_base, std::size_t launch_batch_size, std::int64_t derivative_coordinate,
    double* metric, double* three_center) {
  if (derivative) {
    build_cuda_df_integrals_kernel<true><<<grid, block, shared_bytes, stream>>>(
        batch, orbital_count, auxiliary_count, dummy_index, metric_elements, three_center_elements,
        system_base, launch_batch_size, derivative_coordinate, metric, three_center);
  } else {
    build_cuda_df_integrals_kernel<false><<<grid, block, shared_bytes, stream>>>(
        batch, orbital_count, auxiliary_count, dummy_index, metric_elements, three_center_elements,
        system_base, launch_batch_size, derivative_coordinate, metric, three_center);
  }
}

void launch_build_cuda_df_metric_source_kernel(
    bool derivative, dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    DeviceBatch batch, std::size_t cartesian_orbital_count, std::size_t cartesian_auxiliary_count,
    std::size_t public_naux, std::size_t dummy_index, std::size_t system,
    std::size_t auxiliary_row_begin, std::size_t auxiliary_row_count,
    std::int64_t derivative_coordinate, const DfPublicAoExpansion* auxiliary_to_cartesian,
    double* output, unsigned mapping) {
  if (derivative) {
    build_cuda_df_metric_source_kernel<true><<<grid, block, shared_bytes, stream>>>(
        batch, cartesian_orbital_count, cartesian_auxiliary_count, public_naux, dummy_index, system,
        auxiliary_row_begin, auxiliary_row_count, derivative_coordinate, auxiliary_to_cartesian,
        output, mapping);
  } else {
    build_cuda_df_metric_source_kernel<false><<<grid, block, shared_bytes, stream>>>(
        batch, cartesian_orbital_count, cartesian_auxiliary_count, public_naux, dummy_index, system,
        auxiliary_row_begin, auxiliary_row_count, derivative_coordinate, auxiliary_to_cartesian,
        output, mapping);
  }
}

void launch_build_cuda_df_transformed_tile_kernel(
    bool derivative, dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    DeviceBatch batch, std::size_t cartesian_orbital_count, std::size_t cartesian_auxiliary_count,
    std::size_t public_nbf, std::size_t public_naux, std::size_t dummy_index, std::size_t system,
    std::size_t pair_begin, std::size_t pair_count, std::size_t auxiliary_begin,
    std::size_t auxiliary_count, std::int64_t derivative_coordinate,
    const DfPublicAoExpansion* orbital_to_cartesian,
    const DfPublicAoExpansion* auxiliary_to_cartesian, const double* inverse_square_root,
    bool apply_metric_transform, double* output, unsigned mapping) {
  if (derivative) {
    build_cuda_df_transformed_tile_kernel<true><<<grid, block, shared_bytes, stream>>>(
        batch, cartesian_orbital_count, cartesian_auxiliary_count, public_nbf, public_naux,
        dummy_index, system, pair_begin, pair_count, auxiliary_begin, auxiliary_count,
        derivative_coordinate, orbital_to_cartesian, auxiliary_to_cartesian, inverse_square_root,
        apply_metric_transform, output, mapping);
  } else {
    build_cuda_df_transformed_tile_kernel<false><<<grid, block, shared_bytes, stream>>>(
        batch, cartesian_orbital_count, cartesian_auxiliary_count, public_nbf, public_naux,
        dummy_index, system, pair_begin, pair_count, auxiliary_begin, auxiliary_count,
        derivative_coordinate, orbital_to_cartesian, auxiliary_to_cartesian, inverse_square_root,
        apply_metric_transform, output, mapping);
  }
}

}  // namespace vibeqc::scf::cuda_execution
