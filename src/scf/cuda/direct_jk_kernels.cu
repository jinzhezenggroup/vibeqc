#include <algorithm>
#include <cmath>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>

#include "generated_direct_contraction.cuh"
#include "generated_direct_fock_accumulation.cuh"
#include "generated_direct_source_contraction.cuh"
#include "scf/cuda/direct_bounded_fallback.hpp"
#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_eri_symmetry.cuh"
#include "scf/cuda/direct_jk_kernels.hpp"
#include "scf/cuda/direct_queue_index.cuh"

namespace generativeqc::scf {
namespace {
using namespace cuda_execution;
}

namespace {

generativeqc::integrals::CoulombRange integral_range(DirectCoulombRange range) {
  switch (range) {
    case DirectCoulombRange::Full:
      return generativeqc::integrals::CoulombRange::Full;
    case DirectCoulombRange::Long:
      return generativeqc::integrals::CoulombRange::Long;
    case DirectCoulombRange::Short:
      return generativeqc::integrals::CoulombRange::Short;
  }
  return generativeqc::integrals::CoulombRange::Full;
}

__global__ void independent_jk_finite_kernel(const double* values, std::size_t count,
                                             int* failure) {
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    if (!isfinite(values[i])) atomicExch(failure, 1);
}

/** Map screened flat work to a row without traversing rejected quartets.
 * Inclusive prefixes may contain empty rows, so search for the first > work. */
__device__ void canonical_pair_indices(std::size_t work, std::size_t first_count,
                                       std::size_t second_count, bool same_bucket,
                                       CanonicalPairRows rows, std::size_t& first,
                                       std::size_t& second) {
  if (rows.prefix) {
    std::size_t begin = 0, end = first_count;
    while (begin < end) {
      const auto middle = begin + (end - begin) / 2U;
      if (rows.prefix[middle] <= work)
        begin = middle + 1U;
      else
        end = middle;
    }
    first = begin;
    second = work - (first ? rows.prefix[first - 1U] : 0U);
  } else if (same_bucket) {
    decode_lower_triangle(work, first, second);
  } else {
    first = work / second_count;
    second = work % second_count;
  }
}

__global__ void canonical_pair_keys_kernel(DeviceBatch batch, const std::int32_t* pairs,
                                           const double* bounds, int count, double* keys,
                                           std::int32_t* order) {
  const std::size_t dimension = static_cast<std::size_t>(batch.nbf);
  const std::size_t pairs_per_item = dimension * (dimension + 1U) / 2U;
  for (std::size_t pair = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       pair < static_cast<std::size_t>(count);
       pair += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto item = pair / pairs_per_item;
    keys[pair] =
        bounds[item * dimension * dimension + pairs[2U * pair] * dimension + pairs[2U * pair + 1U]];
    order[pair] = static_cast<std::int32_t>(pair);
  }
}

__global__ void canonical_pair_rows_kernel(const double* keys, std::size_t first_begin,
                                           std::size_t first_count, std::size_t second_begin,
                                           std::size_t second_count, bool same_bucket,
                                           double screening, std::uint64_t* counts) {
  for (std::size_t row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       row < first_count; row += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    std::size_t begin = 0, end = same_bucket ? row + 1U : second_count;
    const double bra = keys[first_begin + row];
    while (begin < end) {
      const auto middle = begin + (end - begin) / 2U;
      // Division by bra would change FP64 rounding at the screening boundary.
      if (bra * keys[second_begin + middle] < screening)
        end = middle;
      else
        begin = middle + 1U;
    }
    counts[row] = begin;
  }
}

/** Select compiler-owned public or Cartesian sources without new recurrence algebra. */
template <unsigned AngularOrder, bool Cartesian, typename Scalar>
__device__ Scalar canonical_quartet(DeviceBatch batch, std::int32_t system, std::int32_t first,
                                    std::int32_t second, std::int32_t third, std::int32_t fourth,
                                    std::int64_t coordinate,
                                    generativeqc::integrals::CoulombRange range, double omega) {
  if constexpr (Cartesian) {
    const auto base = static_cast<std::size_t>(system) * batch.direct_nbf;
    const auto shell_class = direct_quartet_shell_class_device(
        batch.shell_angular[batch.direct_ao_shells[base + first]],
        batch.shell_angular[batch.direct_ao_shells[base + second]],
        batch.shell_angular[batch.direct_ao_shells[base + third]],
        batch.shell_angular[batch.direct_ao_shells[base + fourth]]);
    return dispatch_contracted_eri_cartesian_source_shell_class<AngularOrder, Scalar>(
        shell_class, batch, system, first, second, third, fourth, coordinate, range, omega);
  } else if constexpr (AngularOrder <= 12U) {
    return contracted_eri_order<AngularOrder, Scalar>(batch, system, first, second, third, fourth,
                                                      coordinate, range, omega);
  } else {
    return contracted_eri<Scalar>(batch, system, first, second, third, fourth, coordinate, range,
                                  omega);
  }
}

/** Reuse the scientific compiler's exact contraction and permutation scatter.
 * Bucket homogeneity removes the runtime 0..12 recurrence dispatch from the
 * inner ERI loop. Triangular work enumerates each physical ERI only once. */
template <unsigned AngularOrder, bool Unrestricted, bool Cartesian>
__global__ void canonical_jk_kernel(
    DeviceBatch batch, std::int32_t system, const std::int32_t* pairs, CanonicalPairRows rows,
    std::size_t first_begin, std::size_t first_count, std::size_t second_begin,
    std::size_t second_count, bool same_bucket, std::size_t work_count, bool want_j, bool want_k,
    generativeqc::integrals::CoulombRange exchange_range, double exchange_omega, double screening,
    const double* bounds, const double* density, double* coulomb, double* exchange,
    std::uint64_t* work_census) {
  const std::size_t dimension = static_cast<std::size_t>(batch.nbf);
  const std::size_t matrix = dimension * dimension;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix;
  const std::size_t spin_offset = static_cast<std::size_t>(system) * 2U * matrix;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  if (rows.prefix) work_count = rows.prefix[first_count - 1U];
  unsigned long long candidates = 0, evaluated = 0;
  for (std::size_t work = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       work < work_count; work += stride) {
    std::size_t first_local{}, second_local{};
    canonical_pair_indices(work, first_count, second_count, same_bucket, rows, first_local,
                           second_local);
    if (first_local >= first_count) continue;
    if (work_census) ++candidates;
    const std::size_t first_pair =
        rows.order ? rows.order[first_begin + first_local] : first_begin + first_local;
    const std::size_t second_pair =
        rows.order ? rows.order[second_begin + second_local] : second_begin + second_local;
    const auto first = pairs[2U * first_pair];
    const auto second = pairs[2U * first_pair + 1U];
    const auto third = pairs[2U * second_pair];
    const auto fourth = pairs[2U * second_pair + 1U];
    if (bounds[physical_offset + first * dimension + second] *
            bounds[physical_offset + third * dimension + fourth] <
        screening)
      continue;
    double full_value = 0.0;
    if (want_j || (want_k && exchange_range == generativeqc::integrals::CoulombRange::Full)) {
      full_value = canonical_quartet<AngularOrder, Cartesian, double>(
          batch, system, first, second, third, fourth, -1,
          generativeqc::integrals::CoulombRange::Full, 0.0);
      if (work_census) ++evaluated;
    }
    if (want_j)
      accumulate_direct_fock_integral<Unrestricted>(dimension, physical_offset, spin_offset,
                                                    density, coulomb, first, second, third, fourth,
                                                    full_value, true, false);
    if (want_k) {
      const double exchange_value = exchange_range == generativeqc::integrals::CoulombRange::Full
                                        ? full_value
                                        : canonical_quartet<AngularOrder, Cartesian, double>(
                                              batch, system, first, second, third, fourth, -1,
                                              exchange_range, exchange_omega);
      if (work_census && exchange_range != generativeqc::integrals::CoulombRange::Full) ++evaluated;
      accumulate_direct_fock_integral<Unrestricted>(dimension, physical_offset, spin_offset,
                                                    density, exchange, first, second, third, fourth,
                                                    exchange_value, false, true);
    }
  }
  if (work_census) {
    for (unsigned offset = warpSize / 2; offset; offset /= 2) {
      candidates += __shfl_down_sync(0xffffffffU, candidates, offset);
      evaluated += __shfl_down_sync(0xffffffffU, evaluated, offset);
    }
    if (threadIdx.x % warpSize == 0) {
      atomicAdd(reinterpret_cast<unsigned long long*>(work_census), candidates);
      atomicAdd(reinterpret_cast<unsigned long long*>(work_census + 1U), evaluated);
    }
  }
}

__global__ void independent_eri_tile_kernel(DeviceBatch batch, std::int32_t system, std::size_t b0,
                                            std::size_t b1, std::size_t b2, std::size_t b3,
                                            std::size_t c0, std::size_t c1, std::size_t c2,
                                            std::size_t c3, std::size_t elements, double* eri) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       element < elements; element += stride) {
    std::size_t local = element;
    const auto l = static_cast<std::int32_t>(b3 + local % c3);
    local /= c3;
    const auto k = static_cast<std::int32_t>(b2 + local % c2);
    local /= c2;
    const auto j = static_cast<std::int32_t>(b1 + local % c1);
    local /= c1;
    const auto i = static_cast<std::int32_t>(b0 + local);
    eri[element] = contracted_eri<double>(batch, system, i, j, k, l, -1);
  }
}

/** Schwarz bounds in public AO order, including sparse spherical expansions. */
template <bool Cartesian>
__global__ void independent_jk_bounds_kernel(DeviceBatch batch, double* bounds, int* failure) {
  const std::size_t n = batch.nbf, matrix = n * n;
  const std::size_t item = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (item >= static_cast<std::size_t>(batch.batch_size) * matrix) return;
  const auto system = static_cast<std::int32_t>(item / matrix);
  const auto i = static_cast<std::int32_t>((item % matrix) / n);
  const auto j = static_cast<std::int32_t>(item % n);
  const double value = Cartesian
                           ? contracted_eri_cartesian_source<double>(batch, system, i, j, i, j, -1)
                           : contracted_eri<double>(batch, system, i, j, i, j, -1);
  // NaN bounds must never masquerade as screened-out quartets.
  if (!isfinite(value)) atomicExch(failure, 1);
  bounds[item] = sqrt(fabs(value));
}

/** One output owner reduces all density pairs; no ERI tensor or atomics.
 * Full pair traversal preserves nonsymmetric input orientation. Integral and
 * sparse spherical expansion arithmetic is exactly the existing evaluator.
 */
template <bool MixedJ>
__global__ void independent_jk_kernel(DeviceBatch batch, std::size_t system_begin, bool want_j,
                                      bool want_k, bool unrestricted,
                                      generativeqc::integrals::CoulombRange exchange_range,
                                      double exchange_omega, double screening, const double* bounds,
                                      const double* density, const double* beta, double* j_out,
                                      double* ka_out, double* kb_out,
                                      std::uint64_t* mixed_coulomb_work_count) {
  __shared__ double sums[3][kIndependentJkThreads];
  const std::size_t n = batch.nbf, matrix = n * n;
  const std::size_t item = system_begin * matrix + blockIdx.x;
  const auto system = static_cast<std::int32_t>(item / matrix);
  const std::size_t offset = static_cast<std::size_t>(system) * matrix;
  const auto i = static_cast<std::int32_t>((item % matrix) / n);
  const auto j = static_cast<std::int32_t>(item % n);
  double coulomb = 0.0, alpha_exchange = 0.0, beta_exchange = 0.0;
  unsigned long long mixed_coulomb_work = 0;
  for (std::size_t kl = threadIdx.x; kl < matrix; kl += blockDim.x) {
    const auto k = static_cast<std::int32_t>(kl / n), l = static_cast<std::int32_t>(kl % n);
    const double a = density[offset + kl], b = unrestricted ? beta[offset + kl] : 0.0;
    if (want_j && bounds[item] * bounds[offset + kl] >= screening && a + b != 0.0) {
      const double value =
          MixedJ ? scalar_value(contracted_eri<MixedPrecisionFloat>(batch, system, i, j, k, l, -1))
                 : contracted_eri<double>(batch, system, i, j, k, l, -1);
      coulomb += (a + b) * value;
      if constexpr (MixedJ) ++mixed_coulomb_work;
    }
    if (want_k && bounds[offset + i * n + k] * bounds[offset + j * n + l] >= screening &&
        (a != 0.0 || b != 0.0)) {
      const double value =
          contracted_eri<double>(batch, system, i, k, j, l, -1, exchange_range, exchange_omega);
      alpha_exchange += a * value;
      beta_exchange += b * value;
    }
  }
  if constexpr (MixedJ) {
    if (mixed_coulomb_work_count) {
      for (unsigned offset = warpSize / 2; offset; offset /= 2)
        mixed_coulomb_work += __shfl_down_sync(0xffffffffU, mixed_coulomb_work, offset);
      if (threadIdx.x == 0)
        atomicAdd(reinterpret_cast<unsigned long long*>(mixed_coulomb_work_count),
                  mixed_coulomb_work);
    }
  }
  sums[0][threadIdx.x] = coulomb;
  sums[1][threadIdx.x] = alpha_exchange;
  sums[2][threadIdx.x] = beta_exchange;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
    if (threadIdx.x < stride)
      for (unsigned term = 0; term < 3; ++term)
        sums[term][threadIdx.x] += sums[term][threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if (want_j) j_out[item] = sums[0][0];
    if (want_k) ka_out[item] = sums[1][0];
    if (want_k && unrestricted) kb_out[item] = sums[2][0];
  }
}

/** Differentiate the same screened discrete energy at fixed spin densities.
 *
 * Traverse each ordered AO quartet once, then differentiate only the unique
 * nuclear centers that actually occur in that quartet. Dual3 carries x/y/z
 * together and translational invariance reconstructs the final center. This
 * removes the previous coordinate-by-AO^4 scan without changing screening,
 * public-AO spherical expansion, coefficients, or radial operators.
 */
__global__ void independent_jk_derivative_kernel(
    DeviceBatch batch, std::size_t system_begin, std::size_t system_count, double cj, double ck,
    bool unrestricted, generativeqc::integrals::CoulombRange exchange_range, double exchange_omega,
    double screening, const double* bounds, const double* density, const double* beta,
    double* out) {
  const std::size_t n = batch.nbf, matrix = n * n, quartets = matrix * matrix;
  const std::size_t work_count = system_count * quartets;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t work = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       work < work_count; work += stride) {
    const std::size_t local_system = work / quartets;
    const auto system = static_cast<std::int32_t>(system_begin + local_system);
    const std::size_t quartet = work % quartets;
    const std::size_t offset = static_cast<std::size_t>(system) * matrix;
    const std::size_t ij = quartet / matrix, kl = quartet % matrix;
    if (bounds[offset + ij] * bounds[offset + kl] < screening) continue;

    const auto i = static_cast<std::int32_t>(ij / n), j = static_cast<std::int32_t>(ij % n);
    const auto k = static_cast<std::int32_t>(kl / n), l = static_cast<std::int32_t>(kl % n);
    double full_weight = 0.0, range_weight = 0.0;
    // An absent/zero-weight term must not evaluate a quadratic that can
    // overflow, even when the requested total-density contribution is finite.
    if (cj != 0.0) {
      const double total_ij = density[offset + ij] + (unrestricted ? beta[offset + ij] : 0.0);
      const double total_kl = density[offset + kl] + (unrestricted ? beta[offset + kl] : 0.0);
      full_weight += 0.5 * cj * total_ij * total_kl;
    }
    if (ck != 0.0) {
      const std::size_t ik = static_cast<std::size_t>(i) * n + k;
      const std::size_t jl = static_cast<std::size_t>(j) * n + l;
      const double exchange = density[offset + ik] * density[offset + jl] +
                              (unrestricted ? beta[offset + ik] * beta[offset + jl] : 0.0);
      if (exchange_range == generativeqc::integrals::CoulombRange::Full)
        full_weight += 0.5 * ck * exchange;
      else
        range_weight = 0.5 * ck * exchange;
    }
    if (full_weight == 0.0 && range_weight == 0.0) continue;

    const std::size_t base = static_cast<std::size_t>(system) * n;
    const std::int32_t center_atoms[4] = {
        batch.shell_atoms[batch.ao_shells[base + static_cast<std::size_t>(i)]],
        batch.shell_atoms[batch.ao_shells[base + static_cast<std::size_t>(j)]],
        batch.shell_atoms[batch.ao_shells[base + static_cast<std::size_t>(k)]],
        batch.shell_atoms[batch.ao_shells[base + static_cast<std::size_t>(l)]],
    };
    std::int32_t unique_atoms[4];
    unsigned unique_count = 0;
    for (unsigned center = 0; center < 4; ++center) {
      bool duplicate = false;
      for (unsigned previous = 0; previous < unique_count; ++previous)
        duplicate = duplicate || unique_atoms[previous] == center_atoms[center];
      if (!duplicate) unique_atoms[unique_count++] = center_atoms[center];
    }
    if (unique_count <= 1) continue;

    double reconstructed[3]{};
    for (unsigned center = 0; center + 1 < unique_count; ++center) {
      const std::int64_t coordinate = static_cast<std::int64_t>(unique_atoms[center]) * 3;
      double derivative[3]{};
      if (full_weight != 0.0) {
        const Dual3 value = contracted_eri<Dual3>(batch, system, i, j, k, l, coordinate);
        derivative[0] += full_weight * value.derivative_x;
        derivative[1] += full_weight * value.derivative_y;
        derivative[2] += full_weight * value.derivative_z;
      }
      if (range_weight != 0.0) {
        const Dual3 value = contracted_eri<Dual3>(batch, system, i, j, k, l, coordinate,
                                                  exchange_range, exchange_omega);
        derivative[0] += range_weight * value.derivative_x;
        derivative[1] += range_weight * value.derivative_y;
        derivative[2] += range_weight * value.derivative_z;
      }
      for (unsigned axis = 0; axis < 3; ++axis) {
        reconstructed[axis] += derivative[axis];
        if (derivative[axis] != 0.0)
          atomicAdd(out + static_cast<std::size_t>(coordinate) + axis, derivative[axis]);
      }
    }
    const std::size_t final_coordinate =
        static_cast<std::size_t>(unique_atoms[unique_count - 1]) * 3;
    for (unsigned axis = 0; axis < 3; ++axis)
      if (reconstructed[axis] != 0.0)
        atomicAdd(out + final_coordinate + axis, -reconstructed[axis]);
  }
}

/** One RSH force pass over symmetry-unique public-AO quartets.
 *
 * J uses the full Coulomb derivative, while SR/LR K share
 * Full = Short + Long. Symmetric final-state densities let the eight ERI
 * permutations contribute through one density coefficient and one canonical
 * integral derivative.
 */
template <unsigned AngularOrder, bool Cartesian>
__device__ Dual3 rsh_quartet_derivative(DeviceBatch batch, std::int32_t system, std::size_t first,
                                        std::size_t second, std::size_t third, std::size_t fourth,
                                        std::int64_t coordinate,
                                        generativeqc::integrals::CoulombRange range, double omega) {
  return canonical_quartet<AngularOrder, Cartesian, Dual3>(batch, system, first, second, third,
                                                           fourth, coordinate, range, omega);
}

/** Shared final-density algebra for dense and screened RSH schedules.
 * The sentinel order retains the compatibility evaluator; homogeneous buckets
 * specialize only the compiler's recurrence order, never the force weights. */
template <unsigned AngularOrder, bool Cartesian = false>
__device__ unsigned contract_rsh_quartet(DeviceBatch batch, std::int32_t system, std::size_t i,
                                         std::size_t j, std::size_t k, std::size_t l,
                                         std::size_t source_stride, double cj, double short_ck,
                                         double long_ck, bool unrestricted, double omega,
                                         const double* density, const double* beta, double* out) {
  const std::size_t n = batch.nbf, matrix = n * n;
  const std::size_t offset = static_cast<std::size_t>(system) * matrix;
  unsigned evaluated = 0;
  double j_weight = 0.0, exchange_weight = 0.0;
  for (unsigned permutation = 0; permutation < 8; ++permutation) {
    if (!unique_eri_symmetry_permutation(permutation, i, j, k, l)) continue;
    std::size_t a = 0, b = 0, cc = 0, d = 0;
    eri_symmetry_permutation(permutation, i, j, k, l, a, b, cc, d);
    // Do not form unused total-density sums or spin quadratics: finite
    // spin inputs can overflow those intermediates even when every
    // requested source is finite (for example, cancellation in pure J).
    if (cj != 0.0) {
      const std::size_t ab = a * n + b, cd = cc * n + d;
      const double total_ab = density[offset + ab] + (unrestricted ? beta[offset + ab] : 0.0);
      const double total_cd = density[offset + cd] + (unrestricted ? beta[offset + cd] : 0.0);
      j_weight += 0.5 * cj * total_ab * total_cd;
    }
    if (short_ck != 0.0 || long_ck != 0.0) {
      const std::size_t ac = a * n + cc, bd = b * n + d;
      exchange_weight += 0.5 * (density[offset + ac] * density[offset + bd] +
                                (unrestricted ? beta[offset + ac] * beta[offset + bd] : 0.0));
    }
  }
  const double short_weight = short_ck != 0.0 ? short_ck * exchange_weight : 0.0;
  const double long_weight = long_ck != 0.0 ? long_ck * exchange_weight : 0.0;
  if (j_weight == 0.0 && short_weight == 0.0 && long_weight == 0.0) return 0;

  const std::size_t base = static_cast<std::size_t>(system) * n;
  const std::int32_t center_atoms[4] = {
      batch.shell_atoms[batch.ao_shells[base + i]],
      batch.shell_atoms[batch.ao_shells[base + j]],
      batch.shell_atoms[batch.ao_shells[base + k]],
      batch.shell_atoms[batch.ao_shells[base + l]],
  };
  std::int32_t unique_atoms[4];
  unsigned unique_count = 0;
  for (unsigned center = 0; center < 4; ++center) {
    bool duplicate = false;
    for (unsigned previous = 0; previous < unique_count; ++previous)
      duplicate = duplicate || unique_atoms[previous] == center_atoms[center];
    if (!duplicate) unique_atoms[unique_count++] = center_atoms[center];
  }
  if (unique_count <= 1) return 0;

  double reconstructed[3][3]{};
  for (unsigned center = 0; center + 1 < unique_count; ++center) {
    const std::int64_t coordinate = static_cast<std::int64_t>(unique_atoms[center]) * 3;
    double full[3]{}, long_range[3]{};
    if (j_weight != 0.0 || short_weight != 0.0) {
      const Dual3 value = rsh_quartet_derivative<AngularOrder, Cartesian>(
          batch, system, i, j, k, l, coordinate, generativeqc::integrals::CoulombRange::Full, 0.0);
      ++evaluated;
      full[0] = value.derivative_x;
      full[1] = value.derivative_y;
      full[2] = value.derivative_z;
    }
    if (short_weight != 0.0 || long_weight != 0.0) {
      const Dual3 value = rsh_quartet_derivative<AngularOrder, Cartesian>(
          batch, system, i, j, k, l, coordinate, generativeqc::integrals::CoulombRange::Long,
          omega);
      ++evaluated;
      long_range[0] = value.derivative_x;
      long_range[1] = value.derivative_y;
      long_range[2] = value.derivative_z;
    }
    for (unsigned axis = 0; axis < 3; ++axis) {
      const double source[3] = {
          j_weight * full[axis],
          short_weight * (full[axis] - long_range[axis]),
          long_weight * long_range[axis],
      };
      for (unsigned term = 0; term < 3; ++term) {
        reconstructed[term][axis] += source[term];
        if (source[term] != 0.0)
          atomicAdd(out + term * source_stride + static_cast<std::size_t>(coordinate) + axis,
                    source[term]);
      }
    }
  }
  const std::size_t final_coordinate = static_cast<std::size_t>(unique_atoms[unique_count - 1]) * 3;
  for (unsigned term = 0; term < 3; ++term)
    for (unsigned axis = 0; axis < 3; ++axis)
      if (reconstructed[term][axis] != 0.0)
        atomicAdd(out + term * source_stride + final_coordinate + axis, -reconstructed[term][axis]);
  return evaluated;
}

__global__ void independent_rsh_derivative_kernel(DeviceBatch batch, std::size_t system_begin,
                                                  std::size_t system_count,
                                                  std::size_t source_stride, double cj,
                                                  double short_ck, double long_ck,
                                                  bool unrestricted, double omega, double screening,
                                                  const double* bounds, const double* density,
                                                  const double* beta, double* out) {
  const std::size_t n = batch.nbf, matrix = n * n;
  const std::size_t pair_count = n * (n + 1) / 2;
  const std::size_t unique_quartets = pair_count * (pair_count + 1) / 2;
  const std::size_t work_count = system_count * unique_quartets;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t work = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       work < work_count; work += stride) {
    const auto system = static_cast<std::int32_t>(system_begin + work / unique_quartets);
    std::size_t first_pair{}, second_pair{}, first{}, second{}, third{}, fourth{};
    decode_lower_triangle(work % unique_quartets, first_pair, second_pair);
    decode_lower_triangle(first_pair, first, second);
    decode_lower_triangle(second_pair, third, fourth);
    const auto offset = static_cast<std::size_t>(system) * matrix;
    if (bounds[offset + first * n + second] * bounds[offset + third * n + fourth] < screening)
      continue;
    contract_rsh_quartet<13U>(batch, system, first, second, third, fourth, source_stride, cj,
                              short_ck, long_ck, unrestricted, omega, density, beta, out);
  }
}

template <unsigned AngularOrder, bool Cartesian>
__global__ void canonical_rsh_derivative_kernel(
    DeviceBatch batch, std::int32_t system, const std::int32_t* pairs, CanonicalPairRows rows,
    std::size_t first_begin, std::size_t first_count, std::size_t second_begin,
    std::size_t second_count, bool same_bucket, std::size_t work_count, std::size_t source_stride,
    double cj, double short_ck, double long_ck, bool unrestricted, double omega, double screening,
    const double* bounds, const double* density, const double* beta, double* out,
    std::uint64_t* work_census) {
  const std::size_t dimension = batch.nbf;
  const auto offset = static_cast<std::size_t>(system) * dimension * dimension;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  if (rows.prefix) work_count = rows.prefix[first_count - 1U];
  unsigned long long candidates = 0, evaluated = 0;
  for (std::size_t work = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       work < work_count; work += stride) {
    std::size_t first_local{}, second_local{};
    canonical_pair_indices(work, first_count, second_count, same_bucket, rows, first_local,
                           second_local);
    const auto first_pair =
        rows.order ? rows.order[first_begin + first_local] : first_begin + first_local;
    const auto second_pair =
        rows.order ? rows.order[second_begin + second_local] : second_begin + second_local;
    const auto first = pairs[2U * first_pair], second = pairs[2U * first_pair + 1U];
    const auto third = pairs[2U * second_pair], fourth = pairs[2U * second_pair + 1U];
    if (work_census) ++candidates;
    if (bounds[offset + first * dimension + second] * bounds[offset + third * dimension + fourth] <
        screening)
      continue;
    const auto count = contract_rsh_quartet<AngularOrder, Cartesian>(
        batch, system, first, second, third, fourth, source_stride, cj, short_ck, long_ck,
        unrestricted, omega, density, beta, out);
    if (work_census) evaluated += count;
  }
  if (work_census) {
    for (unsigned offset = warpSize / 2; offset; offset /= 2) {
      candidates += __shfl_down_sync(0xffffffffU, candidates, offset);
      evaluated += __shfl_down_sync(0xffffffffU, evaluated, offset);
    }
    if (threadIdx.x % warpSize == 0) {
      atomicAdd(reinterpret_cast<unsigned long long*>(work_census), candidates);
      atomicAdd(reinterpret_cast<unsigned long long*>(work_census + 1U), evaluated);
    }
  }
}

}  // namespace

namespace cuda_execution {

cudaError_t canonical_pair_workspace(int pairs, int segments, int largest_bucket,
                                     std::size_t& bytes) {
  std::size_t sort_bytes = 0, scan_bytes = 0;
  auto status = cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr, sort_bytes, static_cast<double*>(nullptr), static_cast<double*>(nullptr),
      static_cast<std::int32_t*>(nullptr), static_cast<std::int32_t*>(nullptr), pairs, segments,
      static_cast<int*>(nullptr), static_cast<int*>(nullptr));
  if (status != cudaSuccess) return status;
  status = cub::DeviceScan::InclusiveSum(nullptr, scan_bytes, static_cast<std::uint64_t*>(nullptr),
                                         static_cast<std::uint64_t*>(nullptr), largest_bucket);
  bytes = std::max(sort_bytes, scan_bytes);
  return status;
}

cudaError_t prepare_canonical_pair_order(cudaStream_t stream, DeviceBatch batch,
                                         const std::int32_t* pairs, const double* bounds,
                                         int pair_count, int segment_count,
                                         const int* segment_offsets, double* input_keys,
                                         std::int32_t* input_order, double* sorted_keys,
                                         std::int32_t* sorted_order, void* workspace,
                                         std::size_t workspace_bytes) {
  const auto blocks =
      static_cast<unsigned>(std::min<std::size_t>((pair_count + 127U) / 128U, 4096U));
  canonical_pair_keys_kernel<<<blocks, 128, 0, stream>>>(batch, pairs, bounds, pair_count,
                                                         input_keys, input_order);
  const auto status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  return cub::DeviceSegmentedRadixSort::SortPairsDescending(
      workspace, workspace_bytes, input_keys, sorted_keys, input_order, sorted_order, pair_count,
      segment_count, segment_offsets, segment_offsets + 1, 0, 64, stream);
}

cudaError_t prepare_canonical_pair_rows(cudaStream_t stream, const double* sorted_keys,
                                        std::size_t first_begin, std::size_t first_count,
                                        std::size_t second_begin, std::size_t second_count,
                                        bool same_bucket, double screening, std::uint64_t* prefix,
                                        void* workspace, std::size_t workspace_bytes) {
  if (!first_count) return cudaSuccess;
  const auto blocks =
      static_cast<unsigned>(std::min<std::size_t>((first_count + 127U) / 128U, 4096U));
  canonical_pair_rows_kernel<<<blocks, 128, 0, stream>>>(sorted_keys, first_begin, first_count,
                                                         second_begin, second_count, same_bucket,
                                                         screening, prefix);
  const auto status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  return cub::DeviceScan::InclusiveSum(workspace, workspace_bytes, prefix, prefix,
                                       static_cast<int>(first_count), stream);
}

void launch_independent_jk_finite_kernel(cudaStream_t stream, const double* values,
                                         std::size_t count, int* failure) {
  const unsigned blocks = static_cast<unsigned>(std::min<std::size_t>((count + 127) / 128, 65535));
  independent_jk_finite_kernel<<<blocks, 128, 0, stream>>>(values, count, failure);
}

void launch_independent_eri_tile(cudaStream_t stream, DeviceBatch batch, std::int32_t system,
                                 const std::array<std::size_t, 4>& begin,
                                 const std::array<std::size_t, 4>& count, std::size_t elements,
                                 double* eri) {
  if (!elements) return;
  constexpr unsigned threads = 128;
  const unsigned blocks =
      static_cast<unsigned>(std::min<std::size_t>((elements + threads - 1) / threads, 65535));
  independent_eri_tile_kernel<<<blocks, threads, 0, stream>>>(
      batch, system, begin[0], begin[1], begin[2], begin[3], count[0], count[1], count[2], count[3],
      elements, eri);
}

void launch_independent_jk_bounds_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                         cudaStream_t stream, DeviceBatch batch, double* bounds,
                                         int* failure, bool cartesian) {
  if (cartesian)
    independent_jk_bounds_kernel<true>
        <<<grid, block, shared_bytes, stream>>>(batch, bounds, failure);
  else
    independent_jk_bounds_kernel<false>
        <<<grid, block, shared_bytes, stream>>>(batch, bounds, failure);
}

void launch_independent_jk_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                  cudaStream_t stream, DeviceBatch batch, std::size_t system_begin,
                                  bool want_j, bool want_k, bool unrestricted, bool mixed_j,
                                  DirectCoulombRange exchange_range, double exchange_omega,
                                  double screening, const double* bounds, const double* density,
                                  const double* beta, double* j_out, double* ka_out, double* kb_out,
                                  std::uint64_t* mixed_coulomb_work_count) {
  if (mixed_j)
    independent_jk_kernel<true><<<grid, block, shared_bytes, stream>>>(
        batch, system_begin, want_j, want_k, unrestricted, integral_range(exchange_range),
        exchange_omega, screening, bounds, density, beta, j_out, ka_out, kb_out,
        mixed_coulomb_work_count);
  else
    independent_jk_kernel<false><<<grid, block, shared_bytes, stream>>>(
        batch, system_begin, want_j, want_k, unrestricted, integral_range(exchange_range),
        exchange_omega, screening, bounds, density, beta, j_out, ka_out, kb_out, nullptr);
}

template <bool Cartesian>
void launch_canonical_jk_source(cudaStream_t stream, DeviceBatch batch, std::int32_t system,
                                unsigned angular_order, const std::int32_t* pairs,
                                CanonicalPairRows rows, std::size_t first_begin,
                                std::size_t first_count, std::size_t second_begin,
                                std::size_t second_count, bool same_bucket, bool want_j,
                                bool want_k, bool unrestricted, DirectCoulombRange exchange_range,
                                double exchange_omega, double screening, const double* bounds,
                                const double* density, double* coulomb, double* exchange,
                                std::uint64_t* work_census) {
  const std::size_t work_count =
      same_bucket ? first_count * (first_count + 1U) / 2U : first_count * second_count;
  if (!work_count) return;
  constexpr unsigned threads = 64U;
  const auto blocks =
      static_cast<unsigned>(std::min<std::size_t>((work_count + threads - 1U) / threads, 4096U));
#define GENERATIVEQC_CANONICAL_JK_ORDER(order)                                                     \
  case order:                                                                                      \
    if (unrestricted)                                                                              \
      canonical_jk_kernel<order, true, Cartesian><<<blocks, threads, 0, stream>>>(                 \
          batch, system, pairs, rows, first_begin, first_count, second_begin, second_count,        \
          same_bucket, work_count, want_j, want_k, integral_range(exchange_range), exchange_omega, \
          screening, bounds, density, coulomb, exchange, work_census);                             \
    else                                                                                           \
      canonical_jk_kernel<order, false, Cartesian><<<blocks, threads, 0, stream>>>(                \
          batch, system, pairs, rows, first_begin, first_count, second_begin, second_count,        \
          same_bucket, work_count, want_j, want_k, integral_range(exchange_range), exchange_omega, \
          screening, bounds, density, coulomb, exchange, work_census);                             \
    break
  switch (angular_order) {
    GENERATIVEQC_CANONICAL_JK_ORDER(0);
    GENERATIVEQC_CANONICAL_JK_ORDER(1);
    GENERATIVEQC_CANONICAL_JK_ORDER(2);
    GENERATIVEQC_CANONICAL_JK_ORDER(3);
    GENERATIVEQC_CANONICAL_JK_ORDER(4);
    GENERATIVEQC_CANONICAL_JK_ORDER(5);
    GENERATIVEQC_CANONICAL_JK_ORDER(6);
    GENERATIVEQC_CANONICAL_JK_ORDER(7);
    GENERATIVEQC_CANONICAL_JK_ORDER(8);
    GENERATIVEQC_CANONICAL_JK_ORDER(9);
    GENERATIVEQC_CANONICAL_JK_ORDER(10);
    GENERATIVEQC_CANONICAL_JK_ORDER(11);
    GENERATIVEQC_CANONICAL_JK_ORDER(12);
  }
#undef GENERATIVEQC_CANONICAL_JK_ORDER
}

void launch_canonical_jk_kernel(cudaStream_t stream, DeviceBatch batch, bool cartesian,
                                std::int32_t system, unsigned angular_order,
                                const std::int32_t* pairs, CanonicalPairRows rows,
                                std::size_t first_begin, std::size_t first_count,
                                std::size_t second_begin, std::size_t second_count,
                                bool same_bucket, bool want_j, bool want_k, bool unrestricted,
                                DirectCoulombRange exchange_range, double exchange_omega,
                                double screening, const double* bounds, const double* density,
                                double* coulomb, double* exchange, std::uint64_t* work_census) {
  if (cartesian)
    launch_canonical_jk_source<true>(stream, batch, system, angular_order, pairs, rows, first_begin,
                                     first_count, second_begin, second_count, same_bucket, want_j,
                                     want_k, unrestricted, exchange_range, exchange_omega,
                                     screening, bounds, density, coulomb, exchange, work_census);
  else
    launch_canonical_jk_source<false>(
        stream, batch, system, angular_order, pairs, rows, first_begin, first_count, second_begin,
        second_count, same_bucket, want_j, want_k, unrestricted, exchange_range, exchange_omega,
        screening, bounds, density, coulomb, exchange, work_census);
}

template <bool Cartesian>
void launch_canonical_rsh_derivative_source(
    cudaStream_t stream, DeviceBatch batch, std::int32_t system, unsigned angular_order,
    const std::int32_t* pairs, CanonicalPairRows rows, std::size_t first_begin,
    std::size_t first_count, std::size_t second_begin, std::size_t second_count, bool same_bucket,
    std::size_t source_stride, double cj, double short_ck, double long_ck, bool unrestricted,
    double omega, double screening, const double* bounds, const double* density, const double* beta,
    double* out, std::uint64_t* work_count) {
  const auto dense_count =
      same_bucket ? first_count * (first_count + 1U) / 2U : first_count * second_count;
  if (!dense_count) return;
  constexpr unsigned threads = 64U;
  const auto blocks =
      static_cast<unsigned>(std::min<std::size_t>((dense_count + threads - 1U) / threads, 4096U));
#define GENERATIVEQC_CANONICAL_RSH_ORDER(order)                                              \
  case order:                                                                                \
    canonical_rsh_derivative_kernel<order, Cartesian><<<blocks, threads, 0, stream>>>(       \
        batch, system, pairs, rows, first_begin, first_count, second_begin, second_count,    \
        same_bucket, dense_count, source_stride, cj, short_ck, long_ck, unrestricted, omega, \
        screening, bounds, density, beta, out, work_count);                                  \
    break
  switch (angular_order) {
    GENERATIVEQC_CANONICAL_RSH_ORDER(0);
    GENERATIVEQC_CANONICAL_RSH_ORDER(1);
    GENERATIVEQC_CANONICAL_RSH_ORDER(2);
    GENERATIVEQC_CANONICAL_RSH_ORDER(3);
    GENERATIVEQC_CANONICAL_RSH_ORDER(4);
    GENERATIVEQC_CANONICAL_RSH_ORDER(5);
    GENERATIVEQC_CANONICAL_RSH_ORDER(6);
    GENERATIVEQC_CANONICAL_RSH_ORDER(7);
    GENERATIVEQC_CANONICAL_RSH_ORDER(8);
    GENERATIVEQC_CANONICAL_RSH_ORDER(9);
    GENERATIVEQC_CANONICAL_RSH_ORDER(10);
    GENERATIVEQC_CANONICAL_RSH_ORDER(11);
    GENERATIVEQC_CANONICAL_RSH_ORDER(12);
  }
#undef GENERATIVEQC_CANONICAL_RSH_ORDER
}

void launch_canonical_rsh_derivative_kernel(cudaStream_t stream, DeviceBatch batch, bool cartesian,
                                            std::int32_t system, unsigned angular_order,
                                            const std::int32_t* pairs, CanonicalPairRows rows,
                                            std::size_t first_begin, std::size_t first_count,
                                            std::size_t second_begin, std::size_t second_count,
                                            bool same_bucket, std::size_t source_stride, double cj,
                                            double short_ck, double long_ck, bool unrestricted,
                                            double omega, double screening, const double* bounds,
                                            const double* density, const double* beta, double* out,
                                            std::uint64_t* work_count) {
  if (cartesian)
    launch_canonical_rsh_derivative_source<true>(
        stream, batch, system, angular_order, pairs, rows, first_begin, first_count, second_begin,
        second_count, same_bucket, source_stride, cj, short_ck, long_ck, unrestricted, omega,
        screening, bounds, density, beta, out, work_count);
  else
    launch_canonical_rsh_derivative_source<false>(
        stream, batch, system, angular_order, pairs, rows, first_begin, first_count, second_begin,
        second_count, same_bucket, source_stride, cj, short_ck, long_ck, unrestricted, omega,
        screening, bounds, density, beta, out, work_count);
}

void launch_independent_jk_derivative_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    std::size_t coordinates_per_item, std::size_t system_begin, double cj, double ck,
    bool unrestricted, DirectCoulombRange exchange_range, double exchange_omega, double screening,
    const double* bounds, const double* density, const double* beta, double* out) {
  const std::size_t matrix = static_cast<std::size_t>(batch.nbf) * batch.nbf;
  const std::size_t system_count =
      coordinates_per_item == 0 ? 0 : static_cast<std::size_t>(grid.x) / coordinates_per_item;
  const std::size_t quartet_count = system_count * matrix * matrix;
  const unsigned blocks =
      static_cast<unsigned>(std::min<std::size_t>((quartet_count + block.x - 1) / block.x, 65535));
  if (blocks == 0) return;
  independent_jk_derivative_kernel<<<blocks, block, shared_bytes, stream>>>(
      batch, system_begin, system_count, cj, ck, unrestricted, integral_range(exchange_range),
      exchange_omega, screening, bounds, density, beta, out);
}

void launch_independent_rsh_derivative_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    std::size_t coordinates_per_item, std::size_t system_begin, std::size_t source_stride,
    double cj, double short_ck, double long_ck, bool unrestricted, double omega, double screening,
    const double* bounds, const double* density, const double* beta, double* out) {
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t pair_count = n * (n + 1) / 2;
  const std::size_t unique_quartets = pair_count * (pair_count + 1) / 2;
  const std::size_t system_count =
      coordinates_per_item == 0 ? 0 : static_cast<std::size_t>(grid.x) / coordinates_per_item;
  const std::size_t quartet_count = system_count * unique_quartets;
  const unsigned blocks =
      static_cast<unsigned>(std::min<std::size_t>((quartet_count + block.x - 1) / block.x, 65535));
  if (blocks == 0) return;
  independent_rsh_derivative_kernel<<<blocks, block, shared_bytes, stream>>>(
      batch, system_begin, system_count, source_stride, cj, short_ck, long_ck, unrestricted, omega,
      screening, bounds, density, beta, out);
}

void launch_bounded_shell_fock_source(
    bool unrestricted, unsigned worker_blocks, cudaStream_t stream, DeviceBatch batch,
    double screening, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    std::uint64_t covered_shell_class_mask, const std::uint32_t* class_state,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* output,
    unsigned long long* cursor, bool coulomb_only, bool exchange_only) {
  launch_bounded_direct_fock_source_shell_quartet_kernel(
      unrestricted, worker_blocks, kBoundedDirectThreads, 0, stream, batch, screening,
      shell_pair_bounds, shell_pair_density_bounds, pair_order, shell_pair_block_bounds,
      system_density_bounds, covered_shell_class_mask, class_state, schwarz_bounds, density, active,
      output, cursor, coulomb_only, exchange_only);
}

void launch_bounded_shell_range_exchange_source(
    bool unrestricted, unsigned worker_blocks, cudaStream_t stream, DeviceBatch batch,
    double screening, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint32_t* class_state, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* output, unsigned long long* cursor,
    DirectCoulombRange range, double omega) {
  const DirectRangeOperator radial_operator =
      range == DirectCoulombRange::Long
          ? DirectRangeOperator::Long
          : (range == DirectCoulombRange::Short ? DirectRangeOperator::Short
                                                : DirectRangeOperator::Full);
  launch_bounded_direct_range_exchange_fock_kernel(
      unrestricted, worker_blocks, kBoundedDirectThreads, 0, stream, batch, screening,
      shell_pair_bounds, shell_pair_density_bounds, pair_order, shell_pair_block_bounds,
      system_density_bounds, class_state, schwarz_bounds, density, active, output, cursor,
      radial_operator, omega);
}

void launch_bounded_shell_energy_derivative(
    bool unrestricted, unsigned worker_blocks, cudaStream_t stream, DeviceBatch batch,
    double screening, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint32_t* class_state, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* output, unsigned long long* cursor,
    double coulomb_coefficient, double exchange_coefficient,
    detail::BoundedDirectBlockDomain block_domain) {
  launch_bounded_direct_shell_quartet_kernel_scaled(
      unrestricted, DirectScreeningPurpose::Force, worker_blocks, kBoundedDirectThreads, 0, stream,
      batch, screening, shell_pair_bounds, shell_pair_density_bounds, pair_order,
      shell_pair_block_bounds, system_density_bounds, nullptr, 0U, class_state, schwarz_bounds,
      density, active, output, cursor, nullptr, coulomb_coefficient, exchange_coefficient, true,
      block_domain);
}

void launch_bounded_shell_range_exchange_derivative(
    bool unrestricted, unsigned worker_blocks, cudaStream_t stream, DeviceBatch batch,
    double screening, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint32_t* class_state, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* output, unsigned long long* cursor,
    DirectCoulombRange range, double omega, double exchange_coefficient) {
  const DirectRangeOperator radial_operator =
      range == DirectCoulombRange::Long
          ? DirectRangeOperator::Long
          : (range == DirectCoulombRange::Short ? DirectRangeOperator::Short
                                                : DirectRangeOperator::Full);
  launch_bounded_direct_range_exchange_force_kernel(
      unrestricted, worker_blocks, kBoundedDirectThreads, 0, stream, batch, screening,
      shell_pair_bounds, shell_pair_density_bounds, pair_order, shell_pair_block_bounds,
      system_density_bounds, class_state, schwarz_bounds, density, active, output, cursor,
      radial_operator, omega, exchange_coefficient);
}

void launch_bounded_shell_rsh_derivatives(
    bool unrestricted, unsigned worker_blocks, cudaStream_t stream, DeviceBatch batch,
    double screening, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint32_t* class_state, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* source_forces, unsigned long long* cursor, double omega,
    double coulomb_coefficient, double short_exchange_coefficient,
    double long_exchange_coefficient) {
  launch_bounded_direct_rsh_force_kernel(
      unrestricted, worker_blocks, kBoundedDirectThreads, 0, stream, batch, screening,
      shell_pair_bounds, shell_pair_density_bounds, pair_order, shell_pair_block_bounds,
      system_density_bounds, class_state, schwarz_bounds, density, active, source_forces, cursor,
      omega, coulomb_coefficient, short_exchange_coefficient, long_exchange_coefficient);
}

}  // namespace cuda_execution

}  // namespace generativeqc::scf
