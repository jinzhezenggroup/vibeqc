#ifndef VIBEQC_RUNTIME_CUDA_GAUSSIAN_PRODUCTS_CUH
#define VIBEQC_RUNTIME_CUDA_GAUSSIAN_PRODUCTS_CUH

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace vibeqc::runtime::cuda_gaussian_products {

/** Borrowed normalized AO expansions. All views use the same atom index space.
 * Radial coefficients and sparse Cartesian coefficients already include their
 * respective normalization factors; traversal must not normalize them again.
 */
struct BasisView {
  std::size_t nbf{};
  const std::int32_t *shell_atoms{}, *ao_shells{};
  const std::int64_t* primitive_offsets{};
  const std::uint8_t *term_counts{}, *term_angular{};
  const double *term_coefficients{}, *exponents{}, *coefficients{};
};

/** One real factor borrowing its caller's normalized basis metadata.
 * Views must outlive the contraction. Holding a pointer avoids copying all
 * nine metadata fields into every factor and enlarging the evaluator's stack.
 * No dummy zero-exponent Gaussian participates in this traversal.
 */
struct Factor {
  const BasisView* view{};
  std::int64_t ao{};
  __device__ Factor() = default;
  __device__ Factor(const BasisView& basis, std::int64_t index) : view(&basis), ao(index) {}
};

/** Generic tensor-product traversal, specialized only by rank and policy.
 * Primitive products are assigned cyclically without multiplying unbounded
 * contraction lengths. A policy owns the mathematical definition and channel
 * accumulation; this runtime owns normalized basis traversal and lane ownership.
 */
template <class Policy, unsigned Rank, std::size_t TermCapacity>
struct Product {
  const Factor (&factors)[Rank];
  std::int32_t shells[Rank], atoms[Rank];
  std::int64_t primitive[Rank];
  typename Policy::Vec3 centers[Rank];
  typename Policy::Angular angular[Rank];
  double exponents[Rank];
  typename Policy::Accumulator result{};
  unsigned owner{}, lane, lanes;

  template <unsigned Slot = 0>
  __device__ void terms(double weight) {
    if constexpr (Slot == Rank) {
      Policy::template accumulate<Rank>(result, exponents, centers, angular, weight);
    } else {
      const auto& basis = *factors[Slot].view;
      const auto ao = factors[Slot].ao;
      for (unsigned term = 0; term < basis.term_counts[ao]; ++term) {
        const auto index = ao * TermCapacity + term;
        const auto* powers = basis.term_angular + 3 * index;
        angular[Slot] = {powers[0], powers[1], powers[2]};
        terms<Slot + 1>(weight * basis.term_coefficients[index]);
      }
    }
  }

  template <unsigned Slot = 0>
  __device__ void primitives() {
    if constexpr (Slot == Rank) {
      const auto current = owner;
      if (++owner == lanes) owner = 0;
      if (current == lane) {
        // A cooperative lane loads arithmetic inputs only for products it
        // owns. Loading them while enumerating every other lane's products
        // needlessly extends live ranges and inflates per-thread stack state.
        double weight = 1.0;
#pragma unroll
        for (unsigned slot = 0; slot < Rank; ++slot) {
          exponents[slot] = factors[slot].view->exponents[primitive[slot]];
          weight *= factors[slot].view->coefficients[primitive[slot]];
        }
        terms(weight);
      }
    } else {
      const auto& basis = *factors[Slot].view;
      const auto shell = shells[Slot];
      for (auto p = basis.primitive_offsets[shell]; p < basis.primitive_offsets[shell + 1]; ++p) {
        primitive[Slot] = p;
        primitives<Slot + 1>();
      }
    }
  }
};

template <class Policy, std::size_t TermCapacity, unsigned Rank>
__device__ typename Policy::Accumulator contract(const Factor (&factors)[Rank],
                                                 const double* positions, unsigned lane = 0,
                                                 unsigned lanes = 1) {
  Product<Policy, Rank, TermCapacity> product{factors};
  product.lane = lane;
  product.lanes = lanes;
#pragma unroll
  for (unsigned slot = 0; slot < Rank; ++slot) {
    product.shells[slot] = factors[slot].view->ao_shells[factors[slot].ao];
    product.atoms[slot] = factors[slot].view->shell_atoms[product.shells[slot]];
    const auto* r = positions + 3 * product.atoms[slot];
    product.centers[slot] = {r[0], r[1], r[2]};
  }
  product.primitives();
  return product.result;
}

/** Project center channels onto one physical coordinate, summing shared atoms.
 * Policies provide positive center derivatives in mathematical-factor order.
 * Values never depend on a seed object or on a normalized absent basis factor.
 */
template <unsigned Rank, class Response>
__device__ double coordinate(const Factor (&factors)[Rank], const Response& response,
                             std::int64_t coordinate) {
  double value = 0;
#pragma unroll
  for (unsigned slot = 0; slot < Rank; ++slot) {
    const auto& basis = *factors[slot].view;
    if (basis.shell_atoms[basis.ao_shells[factors[slot].ao]] == coordinate / 3)
      value += response.gradient[slot][coordinate % 3];
  }
  return value;
}

/** One atomic sum per contracted center/channel, independent of primitive count. */
template <unsigned Rank, class Response>
__device__ void scatter(const Factor (&factors)[Rank], const Response& response, double weight,
                        double* gradient) {
#pragma unroll
  for (unsigned slot = 0; slot < Rank; ++slot) {
    const auto& basis = *factors[slot].view;
    const auto atom = basis.shell_atoms[basis.ao_shells[factors[slot].ao]];
#pragma unroll
    for (unsigned axis = 0; axis < 3; ++axis)
      atomicAdd(gradient + 3 * atom + axis, weight * response.gradient[slot][axis]);
  }
}

}  // namespace vibeqc::runtime::cuda_gaussian_products
#endif
