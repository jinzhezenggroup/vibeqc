#ifndef VIBEQC_INTEGRALS_ERI_GEOMETRY_HPP
#define VIBEQC_INTEGRALS_ERI_GEOMETRY_HPP

#include <type_traits>

#include "integrals/range_moments.hpp"

#ifdef __CUDACC__
#define VIBEQC_ERI_HD __host__ __device__
#else
#define VIBEQC_ERI_HD
#endif

namespace vibeqc::integrals {

/** Bind unnormalized primitive geometry to a generated Hermite/AD callable.
 * The caller supplies four positive exponents and twelve center coordinates.
 * Contraction coefficients, component normalization and external cotangents
 * belong in the callable's weights, exactly once. The Geometry template is
 * the compiler's existing geometry-factored weighted ERI interface.
 *
 * This binder uses the explicit radial operator and never screens. It does
 * not replace the established full-Coulomb native/HF geometry path. Derived
 * overflow is reported to the caller; there is no fallback to another kernel.
 */
template <typename Geometry, unsigned MaximumOrder = 13>
VIBEQC_ERI_HD bool make_eri_geometry(const double* exponents, const double* centers,
                                     unsigned maximum_order, CoulombRange range, double omega,
                                     Geometry& geometry) {
  // The explicit bound protects both legacy fourteen-moment geometry and
  // second-order geometry. A smaller caller array cannot silently overflow.
  static_assert(std::extent<decltype(geometry.boys)>::value > MaximumOrder,
                "ERI geometry must own every moment in its declared domain");
  if (maximum_order > MaximumOrder) return false;
  if (!exponents || !centers) return false;
  for (unsigned slot = 0; slot < 4; ++slot) {
    if (!std::isfinite(exponents[slot]) || exponents[slot] <= 0) return false;
    for (unsigned axis = 0; axis < 3; ++axis)
      if (!std::isfinite(centers[3 * slot + axis])) return false;
  }
  const double p = exponents[0] + exponents[1], q = exponents[2] + exponents[3];
  if (!std::isfinite(p + q)) return false;
  const double mu = (exponents[0] / p) * exponents[1];
  const double nu = (exponents[2] / q) * exponents[3];
  geometry.inverse_two_p = 0.5 / p;
  geometry.inverse_two_q = 0.5 / q;
  geometry.rho = p < q ? p / (1 + p / q) : q / (1 + q / p);
  double distance = 0, decay_argument = 0;
  for (unsigned slot = 0; slot < 4; ++slot)
    geometry.product_scales[slot] = exponents[slot] / (slot < 2 ? p : q);
  for (unsigned axis = 0; axis < 3; ++axis) {
    const double ab = centers[axis] - centers[3 + axis];
    const double cd = centers[6 + axis] - centers[9 + axis];
    const double first = centers[axis] - geometry.product_scales[1] * ab;
    const double second = centers[6 + axis] - geometry.product_scales[3] * cd;
    geometry.difference[axis] = first - second;
    distance += geometry.difference[axis] * geometry.difference[axis];
    decay_argument += mu * ab * ab + nu * cd * cd;
    for (unsigned slot = 0; slot < 4; ++slot) {
      geometry.shifts[slot][axis] = (slot < 2 ? first : second) - centers[3 * slot + axis];
      const double pair_decay = slot < 2 ? mu * ab : nu * cd;
      geometry.decay[slot][axis] = (slot % 2 ? 2.0 : -2.0) * pair_decay;
    }
  }
  constexpr double pi = 0x1.921fb54442d18p+1;
  geometry.prefactor =
      (2 * std::pow(pi, 2.5) / p / q / std::sqrt(p + q)) * std::exp(-decay_argument);
  return std::isfinite(geometry.prefactor) &&
         bounded_range_moments<MaximumOrder>(maximum_order, geometry.rho * distance, geometry.rho,
                                             range, omega, geometry.boys);
}
}  // namespace vibeqc::integrals
#undef VIBEQC_ERI_HD
#endif
