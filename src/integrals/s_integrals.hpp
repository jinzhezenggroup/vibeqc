#ifndef VIBEQC_INTEGRALS_S_INTEGRALS_HPP
#define VIBEQC_INTEGRALS_S_INTEGRALS_HPP

#include "core/types.hpp"

#include <cstddef>
#include <vector>

namespace vibeqc::integrals {

struct IntegralData {
  std::size_t nbf{};
  std::size_t ncoord{};
  std::vector<double> overlap;
  std::vector<double> hcore;
  std::vector<double> eri;
  // Derivative arrays are coordinate-major: derivative[coord * size + item].
  std::vector<double> overlap_derivative;
  std::vector<double> hcore_derivative;
  std::vector<double> eri_derivative;
  double nuclear_repulsion{};
  std::vector<double> nuclear_repulsion_derivative;
};

/** Independent two-/three-center Coulomb integral oracle for density fitting. */
struct DensityFittingIntegralData {
  std::size_t nbf{};
  std::size_t naux{};
  std::size_t ncoord{};
  // Row-major metric (P|Q) and three-center tensor (mu nu|P).
  std::vector<double> metric;
  std::vector<double> three_center;
  // Coordinate-major derivatives use the same item order as their values.
  std::vector<double> metric_derivative;
  std::vector<double> three_center_derivative;
};

/**
 * Evaluate normalized, contracted Cartesian or real-spherical integrals.
 *
 * This implementation is the independent CPU oracle. Production CUDA
 * execution evaluates the same formulas on device and does not call this
 * routine or copy these integral tensors to the GPU.
 */
IntegralData build_integrals(const core::System& system);

/**
 * Evaluate normalized two- and three-center density-fitting integrals.
 *
 * Orbital and auxiliary systems must describe the same atoms and geometry but
 * may use independent Cartesian or real-spherical shell sets. This host-only
 * implementation is the correctness oracle for future CUDA DF kernels.
 */
DensityFittingIntegralData build_density_fitting_integrals(
    const core::System& orbital_system,
    const core::System& auxiliary_system);

/** Compatibility name retained for callers that explicitly request Cartesian. */
inline IntegralData build_cartesian_integrals(const core::System& system) {
  return build_integrals(system);
}

/** Backward-compatible name retained for the original s-shell test helpers. */
inline IntegralData build_s_integrals(const core::System& system) {
  return build_integrals(system);
}

}  // namespace vibeqc::integrals

#endif
