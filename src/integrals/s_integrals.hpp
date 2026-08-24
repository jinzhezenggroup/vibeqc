#ifndef QCE_INTEGRALS_S_INTEGRALS_HPP
#define QCE_INTEGRALS_S_INTEGRALS_HPP

#include "core/types.hpp"

#include <cstddef>
#include <vector>

namespace qce::integrals {

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

/**
 * Evaluate normalized, contracted Cartesian Gaussian molecular integrals.
 *
 * This implementation is the independent CPU oracle. Production CUDA
 * execution evaluates the same formulas on device and does not call this
 * routine or copy these integral tensors to the GPU.
 */
IntegralData build_cartesian_integrals(const core::System& system);

/** Backward-compatible name retained for the original s-shell test helpers. */
inline IntegralData build_s_integrals(const core::System& system) {
  return build_cartesian_integrals(system);
}

}  // namespace qce::integrals

#endif
