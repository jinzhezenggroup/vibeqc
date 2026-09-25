#include "scf/gradient/hf_gradient.hpp"

#include <array>

#include "generated_scf_array_native.hpp"

namespace vibeqc::scf::gradient {
std::vector<double> analytic_forces(const integrals::IntegralData& ints, const Matrix& density,
                                    const Matrix& weighted_density,
                                    std::span<const double> two_electron) {
  std::vector<double> forces(ints.ncoord, 0.0);
  generated::hf_stationary_forces<1>(
      forces.data(), ints.ncoord, ints.nbf, std::array<const double*, 1>{density.data()},
      std::array<const double*, 1>{weighted_density.data()}, ints.hcore_derivative.data(),
      ints.overlap_derivative.data(), two_electron.data(), ints.nuclear_repulsion_derivative.data());
  return forces;
}

std::vector<double> analytic_uhf_forces(const integrals::IntegralData& ints,
                                        const Matrix& alpha_density, const Matrix& beta_density,
                                        const Matrix& alpha_weighted_density,
                                        const Matrix& beta_weighted_density,
                                        std::span<const double> two_electron) {
  std::vector<double> forces(ints.ncoord, 0.0);
  generated::hf_stationary_forces<2>(
      forces.data(), ints.ncoord, ints.nbf,
      std::array<const double*, 2>{alpha_density.data(), beta_density.data()},
      std::array<const double*, 2>{alpha_weighted_density.data(), beta_weighted_density.data()},
      ints.hcore_derivative.data(), ints.overlap_derivative.data(), two_electron.data(),
      ints.nuclear_repulsion_derivative.data());
  return forces;
}

}  // namespace vibeqc::scf::gradient
