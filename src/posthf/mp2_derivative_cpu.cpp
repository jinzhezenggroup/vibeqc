#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "integrals/s_integrals.hpp"
#include "posthf/mp2_derivative.hpp"
#include "posthf/mp2_derivative_common.hpp"

namespace vibeqc::mp2 {

std::vector<double> conventional_derivative_cpu(const core::System& system,
                                                const scf::PhysicalReference& reference,
                                                const LagrangianWeights& weights) {
  return detail::conventional_derivative(
      system, reference, weights,
      [&](std::span<const double> overlap, std::span<const double> hcore) {
        return integrals::contract_weighted_one_electron_derivative(system, overlap, hcore, true);
      },
      [&](const std::array<std::size_t, 4>& shells, std::span<const double> local) {
        return integrals::contract_weighted_eri_shell_derivative(system, shells, local);
      });
}

std::vector<double> density_fitted_derivative_cpu(const core::System& orbital,
                                                  const core::System& auxiliary,
                                                  const DensityFittedLagrangianWeights& weights,
                                                  std::size_t stage_budget) {
  if (!stage_budget) throw std::invalid_argument("RI-MP2 derivative requires a memory budget");
  auto gradient = integrals::contract_weighted_one_electron_derivative(orbital, weights.overlap,
                                                                       weights.one_electron, true);
  auto density_fitting = integrals::contract_weighted_density_fitting_derivative(
      orbital, auxiliary, weights.three_center, weights.metric, stage_budget);
  if (gradient.size() != density_fitting.size())
    throw std::runtime_error("RI-MP2 derivative consumers returned inconsistent dimensions");
  for (std::size_t coordinate = 0; coordinate < gradient.size(); ++coordinate)
    gradient[coordinate] += density_fitting[coordinate];
  if (!std::all_of(gradient.begin(), gradient.end(),
                   [](double value) { return std::isfinite(value); }))
    throw std::runtime_error("RI-MP2 derivative contraction produced nonfinite values");
  return gradient;
}

}  // namespace vibeqc::mp2
