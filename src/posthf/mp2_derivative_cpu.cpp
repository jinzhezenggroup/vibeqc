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

}  // namespace vibeqc::mp2
