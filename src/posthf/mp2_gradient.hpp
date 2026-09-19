#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "response/native_gmres.hpp"

namespace vibeqc::posthf {
class NativeBlockProvider;
}
namespace vibeqc::scf {
struct PhysicalReference;
}

namespace vibeqc::mp2 {

struct EnergyAdjoint {
  std::size_t orbitals{};
  std::size_t occupied{};
  std::vector<double> integrals_iajb;
  std::vector<double> orbital_energies;
  const char* equation_hash{};
};

struct OrbitalRhs {
  std::size_t orbitals{};
  std::size_t occupied{};
  std::vector<double> energy_gradient;
  std::vector<double> response_rhs;
  std::vector<double> one_electron;
  std::vector<double> two_electron;
};

struct LagrangianWeights {
  std::size_t orbitals{};
  std::size_t occupied{};
  std::vector<double> one_electron;
  std::vector<double> two_electron;
  std::vector<double> overlap;
  double stationarity_residual{};
};

struct GradientResourcePlan {
  std::size_t provider_bytes{};
  std::size_t adjoint_bytes{};
  std::size_t response_bytes{};
  std::size_t relaxed_weight_bytes{};
  std::size_t shell_cotangent_bytes{};
  std::size_t derivative_staging_bytes{};
  std::size_t derivative_backend_staging_bytes{};
  std::size_t candidate_output_bytes{};
  std::size_t peak_bytes{};
};

EnergyAdjoint canonical_energy_adjoint(std::span<const double> integrals_iajb,
                                       std::span<const double> orbital_energies,
                                       std::size_t occupied, double denominator_threshold);
OrbitalRhs canonical_orbital_rhs(std::span<const double> hcore_mo, std::span<const double> eri_mo,
                                 const EnergyAdjoint& adjoint, double same_space_threshold);
OrbitalRhs canonical_orbital_rhs_streamed(const scf::PhysicalReference& reference,
                                          std::span<const double> hcore_mo,
                                          const posthf::NativeBlockProvider& provider,
                                          const EnergyAdjoint& adjoint, double same_space_threshold,
                                          bool cuda = false, int device_id = 0);
LagrangianWeights canonical_lagrangian_weights(std::span<const double> hcore_mo,
                                               std::span<const double> eri_mo,
                                               const EnergyAdjoint& adjoint,
                                               std::span<const double> response,
                                               double same_space_threshold);
LagrangianWeights canonical_lagrangian_weights_streamed(const scf::PhysicalReference& reference,
                                                        std::span<const double> hcore_mo,
                                                        const posthf::NativeBlockProvider& provider,
                                                        const EnergyAdjoint& adjoint,
                                                        std::span<const double> response,
                                                        double same_space_threshold,
                                                        bool cuda = false, int device_id = 0);
GradientResourcePlan conventional_gradient_plan(
    std::size_t orbitals, std::size_t occupied, std::size_t provider_bytes,
    const response::GmresPlan& response_plan, std::size_t maximum_shell_ao_count,
    std::size_t coordinate_count, std::size_t candidate_output_bytes, std::size_t budget_bytes,
    std::size_t derivative_backend_staging_bytes = 0);

}  // namespace vibeqc::mp2
