#include "dft/ks_final_state.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "scf/reference/mean_field.hpp"

namespace vibeqc::dft {
namespace {

bool finite(const auto& values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

bool valid_model(const KsFinalStateIdentity& identity) {
  const auto& model = identity.model;
  const auto& fock = identity.determinant.model;
  if (model.version != 1 || model.scf_domain_version != 1 || !model.tile_points || !model.owner ||
      (model.spins != 1 && model.spins != 2) ||
      !((fock.backend == scf::FockBackend::Cpu && model.device == -1 && model.spins == 1) ||
        (fock.backend == scf::FockBackend::Cuda && model.device >= 0)) ||
      identity.determinant.occupied.size() != model.spins ||
      (fock.spec.spin == scf::FockSpin::Restricted ? 1U : 2U) != model.spins ||
      fock.precision != scf::FockPrecision::Float64 || fock.spec.derivative_order != 0 ||
      !fock.spec.coulomb.present || fock.spec.coulomb.coefficient != 1.0 ||
      fock.spec.coulomb.approximation != scf::FockApproximation::Exact ||
      fock.spec.coulomb.op != scf::FockOperator::FullRange || fock.spec.exchange.present)
    return false;
  try {
    validate_grid_spec(model.grid);
    scf::validate_resolved_fock_build(fock);
  } catch (const std::invalid_argument&) {
    return false;
  }
  return true;
}

bool finite_components(const EnergyComponents& components) {
  return std::isfinite(components.nuclear) && std::isfinite(components.one_electron) &&
         std::isfinite(components.hartree) && std::isfinite(components.xc) &&
         std::isfinite(components.total());
}

}  // namespace

bool validate_ks_final_state(const KsFinalStateIdentity& current,
                             const scf::reference::Matrix& overlap,
                             const scf::reference::Matrix& hcore, const KsPhysicalState& physical,
                             const KsFinalStateCandidate& candidate,
                             const scf::solver::FinalStateLimits& limits,
                             bool compute_weighted_density, VerifiedKsFinalState& output,
                             std::string& detail) {
  output = {};
  detail.clear();
  if (!valid_model(current) || physical.identity != current || candidate.identity != current ||
      !physical.physical || !finite_components(physical.components) ||
      !std::isfinite(physical.reported_energy) || !std::isfinite(physical.physical_residual) ||
      physical.density.size() != current.model.spins ||
      physical.fock.size() != current.model.spins ||
      candidate.spins.size() != current.model.spins ||
      candidate.fock_density_generation != current.determinant.factor.density_generation) {
    detail = "invalid KS final-state identity, model, physical state or spin shape";
    return false;
  }

  const double component_energy = physical.components.total();
  const double energy_error = std::abs(component_energy - physical.reported_energy);
  const double residual_gate = std::min(1e-9, limits.density_tolerance);
  if (energy_error > limits.energy_tolerance || physical.physical_residual < 0.0 ||
      physical.physical_residual > residual_gate) {
    detail = "KS final state failed component-energy or physical-residual consistency";
    return false;
  }

  scf::solver::PhysicalFockFrame fock{current.determinant, true, physical.fock};
  scf::solver::FinalFrameCandidate orbitals{candidate.identity.determinant,
                                            candidate.fock_density_generation,
                                            candidate.physical_origin, candidate.spins};
  scf::solver::FinalStateDiagnostic determinant;
  if (!scf::solver::validate_final_state(current.determinant, overlap, hcore,
                                         physical.components.nuclear, physical.density, fock,
                                         orbitals, limits, determinant, detail))
    return false;
  determinant.energy = component_energy;
  determinant.energy_change = 0;

  VerifiedKsFinalState verified{
      current,
      physical.density,
      physical.fock,
      {},
      candidate.spins,
      physical.components,
      {determinant, component_energy, energy_error, physical.physical_residual}};
  if (compute_weighted_density) {
    const double weight = current.model.spins == 1 ? 2.0 : 1.0;
    const auto n = static_cast<std::size_t>(std::sqrt(overlap.size()));
    for (std::size_t spin = 0; spin < current.model.spins; ++spin) {
      auto weighted = scf::reference::energy_weighted_density(
          verified.orbitals[spin].vectors, verified.orbitals[spin].values, n,
          current.determinant.occupied[spin], weight);
      if (!finite(weighted)) {
        detail = "nonfinite validated KS energy-weighted density";
        return false;
      }
      verified.weighted_density.push_back(std::move(weighted));
    }
  }
  output = std::move(verified);
  return true;
}

}  // namespace vibeqc::dft
