#include "dft/ks_final_state.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "dft/semilocal_family.hpp"
#include "scf/mean_field.hpp"
#include "scf/reference/mean_field.hpp"

#if VIBEQC_HAS_CUDA
#include "generated_split_hybrid_registry.cuh"
#endif

namespace vibeqc::dft {
namespace {

bool finite(const auto& values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

bool valid_model(const KsFinalStateIdentity& identity) {
  const auto& model = identity.model;
  const auto& fock = identity.determinant.model;
#if VIBEQC_HAS_CUDA
  const bool split_hybrid = generated::split_hybrid_registered(model.functional);
#else
  const bool split_hybrid = false;
#endif
  SemilocalFamily family;
  try {
    family = split_hybrid ? SemilocalFamily::R2scan : semilocal_family_from_code(model.functional);
  } catch (const std::invalid_argument&) {
    return false;
  }
  const bool pbe = family == SemilocalFamily::Pbe;
  const bool b3lyp = family == SemilocalFamily::B3lyp;
  const bool wb97mv = family == SemilocalFamily::Wb97mv;
  const bool cuda_pbe0 = pbe && fock.backend == scf::FockBackend::Cuda &&
                         fock.spec.coulomb.approximation == scf::FockApproximation::Exact &&
                         fock.spec.exchange.present &&
                         fock.spec.exchange.approximation == scf::FockApproximation::Exact &&
                         model.semilocal_exchange_scale == 0.75 &&
                         model.semilocal_correlation_scale == 1.0 && !model.range_correction &&
                         !model.nonlocal_correlation &&
                         fock.spec.exchange.coefficient == (model.spins == 1 ? -0.125 : -0.25);
  // Admit the prepared exact-K composition for the three CUDA semilocal
  // families, while the separately qualified PBE0 scale keeps its own gate.
  const bool cuda_primary_exchange =
      fock.backend == scf::FockBackend::Cuda && semilocal_family_has_cuda_ks(family) &&
      fock.spec.coulomb.approximation == scf::FockApproximation::Exact &&
      fock.spec.exchange.approximation == scf::FockApproximation::Exact &&
      !model.range_correction && !model.nonlocal_correlation;
  const bool cuda_range_exchange = [&] {
    if (!pbe || fock.backend != scf::FockBackend::Cuda || !model.range_correction ||
        model.nonlocal_correlation || model.semilocal_exchange_scale != 1.0 ||
        model.semilocal_correlation_scale != 1.0)
      return false;
    const auto& correction = *model.range_correction;
    return fock.spec.coulomb.approximation == scf::FockApproximation::Exact &&
           (!fock.spec.exchange.present ||
            (fock.spec.exchange.approximation == scf::FockApproximation::Exact &&
             fock.spec.exchange.op == scf::FockOperator::FullRange)) &&
           correction.backend == scf::FockBackend::Cuda && correction.spec.spin == fock.spec.spin &&
           correction.spec.derivative_order == 0 && !correction.spec.coulomb.present &&
           correction.spec.exchange.present &&
           correction.spec.exchange.approximation == scf::FockApproximation::Exact &&
           correction.spec.exchange.op == scf::FockOperator::LongRange &&
           correction.spec.exchange.omega > 0.0 &&
           correction.screening_tolerance == fock.screening_tolerance;
  }();
#if VIBEQC_HAS_CUDA
  const auto split_composition = generated::split_hybrid_composition(model.functional);
  const bool valid_split_exchange =
      !split_hybrid ||
      (fock.backend == scf::FockBackend::Cuda && split_composition.matched &&
       split_composition.exact_exchange_denominator && fock.spec.exchange.present &&
       fock.spec.coulomb.approximation == scf::FockApproximation::Exact &&
       fock.spec.exchange.approximation == scf::FockApproximation::Exact &&
       !model.range_correction && !model.nonlocal_correlation &&
       fock.spec.exchange.coefficient ==
           -static_cast<double>(split_composition.exact_exchange_numerator) /
               static_cast<double>(split_composition.exact_exchange_denominator) /
               (model.spins == 1 ? 2.0 : 1.0));
#else
  const bool valid_split_exchange = true;
#endif
  if (model.version != 1 ||
      model.scf_domain_version != (split_hybrid ? 4U : semilocal_family_domain_version(family)) ||
      !valid_split_exchange || !model.tile_points || !model.owner ||
      (model.spins != 1 && model.spins != 2) ||
      !((fock.backend == scf::FockBackend::Cpu && model.device == -1) ||
        (fock.backend == scf::FockBackend::Cuda && model.device >= 0)) ||
      identity.determinant.occupied.size() != model.spins ||
      (fock.spec.spin == scf::FockSpin::Restricted ? 1U : 2U) != model.spins ||
      fock.precision != scf::FockPrecision::Float64 || fock.spec.derivative_order != 0 ||
      !fock.spec.coulomb.present || fock.spec.coulomb.coefficient != 1.0 ||
      (fock.spec.coulomb.approximation != scf::FockApproximation::Exact &&
       fock.spec.coulomb.approximation != scf::FockApproximation::DensityFitted) ||
      fock.spec.coulomb.op != scf::FockOperator::FullRange ||
      !std::isfinite(model.semilocal_exchange_scale) || model.semilocal_exchange_scale < 0 ||
      !std::isfinite(model.semilocal_correlation_scale) || model.semilocal_correlation_scale < 0 ||
      (fock.spec.exchange.present &&
       (fock.spec.exchange.op != scf::FockOperator::FullRange ||
        (fock.spec.exchange.approximation != scf::FockApproximation::Exact &&
         fock.spec.exchange.approximation != scf::FockApproximation::DensityFitted) ||
        fock.spec.exchange.coefficient >= 0)) ||
      (!b3lyp && !wb97mv &&
       (!pbe || (fock.backend == scf::FockBackend::Cuda && !cuda_pbe0 && !cuda_range_exchange)) &&
       (model.semilocal_exchange_scale != 1 || model.semilocal_correlation_scale != 1 ||
        (fock.spec.exchange.present && !cuda_primary_exchange && !cuda_range_exchange) ||
        (model.range_correction && !cuda_range_exchange))) ||
      (b3lyp && (model.semilocal_exchange_scale != 1 || model.semilocal_correlation_scale != 1 ||
                 !fock.spec.exchange.present ||
                 fock.spec.exchange.coefficient != (model.spins == 1 ? -0.1 : -0.2) ||
                 (fock.backend == scf::FockBackend::Cuda &&
                  (fock.spec.coulomb.approximation != scf::FockApproximation::Exact ||
                   fock.spec.exchange.approximation != scf::FockApproximation::Exact)))))
    return false;
  try {
    if (wb97mv) {
      if (!model.range_correction || !model.nonlocal_correlation ||
          model.nonlocal_density_domain != nlc::Vv10DensityDomain::MolecularV1 ||
          model.semilocal_exchange_scale != 1.0 || model.semilocal_correlation_scale != 1.0)
        return false;
      scf::require_wb97mv_composition(fock, *model.range_correction, *model.nonlocal_correlation);
    }
    validate_grid_spec(model.grid);
    scf::validate_resolved_fock_build(fock);
    if (model.range_correction) scf::validate_resolved_fock_build(*model.range_correction);
  } catch (const std::invalid_argument&) {
    return false;
  }
  return true;
}

bool finite_components(const EnergyComponents& components) {
  return std::isfinite(components.nuclear) && std::isfinite(components.one_electron) &&
         std::isfinite(components.hartree) && std::isfinite(components.xc) &&
         std::isfinite(components.exact_exchange) && std::isfinite(components.total());
}

}  // namespace

core::ElectronicReferenceView electronic_reference(const VerifiedKsFinalState& state,
                                                   const scf::reference::Matrix& overlap,
                                                   const scf::reference::Matrix& hcore) {
  const auto spins = state.orbitals.size();
  if (!spins || spins > 2 || state.identity.determinant.occupied.size() != spins ||
      state.density.size() != spins || state.fock.size() != spins ||
      (!state.weighted_density.empty() && state.weighted_density.size() != spins))
    throw std::invalid_argument("invalid verified KS reference shape");

  core::ElectronicReferenceView view;
  view.basis_functions = state.orbitals.front().values.size();
  view.spin_channels = spins;
  view.overlap = overlap;
  view.hcore = hcore;
  view.energy = state.diagnostic.component_energy;
  for (std::size_t spin = 0; spin < spins; ++spin) {
    const auto& orbital = state.orbitals[spin];
    view.channels[spin] = {state.identity.determinant.occupied[spin],
                           orbital.vectors,
                           orbital.values,
                           state.density[spin],
                           state.fock[spin],
                           state.weighted_density.empty()
                               ? std::span<const double>{}
                               : std::span<const double>{state.weighted_density[spin]}};
  }
  core::validate_electronic_reference_shape(view);
  return view;
}

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
