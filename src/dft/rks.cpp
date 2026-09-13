#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "dft/ao_grid.hpp"
#include "dft/grid.hpp"
#include "dft/xc.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/fock_build.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/initial_guess/density.hpp"
#include "scf/mean_field.hpp"
#include "scf/reference/mean_field.hpp"
#include "scf/solver/diis.hpp"
#include "scf/solver/proposal_control.hpp"

namespace vibeqc::scf {
namespace {

using initial_guess::prepare_initial_density;
using initial_guess::prepare_initial_uhf_density;
using initial_guess::spin_occupations;
using reference::commutator_residual;
using reference::concatenate;
using reference::density_from_orbitals;
using reference::density_rms;
using reference::dot;
using reference::EigenResult;
using reference::generalized_eigen;
using reference::Matrix;
using reference::residual_rms;
using reference::split_spin_matrices;
using reference::symmetric_orthogonalizer;
using solver::Diis;
using solver::validate_seed;

// A run owns its immutable basis/grid binding and reference identity. IDs
// never alias across prepared replays or concurrent callers; exhaustion fails
// before wraparound rather than authorizing a previously exported factor.
std::uint64_t next_rks_identity() {
  static std::atomic<std::uint64_t> next{1};
  auto value = next.load(std::memory_order_relaxed);
  do {
    if (value == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("RKS density identity exhausted");
  } while (!next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed));
  return value;
}

struct RksEvaluation {
  Matrix fock;
  double energy{};
  dft::XcDensityDiagnostic density_diagnostic;
};

struct UksEvaluation {
  Matrix alpha_fock;
  Matrix beta_fock;
  double energy{};
};

using RksXcEvaluator = dft::XcIntegral (*)(const dft::AoBasis&, const dft::MolecularGrid&,
                                           const Matrix&, dft::XcDensitySource);

dft::XcIntegral evaluate_lda_xc_rks(const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                                    const Matrix& density, dft::XcDensitySource source) {
  return dft::integrate_lda_xc_pw_rks(basis, grid, density, 256, source);
}

dft::XcIntegral evaluate_pbe_xc_rks(const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                                    const Matrix& density, dft::XcDensitySource source) {
  return dft::integrate_pbe_rks_with_tail(basis, grid, density, 256, source);
}

dft::SpinXcIntegral evaluate_lda_xc_uks(const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                                        const Matrix& alpha_density, const Matrix& beta_density) {
  return dft::integrate_lda_xc_pw_uks(basis, grid, alpha_density, beta_density);
}

dft::SpinXcIntegral evaluate_pbe_xc_uks(const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                                        const Matrix& alpha_density, const Matrix& beta_density) {
  return dft::integrate_pbe_uks_with_tail(basis, grid, alpha_density, beta_density);
}

RksEvaluation evaluate_rks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                           const dft::MolecularGrid& grid, const Matrix& density,
                           RksXcEvaluator evaluate_xc, const char* method_name,
                           dft::XcDensitySource source, std::size_t retained_capacity) {
  const auto& strategy = plan.strategy();
  const auto& ints = plan.one_electron();
  const auto jk = plan.build(density);
  RksEvaluation result;
  result.fock = assemble_fock(strategy, ints.hcore, jk).alpha;
  const auto xc = evaluate_xc(basis, grid, density, source);
  result.density_diagnostic = xc.density_diagnostic;
  // The AO tile and potential were live together with these J/Fock buffers
  // inside evaluate_xc. Its peak excludes borrowed D/factor to avoid charging
  // the caller's retained state twice.
  runtime::sample_cpu_capacity(runtime::add_capacity(
      retained_capacity,
      runtime::add_capacity(xc.density_diagnostic.owned_numeric_bytes,
                            runtime::vector_capacities(result.fock, jk.coulomb, jk.exchange_alpha,
                                                       jk.exchange_beta))));
  if (xc.potential.size() != result.fock.size())
    throw std::runtime_error(std::string(method_name) +
                             " XC potential dimensions do not match the Fock matrix");
  for (std::size_t i = 0; i < result.fock.size(); ++i) result.fock[i] += xc.potential[i];
  result.energy = ints.nuclear_repulsion + dot(density, ints.hcore) +
                  contract_fock_energy(strategy, jk, density) + xc.energy;
  if (!std::isfinite(result.energy))
    throw std::runtime_error(std::string("nonfinite ") + method_name + " RKS energy");
  return result;
}

using UksXcEvaluator = dft::SpinXcIntegral (*)(const dft::AoBasis&, const dft::MolecularGrid&,
                                               const Matrix&, const Matrix&);

UksEvaluation evaluate_uks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                           const dft::MolecularGrid& grid, const Matrix& alpha_density,
                           const Matrix& beta_density, UksXcEvaluator evaluate_xc,
                           const char* method_name) {
  const auto& strategy = plan.strategy();
  const auto& ints = plan.one_electron();
  const auto jk = plan.build(alpha_density, beta_density);
  auto fock = assemble_fock(strategy, ints.hcore, jk);
  const auto xc = evaluate_xc(basis, grid, alpha_density, beta_density);
  if (xc.potential[0].size() != fock.alpha.size() || xc.potential[1].size() != fock.beta.size())
    throw std::runtime_error(std::string(method_name) +
                             " UKS XC potential dimensions do not match the Fock matrices");
  for (std::size_t i = 0; i < fock.alpha.size(); ++i) {
    fock.alpha[i] += xc.potential[0][i];
    fock.beta[i] += xc.potential[1][i];
  }
  UksEvaluation result;
  result.alpha_fock = std::move(fock.alpha);
  result.beta_fock = std::move(fock.beta);
  result.energy = ints.nuclear_repulsion + dot(alpha_density, ints.hcore) +
                  dot(beta_density, ints.hcore) +
                  contract_fock_energy(strategy, jk, alpha_density, beta_density) + xc.energy;
  if (!std::isfinite(result.energy))
    throw std::runtime_error(std::string("nonfinite ") + method_name + " UKS energy");
  return result;
}

ScfResult run_rks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                  const dft::MolecularGrid& grid, const ScfOptions& options,
                  const std::vector<double>* initial_density, RksXcEvaluator evaluate_xc,
                  const char* method_name) {
  if (options.xc_density_route != dft::XcDensityRoute::DensityMatrix &&
      options.xc_density_route != dft::XcDensityRoute::OccupiedOrbitals)
    throw std::invalid_argument("unsupported RKS XC density route");
  const auto& strategy = plan.strategy();
  validate_resolved_fock_build(strategy);
  const auto& system = plan.system();
  const auto& ints = plan.one_electron();
  if (options.compute_forces)
    throw std::invalid_argument(std::string(method_name) + " RKS forces are not implemented");
  if (strategy.backend != FockBackend::Cpu || strategy.spec.spin != FockSpin::Restricted ||
      strategy.spec.derivative_order != 0 || !strategy.spec.coulomb.present ||
      strategy.spec.coulomb.coefficient != 1.0 || strategy.spec.exchange.present)
    throw std::invalid_argument(std::string(method_name) +
                                " RKS requires a CPU Coulomb-only Fock strategy");
  if (system.electron_count <= 0 || system.electron_count % 2 || system.multiplicity != 1)
    throw std::invalid_argument(std::string(method_name) +
                                " RKS requires a closed-shell electron count");
  if (basis.nao != ints.nbf || basis.natom != system.atoms.size() || grid.point_count() == 0 ||
      grid.system().atoms.size() != system.atoms.size())
    throw std::invalid_argument(std::string(method_name) +
                                " RKS prepared grid/basis state is inconsistent");

  const std::size_t n = ints.nbf;
  const std::size_t occupied = static_cast<std::size_t>(system.electron_count / 2);
  if (occupied > n) throw std::runtime_error("basis has fewer orbitals than occupied pairs");
  const Matrix orthogonalizer = symmetric_orthogonalizer(ints.overlap, n);
  EigenResult orbitals;
  Matrix density =
      prepare_initial_density(system, ints, orthogonalizer, occupied, initial_density, orbitals);
  if (options.strict_initial_density && initial_density) {
    validate_seed(ints.overlap, *initial_density, n, {static_cast<unsigned>(system.electron_count)},
                  2.0);
    density = *initial_density;
  }
  Diis diis(options.diis_history);
  ScfResult result;
  result.initial_density_used = initial_density != nullptr;
  auto& diagnostic = result.xc_density_diagnostic;
  diagnostic.physical_residual = std::numeric_limits<double>::infinity();
  std::shared_ptr<const OccupiedDensityFactor> factor;
  DensityFactorIdentity identity{};
  const bool use_orbitals = options.xc_density_route == dft::XcDensityRoute::OccupiedOrbitals;
  if (use_orbitals) {
    const auto owner = next_rks_identity();
    identity = {owner, owner, 0, 0};
  }
  const auto retained_capacity = [&] {
    return runtime::add_capacity(
        runtime::add_capacity(plan.cpu_observation_capacity(), diis.numeric_capacity()),
        runtime::add_capacity(factor ? factor->numeric_capacity_bytes() : 0,
                              runtime::vector_capacities(
                                  orthogonalizer, density, orbitals.values, orbitals.vectors,
                                  basis.packed, grid.points(), grid.weights(), grid.owners())));
  };
  const auto make_current_factor = [&](std::size_t extra_live_bytes = 0) {
    // Only an actual unmixed eigensolver state advances both generations.
    // Release the previous snapshot before packing the new one. Current D
    // remains owned for convergence tests and the Coulomb provider.
    factor.reset();
    ++identity.orbital_generation;
    ++identity.density_generation;
    Matrix packed(n * occupied), occupations(occupied, 2.0);
    for (std::size_t mu = 0; mu < n; ++mu)
      for (std::size_t o = 0; o < occupied; ++o)
        packed[mu * occupied + o] = orbitals.vectors[mu * n + o];
    factor = std::make_shared<const OccupiedDensityFactor>(identity, DensityFactorSpin::Restricted,
                                                           n, packed, occupations);
    const auto packing_bytes = runtime::vector_capacities(packed, occupations);
    diagnostic.packed_coefficient_elements += packed.size();
    diagnostic.factor_peak_bytes =
        std::max(diagnostic.factor_peak_bytes,
                 runtime::add_capacity(factor->numeric_capacity_bytes(), packing_bytes));
    runtime::sample_cpu_capacity(runtime::add_capacity(
        retained_capacity(), runtime::add_capacity(packing_bytes, extra_live_bytes)));
  };
  if (use_orbitals && !initial_density) make_current_factor();
  const auto evaluate_current = [&](std::size_t extra_live_bytes = 0) {
    auto physical = evaluate_rks(plan, basis, grid, density, evaluate_xc, method_name,
                                 {options.xc_density_route, factor.get(), identity},
                                 runtime::add_capacity(retained_capacity(), extra_live_bytes));
    const auto& record = physical.density_diagnostic;
    if (record.executed == dft::XcDensityRoute::OccupiedOrbitals)
      ++diagnostic.orbital_calls;
    else
      ++diagnostic.density_calls;
    if (record.fallback != dft::XcDensityFallback::None) ++diagnostic.fallback_calls;
    diagnostic.xc_peak_bytes = std::max(diagnostic.xc_peak_bytes, record.owned_numeric_bytes);
    return physical;
  };
  const auto next_density_from_orbitals = [&](std::size_t extra_live_bytes = 0) {
    if (use_orbitals) make_current_factor(extra_live_bytes);
    // Reuse the producer's exact witness rather than reconstructing D twice.
    Matrix next = use_orbitals ? Matrix(factor->density().begin(), factor->density().end())
                               : density_from_orbitals(orbitals.vectors, n, occupied);
    runtime::sample_cpu_capacity(runtime::add_capacity(
        retained_capacity(), runtime::add_capacity(runtime::vector_bytes(next), extra_live_bytes)));
    return next;
  };
  const auto retain_factor = [&] {
    result.xc_density_factor = factor;
    diagnostic.final_identity = identity;
  };
  double previous_energy = std::numeric_limits<double>::infinity();
  for (unsigned iteration = 1; iteration <= options.max_iterations; ++iteration) {
    ++result.fock_builds;
    const auto physical = evaluate_current();
    const Matrix residual = commutator_residual(physical.fock, density, ints.overlap, n);
    const Matrix effective_fock = diis.update(physical.fock, residual);
    orbitals = generalized_eigen(effective_fock, orthogonalizer, n);
    const auto iteration_bytes =
        runtime::vector_capacities(physical.fock, residual, effective_fock);
    Matrix next_density = next_density_from_orbitals(iteration_bytes);

    runtime::sample_cpu_capacity(runtime::add_capacity(
        retained_capacity(),
        runtime::add_capacity(iteration_bytes, runtime::vector_bytes(next_density))));
    result.iterations = iteration;
    result.energy = physical.energy;
    result.energy_change = std::isfinite(previous_energy)
                               ? std::abs(physical.energy - previous_energy)
                               : std::numeric_limits<double>::infinity();
    result.density_rms = density_rms(next_density, density);
    result.physical_residual_rms = residual_rms(residual);
    if (iteration > 1 && result.energy_change < options.energy_tolerance &&
        result.density_rms < options.density_tolerance &&
        result.physical_residual_rms < options.density_tolerance) {
      density = std::move(next_density);
      result.converged = true;
      break;
    }
    previous_energy = physical.energy;
    density = std::move(next_density);
  }
  if (!result.converged) {
    retain_factor();
    result.density = std::move(density);
    return result;
  }

  result.fock_builds += 2;
  auto final = evaluate_current();
  orbitals = generalized_eigen(final.fock, orthogonalizer, n);
  density = next_density_from_orbitals(runtime::vector_bytes(final.fock));
  final = evaluate_current(runtime::vector_bytes(final.fock));
  const auto final_residual = commutator_residual(final.fock, density, ints.overlap, n);
  diagnostic.physical_residual = residual_rms(final_residual);
  runtime::sample_cpu_capacity(runtime::add_capacity(
      retained_capacity(), runtime::vector_capacities(final.fock, final_residual)));
  retain_factor();
  result.energy = final.energy;
  result.physical_residual_rms =
      residual_rms(commutator_residual(final.fock, density, ints.overlap, n));
  result.density = std::move(density);
  return result;
}

/** A virtual-space level shift stabilizes an integer-occupation proposal in
 * a stationary spin-flip cycle. In the AO metric the virtual projector is
 * S-SDS for a unit-occupation spin density. Only the proposal is shifted;
 * energies and commutators always use the unshifted physical operator. */
EigenResult stabilized_uks_orbitals(Matrix fock, const Matrix& density, const Matrix& overlap,
                                    const Matrix& orthogonalizer, std::size_t n) {
  const Matrix occupied = reference::multiply(reference::multiply(overlap, density, n), overlap, n);
  constexpr double shift = 0.1;  // Hartree; numerical occupation stabilization.
  for (std::size_t i = 0; i < fock.size(); ++i) fock[i] += shift * (overlap[i] - occupied[i]);
  return generalized_eigen(fock, orthogonalizer, n);
}

ScfResult run_uks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                  const dft::MolecularGrid& grid, const ScfOptions& options,
                  const std::vector<double>* initial_density, UksXcEvaluator evaluate_xc,
                  const char* method_name) {
  const auto& strategy = plan.strategy();
  validate_resolved_fock_build(strategy);
  const auto& system = plan.system();
  const auto& ints = plan.one_electron();
  if (options.compute_forces)
    throw std::invalid_argument(std::string(method_name) + " UKS forces are not implemented");
  if (strategy.backend != FockBackend::Cpu || strategy.spec.spin != FockSpin::Unrestricted ||
      strategy.spec.derivative_order != 0 || !strategy.spec.coulomb.present ||
      strategy.spec.coulomb.coefficient != 1.0 || strategy.spec.exchange.present)
    throw std::invalid_argument(std::string(method_name) +
                                " UKS requires a CPU Coulomb-only unrestricted Fock strategy");
  const auto [alpha_occupied, beta_occupied] = spin_occupations(system);
  if (basis.nao != ints.nbf || basis.natom != system.atoms.size() || grid.point_count() == 0 ||
      grid.system().atoms.size() != system.atoms.size())
    throw std::invalid_argument(std::string(method_name) +
                                " UKS prepared grid/basis state is inconsistent");

  const std::size_t n = ints.nbf;
  if (alpha_occupied > n || beta_occupied > n)
    throw std::runtime_error("basis has fewer orbitals than required UKS spin occupations");
  const Matrix orthogonalizer = symmetric_orthogonalizer(ints.overlap, n);
  EigenResult alpha_orbitals;
  EigenResult beta_orbitals;
  auto [alpha_density, beta_density] =
      prepare_initial_uhf_density(ints, orthogonalizer, alpha_occupied, beta_occupied,
                                  initial_density, alpha_orbitals, beta_orbitals);
  if (options.strict_initial_density && initial_density) {
    validate_seed(ints.overlap, *initial_density, n,
                  {static_cast<unsigned>(alpha_occupied), static_cast<unsigned>(beta_occupied)},
                  1.0);
    std::tie(alpha_density, beta_density) = split_spin_matrices(*initial_density, n * n);
  }
  Diis diis(options.diis_history);
  ScfResult result;
  result.initial_density_used = initial_density != nullptr;
  double previous_energy = std::numeric_limits<double>::infinity();
  bool stabilize_occupations = false;
  for (unsigned iteration = 1; iteration <= options.max_iterations; ++iteration) {
    ++result.fock_builds;
    auto physical =
        evaluate_uks(plan, basis, grid, alpha_density, beta_density, evaluate_xc, method_name);
    const Matrix alpha_residual =
        commutator_residual(physical.alpha_fock, alpha_density, ints.overlap, n);
    const Matrix beta_residual =
        commutator_residual(physical.beta_fock, beta_density, ints.overlap, n);
    const Matrix physical_fock = concatenate(physical.alpha_fock, physical.beta_fock);
    const Matrix physical_residual = concatenate(alpha_residual, beta_residual);
    const Matrix effective_joined = diis.update(physical_fock, physical_residual);
    std::tie(physical.alpha_fock, physical.beta_fock) =
        split_spin_matrices(effective_joined, n * n);
    alpha_orbitals = stabilize_occupations
                         ? stabilized_uks_orbitals(physical.alpha_fock, alpha_density, ints.overlap,
                                                   orthogonalizer, n)
                         : generalized_eigen(physical.alpha_fock, orthogonalizer, n);
    beta_orbitals = stabilize_occupations
                        ? stabilized_uks_orbitals(physical.beta_fock, beta_density, ints.overlap,
                                                  orthogonalizer, n)
                        : generalized_eigen(physical.beta_fock, orthogonalizer, n);
    Matrix next_alpha = density_from_orbitals(alpha_orbitals.vectors, n, alpha_occupied, 1.0);
    Matrix next_beta = density_from_orbitals(beta_orbitals.vectors, n, beta_occupied, 1.0);

    runtime::sample_cpu_capacity(runtime::add_capacity(
        runtime::add_capacity(plan.cpu_observation_capacity(), diis.numeric_capacity()),
        runtime::vector_capacities(orthogonalizer, alpha_density, beta_density, physical.alpha_fock,
                                   physical.beta_fock, alpha_residual, beta_residual, physical_fock,
                                   physical_residual, effective_joined, alpha_orbitals.values,
                                   alpha_orbitals.vectors, beta_orbitals.values,
                                   beta_orbitals.vectors, next_alpha, next_beta)));
    result.iterations = iteration;
    result.energy = physical.energy;
    result.energy_change = std::isfinite(previous_energy)
                               ? std::abs(physical.energy - previous_energy)
                               : std::numeric_limits<double>::infinity();
    result.density_rms =
        density_rms(concatenate(next_alpha, next_beta), concatenate(alpha_density, beta_density));
    result.physical_residual_rms = residual_rms(physical_residual);
    // Symmetry-related determinants can alternate with identical energy and
    // tiny physical residual while their density difference stays finite.
    // Preserve the density gate: enable a standard virtual-space shift for
    // subsequent proposals and require another evaluated iteration to pass.
    // This also handles small finite frontier splittings, not just arbitrary
    // eigenvector rotations inside an exactly degenerate subspace.
    if (iteration > 1 && result.energy_change < options.energy_tolerance &&
        result.physical_residual_rms < options.density_tolerance &&
        result.density_rms >= options.density_tolerance)
      stabilize_occupations = true;
    if (iteration > 1 && result.energy_change < options.energy_tolerance &&
        result.density_rms < options.density_tolerance &&
        result.physical_residual_rms < options.density_tolerance) {
      // Return the density whose physical energy and residual passed all
      // gates. An additional untested physical-Fock update can leave the
      // stationary neighborhood after DIIS has suppressed an unstable mode.
      result.converged = true;
      break;
    }
    // At the iteration limit, retain the evaluated state too: diagnostics
    // must describe the returned density even when convergence failed.
    if (iteration == options.max_iterations) break;
    previous_energy = physical.energy;
    alpha_density = std::move(next_alpha);
    beta_density = std::move(next_beta);
  }
  result.density = concatenate(alpha_density, beta_density);
  return result;
}

}  // namespace

ScfResult run_lda_rks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                      const dft::MolecularGrid& grid, const ScfOptions& options,
                      const std::vector<double>* initial_density) {
  return run_rks(plan, basis, grid, options, initial_density, evaluate_lda_xc_rks, "LDA");
}

ScfResult run_pbe_rks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                      const dft::MolecularGrid& grid, const ScfOptions& options,
                      const std::vector<double>* initial_density) {
  return run_rks(plan, basis, grid, options, initial_density, evaluate_pbe_xc_rks, "PBE");
}

ScfResult run_lda_uks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                      const dft::MolecularGrid& grid, const ScfOptions& options,
                      const std::vector<double>* initial_density) {
  return run_uks(plan, basis, grid, options, initial_density, evaluate_lda_xc_uks, "LDA");
}

ScfResult run_pbe_uks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                      const dft::MolecularGrid& grid, const ScfOptions& options,
                      const std::vector<double>* initial_density) {
  return run_uks(plan, basis, grid, options, initial_density, evaluate_pbe_xc_uks, "PBE");
}

}  // namespace vibeqc::scf
