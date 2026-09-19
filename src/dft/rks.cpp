#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
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
#include "scf/solver/self_consistent.hpp"

namespace vibeqc::scf {
namespace {

using initial_guess::prepare_initial_density;
using reference::commutator_residual;
using reference::density_from_orbitals;
using reference::density_rms;
using reference::dot;
using reference::EigenResult;
using reference::generalized_eigen;
using reference::Matrix;
using reference::residual_rms;
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
  dft::EnergyComponents components;
  dft::XcDensityDiagnostic density_diagnostic;
};

using RksXcEvaluator = dft::XcIntegral (*)(const dft::AoBasis&, const dft::MolecularGrid&,
                                           const Matrix&, dft::XcDensitySource, std::size_t);

dft::XcIntegral evaluate_lda_xc_rks(const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                                    const Matrix& density, dft::XcDensitySource source,
                                    std::size_t tile) {
  return dft::integrate_lda_xc_pw_rks(basis, grid, density, tile, source);
}

dft::XcIntegral evaluate_pbe_xc_rks(const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                                    const Matrix& density, dft::XcDensitySource source,
                                    std::size_t tile) {
  return dft::integrate_pbe_rks_with_tail(basis, grid, density, tile, source);
}

RksEvaluation evaluate_rks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                           const dft::MolecularGrid& grid, const Matrix& density,
                           RksXcEvaluator evaluate_xc, const char* method_name,
                           dft::XcDensitySource source, std::size_t retained_capacity,
                           std::size_t tile) {
  const auto& strategy = plan.strategy();
  const auto& ints = plan.one_electron();
  const auto jk = plan.build(density);
  RksEvaluation result;
  result.fock = assemble_fock(strategy, ints.hcore, jk).alpha;
  const auto xc = evaluate_xc(basis, grid, density, source, tile);
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
  result.components = {ints.nuclear_repulsion, dot(density, ints.hcore),
                       contract_fock_energy(strategy, jk, density), xc.energy};
  result.energy = result.components.total();
  if (!std::isfinite(result.energy))
    throw std::runtime_error(std::string("nonfinite ") + method_name + " RKS energy");
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

  if (!plan.matches(grid.system(), nullptr, strategy, -1, 0) ||
      basis.packed != dft::AoBasis(system).packed)
    throw std::invalid_argument("RKS refuses a stale geometry, basis, charge or spin binding");

  const std::size_t n = ints.nbf;
  const std::size_t occupied = static_cast<std::size_t>(system.electron_count / 2);
  if (occupied > n) throw std::runtime_error("basis has fewer orbitals than occupied pairs");
  const Matrix orthogonalizer = symmetric_orthogonalizer(ints.overlap, n);
  std::optional<EigenResult> initial_orbitals;
  Matrix density = prepare_initial_density(system, ints, orthogonalizer, occupied, initial_density,
                                           initial_orbitals);
  // Only a cold seed carries a core frame. Warm consumers solve their first
  // target Fock before reading orbitals; RKS packs its initial factor cold-only.
  EigenResult orbitals = std::move(initial_orbitals).value_or(EigenResult{});
  if (options.strict_initial_density && initial_density) {
    validate_seed(ints.overlap, *initial_density, n, {static_cast<unsigned>(system.electron_count)},
                  2.0);
    density = *initial_density;
  }
  Diis diis(options.diis_history, true);
  ScfResult result;
  result.initial_density_used = initial_density != nullptr;
  auto& ks = result.dft_diagnostic;
  ks.occupations = {occupied, occupied};
  ks.grid_points = grid.point_count();
  ks.tile_points = std::min(options.xc_tile_points, grid.point_count());
  ks.ao_order = std::string_view(method_name) == "PBE" ? 1 : 0;
  auto& diagnostic = result.xc_density_diagnostic;
  diagnostic.physical_residual = std::numeric_limits<double>::infinity();
  std::shared_ptr<const OccupiedDensityFactor> factor;
  DensityFactorIdentity identity{};
  const bool use_orbitals = options.xc_density_route == dft::XcDensityRoute::OccupiedOrbitals;
  if (use_orbitals) {
    const auto owner = next_rks_identity();
    identity = {owner, owner, 0, 0};
  }
  const auto retained_capacity = [&](const Matrix& current_density) {
    return runtime::add_capacity(
        runtime::add_capacity(plan.cpu_observation_capacity(), diis.numeric_capacity()),
        runtime::add_capacity(
            factor ? factor->numeric_capacity_bytes() : 0,
            runtime::vector_capacities(orthogonalizer, current_density, orbitals.values,
                                       orbitals.vectors, basis.packed, grid.points(),
                                       grid.weights(), grid.owners(), ks.history)));
  };
  const auto make_current_factor = [&](const Matrix& current_density,
                                       std::size_t extra_live_bytes = 0) {
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
    runtime::sample_cpu_capacity(
        runtime::add_capacity(retained_capacity(current_density),
                              runtime::add_capacity(packing_bytes, extra_live_bytes)));
  };
  if (use_orbitals && !initial_density) make_current_factor(density);
  const auto evaluate_current = [&](const Matrix& current_density,
                                    std::size_t extra_live_bytes = 0) {
    auto physical =
        evaluate_rks(plan, basis, grid, current_density, evaluate_xc, method_name,
                     {options.xc_density_route, factor.get(), identity},
                     runtime::add_capacity(retained_capacity(current_density), extra_live_bytes),
                     options.xc_tile_points);
    const auto& record = physical.density_diagnostic;
    if (record.executed == dft::XcDensityRoute::OccupiedOrbitals)
      ++diagnostic.orbital_calls;
    else
      ++diagnostic.density_calls;
    if (record.fallback != dft::XcDensityFallback::None) ++diagnostic.fallback_calls;
    diagnostic.xc_peak_bytes = std::max(diagnostic.xc_peak_bytes, record.owned_numeric_bytes);
    return physical;
  };
  const auto next_density_from_orbitals = [&](const Matrix& current_density,
                                              std::size_t extra_live_bytes = 0) {
    if (use_orbitals) make_current_factor(current_density, extra_live_bytes);
    // Reuse the producer's exact witness rather than reconstructing D twice.
    Matrix next = use_orbitals ? Matrix(factor->density().begin(), factor->density().end())
                               : density_from_orbitals(orbitals.vectors, n, occupied);
    runtime::sample_cpu_capacity(runtime::add_capacity(
        retained_capacity(current_density),
        runtime::add_capacity(runtime::vector_bytes(next), extra_live_bytes)));
    return next;
  };
  const auto retain_factor = [&] {
    result.xc_density_factor = factor;
    diagnostic.final_identity = identity;
  };
  struct RksLoopEvaluation {
    std::shared_ptr<const OccupiedDensityFactor> current_factor;
    DensityFactorIdentity current_identity{};
    dft::EnergyComponents components;
    Matrix next_density;
    double energy{};
    double state_rms{};
    double residual_rms{};
    double spin_electrons{};
  };

  const solver::SelfConsistentPolicy policy{options.max_iterations, options.energy_tolerance,
                                            options.density_tolerance,
                                            std::min(1.0e-9, options.density_tolerance), true};
  auto outcome = solver::run_self_consistent(
      std::move(density), policy,
      [&](const Matrix& current_density, unsigned) {
        const auto current_factor = factor;
        const auto current_identity = identity;
        ++result.fock_builds;
        const auto physical = evaluate_current(current_density);
        const Matrix residual =
            commutator_residual(physical.fock, current_density, ints.overlap, n);
        const Matrix effective_fock = diis.update(physical.fock, residual);
        orbitals = generalized_eigen(effective_fock, orthogonalizer, n);
        const auto iteration_bytes =
            runtime::vector_capacities(physical.fock, residual, effective_fock);
        Matrix next_density = next_density_from_orbitals(current_density, iteration_bytes);

        runtime::sample_cpu_capacity(runtime::add_capacity(
            retained_capacity(current_density),
            runtime::add_capacity(iteration_bytes, runtime::vector_bytes(next_density))));
        const double state_rms = density_rms(next_density, current_density);
        const double physical_residual = residual_rms(residual);
        const double spin_electrons = dot(current_density, ints.overlap) / 2.0;
        return RksLoopEvaluation{current_factor,          current_identity, physical.components,
                                 std::move(next_density), physical.energy,  state_rms,
                                 physical_residual,       spin_electrons};
      },
      [&](Matrix& current_density, RksLoopEvaluation evaluation,
          const solver::SelfConsistentProgress& progress) {
        if (!progress.converged && progress.iteration == options.max_iterations) {
          // A failed return must keep E/residual/D/factor on the same physical
          // generation rather than publishing the last unchecked proposal.
          factor = std::move(evaluation.current_factor);
          identity = evaluation.current_identity;
          return std::move(current_density);
        }
        return std::move(evaluation.next_density);
      },
      [&](const solver::SelfConsistentProgress& progress, const RksLoopEvaluation& evaluation) {
        result.iterations = progress.iteration;
        result.energy = progress.energy;
        result.energy_change = progress.energy_change;
        result.density_rms = progress.state_rms;
        ks.physical_residual = progress.residual_rms;
        result.physical_residual_rms = ks.physical_residual;
        ks.components = evaluation.components;
        ks.electrons = {evaluation.spin_electrons, evaluation.spin_electrons};
        ks.density_change = progress.state_rms;
        ks.history.push_back({progress.iteration,
                              evaluation.components,
                              progress.energy_change,
                              progress.state_rms,
                              progress.residual_rms,
                              {evaluation.spin_electrons, evaluation.spin_electrons}});
      });
  density = std::move(outcome.state);
  result.converged = outcome.converged;

  if (!result.converged) {
    diagnostic.physical_residual = ks.physical_residual;
    retain_factor();
    result.density = std::move(density);
    return result;
  }

  result.fock_builds += 2;
  auto final = evaluate_current(density);
  orbitals = generalized_eigen(final.fock, orthogonalizer, n);
  Matrix projected = next_density_from_orbitals(density, runtime::vector_bytes(final.fock));
  result.density_rms = density_rms(projected, density);
  density = std::move(projected);
  final = evaluate_current(density, runtime::vector_bytes(final.fock));
  const auto final_residual = commutator_residual(final.fock, density, ints.overlap, n);
  diagnostic.physical_residual = residual_rms(final_residual);
  ks.physical_residual = diagnostic.physical_residual;
  result.physical_residual_rms = ks.physical_residual;
  ks.components = final.components;
  const double final_spin_electrons = dot(density, ints.overlap) / 2.0;
  ks.electrons = {final_spin_electrons, final_spin_electrons};
  ks.density_change = result.density_rms;
  result.energy_change = std::abs(final.energy - result.energy);
  result.converged = result.energy_change < options.energy_tolerance &&
                     result.density_rms < options.density_tolerance &&
                     ks.physical_residual < std::min(1.0e-9, options.density_tolerance);
  runtime::sample_cpu_capacity(runtime::add_capacity(
      retained_capacity(density), runtime::vector_capacities(final.fock, final_residual)));
  retain_factor();
  result.energy = final.energy;
  if (result.converged && options.retain_ks_state) result.ks_physical_fock = std::move(final.fock);
  result.density = std::move(density);
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

}  // namespace vibeqc::scf
