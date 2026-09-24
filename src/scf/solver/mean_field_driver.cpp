#include "scf/solver/mean_field_driver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "runtime/host_component_trace.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/gradient/hf_gradient.hpp"
#include "scf/initial_guess/density.hpp"
#include "scf/reference/mean_field.hpp"
#include "scf/solver/diis.hpp"
#include "scf/solver/proposal_control.hpp"
#include "solver/self_consistent.hpp"

namespace vibeqc::scf::solver {
namespace {
using initial_guess::prepare_initial_density;
using initial_guess::prepare_initial_uhf_density;
using initial_guess::spin_occupations;
using reference::commutator_residual;
using reference::concatenate;
using reference::density_from_orbitals;
using reference::density_rms;
using reference::EigenResult;
using reference::electronic_energy;
using reference::energy_weighted_density;
using reference::generalized_eigen;
using reference::Matrix;
using reference::residual_rms;
using reference::split_spin_matrices;
using reference::uhf_electronic_energy;

template <class... Vectors>
void sample_scf_buffers(const PreparedFockPlan& plan, const Diis& diis,
                        const Vectors&... vectors) noexcept {
  if (!runtime::cpu_resource_observation.active) return;
  runtime::sample_cpu_capacity(runtime::add_capacity(
      plan.cpu_observation_capacity(),
      runtime::add_capacity(diis.numeric_capacity(), runtime::vector_capacities(vectors...))));
}

bool incremental_direct_jk_eligible(const ResolvedFockBuild& strategy) {
  const auto exact = [](const FockTermSpec& term) {
    return !term.present || term.approximation == FockApproximation::Exact;
  };
  return (strategy.spec.coulomb.present || strategy.spec.exchange.present) &&
         exact(strategy.spec.coulomb) && exact(strategy.spec.exchange);
}

void add_jk_in_place(DirectJkMatrices& target, const DirectJkMatrices& base) {
  if (target.nbf != base.nbf) throw std::logic_error("incremental J/K AO dimension mismatch");
  const auto add = [](std::vector<double>& values, const std::vector<double>& anchor) {
    if (values.size() != anchor.size())
      throw std::logic_error("incremental J/K selected-output mismatch");
    for (std::size_t i = 0; i < values.size(); ++i) values[i] += anchor[i];
  };
  add(target.coulomb, base.coulomb);
  add(target.exchange_alpha, base.exchange_alpha);
  add(target.exchange_beta, base.exchange_beta);
}

/** Exact accepted-iterate incremental J/K baseline for #990.
 *
 * Only the main SCF trajectory mutates the anchor. Proposal/audit builds and
 * final physical-state rebuilds use PreparedFockPlan directly, so a rejected
 * trial can never become the next anchor. This slice intentionally performs no
 * density-weighted quartet skipping yet: it establishes exact delta-D parity,
 * bounded periodic full refresh, and trustworthy work counters first.
 */
class ExactIncrementalDirectJk {
 public:
  ExactIncrementalDirectJk(const PreparedFockPlan& plan, const ScfOptions& options,
                           IncrementalDirectJkDiagnostic& diagnostic)
      : plan_(plan),
        diagnostic_(diagnostic),
        rebuild_interval_(options.incremental_direct_jk_rebuild_interval) {
    diagnostic_.requested = options.incremental_direct_jk;
    diagnostic_.active =
        options.incremental_direct_jk && incremental_direct_jk_eligible(plan.strategy());
  }

  DirectJkMatrices build(const Matrix& density, const Matrix& beta = {}) {
    if (!diagnostic_.active) return plan_.build(density, beta);
    const runtime::CpuRetainedCapacity anchor_capacity(numeric_capacity());
    if (!anchored_ || (rebuild_interval_ != 0 && delta_updates_since_full_ >= rebuild_interval_)) {
      const bool refresh = anchored_;
      auto current = plan_.build(density, beta);
      anchor_density_ = density;
      anchor_beta_ = beta;
      anchor_jk_ = current;
      anchored_ = true;
      delta_updates_since_full_ = 0;
      ++diagnostic_.anchor_full_builds;
      if (refresh) ++diagnostic_.periodic_rebuilds;
      return current;
    }
    if (density.size() != anchor_density_.size() || beta.size() != anchor_beta_.size())
      throw std::logic_error("incremental J/K density shape changed within one SCF solve");
    Matrix delta(density.size());
    Matrix delta_beta(beta.size());
    const runtime::CpuRetainedCapacity delta_capacity(
        runtime::vector_capacities(delta, delta_beta));
    for (std::size_t i = 0; i < density.size(); ++i) {
      delta[i] = density[i] - anchor_density_[i];
      diagnostic_.max_abs_delta_density =
          std::max(diagnostic_.max_abs_delta_density, std::abs(delta[i]));
    }
    for (std::size_t i = 0; i < beta.size(); ++i) {
      delta_beta[i] = beta[i] - anchor_beta_[i];
      diagnostic_.max_abs_delta_density =
          std::max(diagnostic_.max_abs_delta_density, std::abs(delta_beta[i]));
    }
    auto current = plan_.build(delta, delta_beta);
    add_jk_in_place(current, anchor_jk_);
    anchor_density_ = density;
    anchor_beta_ = beta;
    anchor_jk_ = current;
    ++delta_updates_since_full_;
    ++diagnostic_.delta_builds;
    ++diagnostic_.anchor_updates;
    return current;
  }

  std::size_t numeric_capacity() const noexcept {
    return runtime::vector_capacities(anchor_density_, anchor_beta_, anchor_jk_.coulomb,
                                      anchor_jk_.exchange_alpha, anchor_jk_.exchange_beta);
  }

  void note_bypass_full_build() noexcept {
    if (diagnostic_.active) ++diagnostic_.bypass_full_builds;
  }
  void note_post_scf_full_builds(std::uint64_t count) noexcept {
    if (diagnostic_.active) diagnostic_.post_scf_full_builds += count;
  }

 private:
  const PreparedFockPlan& plan_;
  IncrementalDirectJkDiagnostic& diagnostic_;
  unsigned rebuild_interval_{};
  bool anchored_{};
  unsigned delta_updates_since_full_{};
  Matrix anchor_density_;
  Matrix anchor_beta_;
  DirectJkMatrices anchor_jk_;
};

std::pair<Matrix, Matrix> build_uhf_focks(const PreparedFockPlan& plan, const Matrix& hcore,
                                          const Matrix& alpha_density, const Matrix& beta_density) {
  auto fock = assemble_fock(plan.strategy(), hcore, plan.build(alpha_density, beta_density));
  return {std::move(fock.alpha), std::move(fock.beta)};
}

Matrix build_fock(const PreparedFockPlan& plan, const Matrix& hcore, const Matrix& density) {
  return assemble_fock(plan.strategy(), hcore, plan.build(density)).alpha;
}

/** Substitute only the provider of this actual matrix. Iterative matrices may
 * be DIIS-extrapolated, so their frames never authorize physical-state reuse. */
EigenResult diagonalize(const PreparedFockPlan& plan, const Matrix& matrix,
                        const integrals::IntegralData& ints, const Matrix& orthogonalizer,
                        PreparedFockPlan::EigenUse use) {
  namespace trace = runtime::host_trace;
  trace::Reason reason(use == PreparedFockPlan::EigenUse::Iteration
                           ? trace::EigenReason::iteration
                           : trace::EigenReason::final_fock);
  const auto operation = plan.eigen_operation(use);
  return operation ? operation(matrix, &ints.overlap, &orthogonalizer, ints.nbf)
                   : generalized_eigen(matrix, orthogonalizer, ints.nbf);
}

void finalize_scf(const PreparedFockPlan& plan, const integrals::IntegralData& ints,
                  const Matrix& orthogonalizer, std::size_t occupied, Matrix& density,
                  bool compute_forces, ScfResult& result) {
  const std::size_t n = ints.nbf;
  runtime::host_trace::Region final_trace("host_finalization", n);
  Matrix final_fock = build_fock(plan, ints.hcore, density);
  EigenResult orbitals =
      diagonalize(plan, final_fock, ints, orthogonalizer, PreparedFockPlan::EigenUse::Finalization);
  density = density_from_orbitals(orbitals.vectors, n, occupied);
  final_fock = build_fock(plan, ints.hcore, density);
  result.energy = electronic_energy(density, ints.hcore, final_fock) + ints.nuclear_repulsion;
  if (compute_forces) {
    const Matrix weighted = energy_weighted_density(orbitals.vectors, orbitals.values, n, occupied);
    result.forces =
        gradient::analytic_forces(ints, density, weighted, plan.energy_derivative(density));
  }
  result.density = density;
}

void finalize_uhf(const PreparedFockPlan& plan, const integrals::IntegralData& ints,
                  const Matrix& orthogonalizer, std::size_t alpha_occupied,
                  std::size_t beta_occupied, Matrix& alpha_density, Matrix& beta_density,
                  bool compute_forces, ScfResult& result) {
  const std::size_t n = ints.nbf;
  runtime::host_trace::Region final_trace("host_finalization", n);
  auto [alpha_fock, beta_fock] = build_uhf_focks(plan, ints.hcore, alpha_density, beta_density);
  EigenResult alpha_orbitals =
      diagonalize(plan, alpha_fock, ints, orthogonalizer, PreparedFockPlan::EigenUse::Finalization);
  EigenResult beta_orbitals =
      diagonalize(plan, beta_fock, ints, orthogonalizer, PreparedFockPlan::EigenUse::Finalization);
  alpha_density = density_from_orbitals(alpha_orbitals.vectors, n, alpha_occupied, 1.0);
  beta_density = density_from_orbitals(beta_orbitals.vectors, n, beta_occupied, 1.0);
  std::tie(alpha_fock, beta_fock) = build_uhf_focks(plan, ints.hcore, alpha_density, beta_density);
  result.energy =
      uhf_electronic_energy(alpha_density, beta_density, ints.hcore, alpha_fock, beta_fock) +
      ints.nuclear_repulsion;
  if (compute_forces) {
    const Matrix alpha_weighted = energy_weighted_density(
        alpha_orbitals.vectors, alpha_orbitals.values, n, alpha_occupied, 1.0);
    const Matrix beta_weighted =
        energy_weighted_density(beta_orbitals.vectors, beta_orbitals.values, n, beta_occupied, 1.0);
    result.forces = gradient::analytic_uhf_forces(
        ints, alpha_density, beta_density, alpha_weighted, beta_weighted,
        plan.energy_derivative(alpha_density, beta_density));
  }
  result.density = concatenate(alpha_density, beta_density);
}

}  // namespace

ScfResult run_rhf_host_plan(const core::System& system, const ScfOptions& options,
                            const integrals::IntegralData& ints, const PreparedFockPlan& plan,
                            const std::vector<double>* initial_density,
                            initial_guess::OverlapOrthogonalizer* overlap_cache) {
  const std::size_t n = ints.nbf;
  const std::size_t occupied = static_cast<std::size_t>(system.electron_count / 2);
  if (occupied > n) {
    throw std::runtime_error("basis has fewer orbitals than occupied electron pairs");
  }
  const Matrix orthogonalizer = plan.overlap_orthogonalizer(overlap_cache);
  std::optional<EigenResult> initial_orbitals;
  Matrix density = prepare_initial_density(system, ints, orthogonalizer, occupied, initial_density,
                                           initial_orbitals,
                                           initial_guess::InitialOrbitalRequest::ColdDensityOnly,
                                           plan.eigen_operation(PreparedFockPlan::EigenUse::Setup));
  // Only a cold seed carries a core frame. Warm consumers solve their first
  // target Fock before reading orbitals; RKS packs its initial factor cold-only.
  EigenResult orbitals = std::move(initial_orbitals).value_or(EigenResult{});
  if (options.strict_initial_density && initial_density) {
    validate_seed(ints.overlap, *initial_density, n, {static_cast<unsigned>(system.electron_count)},
                  2.0, plan.eigen_operation(PreparedFockPlan::EigenUse::Setup));
    density = *initial_density;
  }
  Diis diis(options.diis_history);
  const auto generation = new_scf_generation(options);
  unsigned proposal_failures = 0;

  ScfResult result;
  result.initial_density_used = initial_density != nullptr;
  ExactIncrementalDirectJk incremental_jk(plan, options, result.incremental_direct_jk);
  const bool require_residual = options.hooks && options.hooks->propose;

  struct RhfEvaluation {
    Matrix fock;
    Matrix residual;
    Matrix next_density;
    double energy{};
    double state_rms{};
    double residual_rms{};
  };

  const ::vibeqc::solver::SelfConsistentPolicy policy{
      options.max_iterations, options.energy_tolerance, options.density_tolerance,
      options.density_tolerance, require_residual};
  auto outcome = ::vibeqc::solver::run_self_consistent(
      std::move(density), policy,
      [&](const Matrix& current_density, unsigned) {
        ++result.fock_builds;
        Matrix fock =
            assemble_fock(plan.strategy(), ints.hcore, incremental_jk.build(current_density)).alpha;
        const runtime::CpuRetainedCapacity anchor_capacity(incremental_jk.numeric_capacity());
        const double energy =
            electronic_energy(current_density, ints.hcore, fock) + ints.nuclear_repulsion;
        Matrix residual = commutator_residual(fock, current_density, ints.overlap, n);
        const Matrix effective_fock = diis.update(fock, residual);
        orbitals = diagonalize(plan, effective_fock, ints, orthogonalizer,
                               PreparedFockPlan::EigenUse::Iteration);
        Matrix next_density = density_from_orbitals(orbitals.vectors, n, occupied);

        sample_scf_buffers(plan, diis, orthogonalizer, current_density, fock, residual,
                           effective_fock, orbitals.values, orbitals.vectors, next_density);

        const double state_rms = density_rms(next_density, current_density);
        const double physical_residual_rms = require_residual ? residual_rms(residual) : 0.0;
        return RhfEvaluation{std::move(fock), std::move(residual), std::move(next_density),
                             energy,          state_rms,           physical_residual_rms};
      },
      [&](const Matrix& current_density, RhfEvaluation evaluation,
          const ::vibeqc::solver::SelfConsistentProgress& progress) {
        Matrix next_density = std::move(evaluation.next_density);
        if (options.hooks) {
          next_density = safeguarded_update(
              options, generation, progress.iteration, ints.overlap, current_density,
              evaluation.fock, evaluation.residual, std::move(next_density),
              {static_cast<unsigned>(system.electron_count)}, 2.0, result, diis, proposal_failures,
              progress.converged, [&](const Matrix& trial) {
                const runtime::CpuRetainedCapacity anchor_capacity(
                    incremental_jk.numeric_capacity());
                incremental_jk.note_bypass_full_build();
                const Matrix trial_fock = build_fock(plan, ints.hcore, trial);
                return std::make_pair(
                    electronic_energy(trial, ints.hcore, trial_fock) + ints.nuclear_repulsion,
                    commutator_residual(trial_fock, trial, ints.overlap, n));
              });
        }
        return next_density;
      },
      [&](const ::vibeqc::solver::SelfConsistentProgress& progress, const RhfEvaluation&) {
        result.iterations = progress.iteration;
        result.energy = progress.energy;
        result.energy_change = progress.energy_change;
        result.density_rms = progress.state_rms;
      });
  density = std::move(outcome.state);
  result.converged = outcome.converged;

  if (!result.converged) {
    // Failed traces retain the last iterate, never a converged reference.
    result.density = density;
    return result;
  }
  result.fock_builds += 2;  // Physical rebuilds performed by finalization.
  incremental_jk.note_post_scf_full_builds(2);
  const runtime::CpuRetainedCapacity anchor_capacity(incremental_jk.numeric_capacity());

  // Rebuild and diagonalize the un-extrapolated converged Fock matrix. The
  // resulting orbitals define the energy-weighted density in the Pulay term.
  finalize_scf(plan, ints, orthogonalizer, occupied, density, options.compute_forces, result);
  return result;
}

ScfResult run_uhf_host_plan(const core::System& system, const ScfOptions& options,
                            const integrals::IntegralData& ints, const PreparedFockPlan& plan,
                            const std::vector<double>* initial_density,
                            initial_guess::OverlapOrthogonalizer* overlap_cache) {
  const std::size_t n = ints.nbf;
  const auto [alpha_occupied, beta_occupied] = spin_occupations(system);
  if (alpha_occupied > n || beta_occupied > n) {
    throw std::runtime_error("basis has fewer orbitals than required UHF spin occupations");
  }
  const Matrix orthogonalizer = plan.overlap_orthogonalizer(overlap_cache);
  std::optional<EigenResult> initial_alpha, initial_beta;
  auto [alpha_density, beta_density] = prepare_initial_uhf_density(
      ints, orthogonalizer, alpha_occupied, beta_occupied, initial_density, initial_alpha,
      initial_beta, initial_guess::InitialOrbitalRequest::ColdDensityOnly,
      plan.eigen_operation(PreparedFockPlan::EigenUse::Setup));
  EigenResult alpha_orbitals = std::move(initial_alpha).value_or(EigenResult{});
  EigenResult beta_orbitals = std::move(initial_beta).value_or(EigenResult{});
  if (options.strict_initial_density && initial_density) {
    validate_seed(ints.overlap, *initial_density, n,
                  {static_cast<unsigned>(alpha_occupied), static_cast<unsigned>(beta_occupied)},
                  1.0, plan.eigen_operation(PreparedFockPlan::EigenUse::Setup));
    std::tie(alpha_density, beta_density) = split_spin_matrices(*initial_density, n * n);
  }
  Diis diis(options.diis_history);
  const auto generation = new_scf_generation(options);
  unsigned proposal_failures = 0;

  ScfResult result;
  result.initial_density_used = initial_density != nullptr;
  ExactIncrementalDirectJk incremental_jk(plan, options, result.incremental_direct_jk);
  const bool require_residual = options.hooks && options.hooks->propose;

  struct UhfState {
    Matrix alpha;
    Matrix beta;
  };
  struct UhfEvaluation {
    Matrix physical_fock;
    Matrix physical_residual;
    Matrix next_alpha;
    Matrix next_beta;
    double energy{};
    double state_rms{};
    double residual_rms{};
  };

  const ::vibeqc::solver::SelfConsistentPolicy policy{options.max_iterations, options.energy_tolerance,
                                    options.density_tolerance, options.density_tolerance,
                                    require_residual};
  auto outcome = ::vibeqc::solver::run_self_consistent(
      UhfState{std::move(alpha_density), std::move(beta_density)}, policy,
      [&](const UhfState& state, unsigned) {
        auto fock = assemble_fock(plan.strategy(), ints.hcore,
                                  incremental_jk.build(state.alpha, state.beta));
        const runtime::CpuRetainedCapacity anchor_capacity(incremental_jk.numeric_capacity());
        auto alpha_fock = std::move(fock.alpha);
        auto beta_fock = std::move(fock.beta);
        ++result.fock_builds;
        const double energy =
            uhf_electronic_energy(state.alpha, state.beta, ints.hcore, alpha_fock, beta_fock) +
            ints.nuclear_repulsion;
        const Matrix alpha_residual = commutator_residual(alpha_fock, state.alpha, ints.overlap, n);
        const Matrix beta_residual = commutator_residual(beta_fock, state.beta, ints.overlap, n);
        Matrix physical_fock = concatenate(alpha_fock, beta_fock);
        Matrix physical_residual = concatenate(alpha_residual, beta_residual);
        const Matrix effective_joined = diis.update(physical_fock, physical_residual);
        std::tie(alpha_fock, beta_fock) = split_spin_matrices(effective_joined, n * n);
        alpha_orbitals = diagonalize(plan, alpha_fock, ints, orthogonalizer,
                                     PreparedFockPlan::EigenUse::Iteration);
        beta_orbitals = diagonalize(plan, beta_fock, ints, orthogonalizer,
                                    PreparedFockPlan::EigenUse::Iteration);
        Matrix next_alpha = density_from_orbitals(alpha_orbitals.vectors, n, alpha_occupied, 1.0);
        Matrix next_beta = density_from_orbitals(beta_orbitals.vectors, n, beta_occupied, 1.0);

        sample_scf_buffers(plan, diis, orthogonalizer, state.alpha, state.beta, alpha_fock,
                           beta_fock, alpha_residual, beta_residual, physical_fock,
                           physical_residual, effective_joined, alpha_orbitals.values,
                           alpha_orbitals.vectors, beta_orbitals.values, beta_orbitals.vectors,
                           next_alpha, next_beta);

        const double state_rms =
            density_rms(concatenate(next_alpha, next_beta), concatenate(state.alpha, state.beta));
        const double physical_residual_rms =
            require_residual ? residual_rms(physical_residual) : 0.0;
        return UhfEvaluation{std::move(physical_fock),
                             std::move(physical_residual),
                             std::move(next_alpha),
                             std::move(next_beta),
                             energy,
                             state_rms,
                             physical_residual_rms};
      },
      [&](const UhfState& state, UhfEvaluation evaluation,
          const ::vibeqc::solver::SelfConsistentProgress& progress) {
        if (!options.hooks) {
          return UhfState{std::move(evaluation.next_alpha), std::move(evaluation.next_beta)};
        }

        const Matrix next = safeguarded_update(
            options, generation, progress.iteration, ints.overlap,
            concatenate(state.alpha, state.beta), evaluation.physical_fock,
            evaluation.physical_residual, concatenate(evaluation.next_alpha, evaluation.next_beta),
            {static_cast<unsigned>(alpha_occupied), static_cast<unsigned>(beta_occupied)}, 1.0,
            result, diis, proposal_failures, progress.converged, [&](const Matrix& trial) {
              const auto [a, b] = split_spin_matrices(trial, n * n);
              const runtime::CpuRetainedCapacity anchor_capacity(incremental_jk.numeric_capacity());
              incremental_jk.note_bypass_full_build();
              const auto [fa, fb] = build_uhf_focks(plan, ints.hcore, a, b);
              return std::make_pair(
                  uhf_electronic_energy(a, b, ints.hcore, fa, fb) + ints.nuclear_repulsion,
                  concatenate(commutator_residual(fa, a, ints.overlap, n),
                              commutator_residual(fb, b, ints.overlap, n)));
            });
        auto [next_alpha, next_beta] = split_spin_matrices(next, n * n);
        return UhfState{std::move(next_alpha), std::move(next_beta)};
      },
      [&](const ::vibeqc::solver::SelfConsistentProgress& progress, const UhfEvaluation&) {
        result.iterations = progress.iteration;
        result.energy = progress.energy;
        result.energy_change = progress.energy_change;
        result.density_rms = progress.state_rms;
      });
  alpha_density = std::move(outcome.state.alpha);
  beta_density = std::move(outcome.state.beta);
  result.converged = outcome.converged;

  if (!result.converged) {
    // Failed traces retain the last iterate, never a converged reference.
    result.density = concatenate(alpha_density, beta_density);
    return result;
  }
  result.fock_builds += 2;  // Physical rebuilds performed by finalization.
  incremental_jk.note_post_scf_full_builds(2);
  const runtime::CpuRetainedCapacity anchor_capacity(incremental_jk.numeric_capacity());

  // As in RHF, rebuild from the un-extrapolated converged spin Fock matrices
  // before forming orbital-weighted Pulay densities and analytic forces.
  finalize_uhf(plan, ints, orthogonalizer, alpha_occupied, beta_occupied, alpha_density,
               beta_density, options.compute_forces, result);
  return result;
}

}  // namespace vibeqc::scf::solver
