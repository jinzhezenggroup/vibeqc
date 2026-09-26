#include "dft/cosx_scf.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "scf/gradient/hf_gradient.hpp"
#include "scf/initial_guess/density.hpp"
#include "scf/reference/linalg.hpp"
#include "scf/reference/mean_field.hpp"
#include "scf/solver/diis.hpp"
#include "scf/solver/proposal_control.hpp"

namespace vibeqc::dft {
namespace {

using scf::reference::commutator_residual;
using scf::reference::concatenate;
using scf::reference::density_from_orbitals;
using scf::reference::density_rms;
using scf::reference::EigenResult;
using scf::reference::electronic_energy;
using scf::reference::energy_weighted_density;
using scf::reference::generalized_eigen;
using scf::reference::Matrix;
using scf::reference::residual_rms;
using scf::reference::split_spin_matrices;
using scf::reference::symmetric_orthogonalizer;
using scf::reference::uhf_electronic_energy;

void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}

void validate_options(const PreparedCosxFockPlan& plan, const scf::ScfOptions& options,
                      scf::FockSpin spin) {
  const auto& strategy = plan.strategy();
  scf::validate_resolved_fock_build(strategy);
  require(strategy.backend == scf::FockBackend::Cuda &&
              strategy.schedule == scf::FockSchedule::CudaIndependent &&
              strategy.spec.spin == spin &&
              strategy.spec.derivative_order == (options.compute_forces ? 1U : 0U) &&
              strategy.spec.exchange.present &&
              strategy.spec.exchange.approximation == scf::FockApproximation::SeminumericalCosx,
          "COSX SCF requires a matching value/force prepared CUDA exchange strategy");
  require(options.hooks == nullptr, "COSX SCF proposal hooks are not implemented");
  if (options.resolved_fock_build)
    require(*options.resolved_fock_build == strategy,
            "COSX SCF options do not match the prepared Fock strategy");
  require(options.max_iterations > 0 && options.diis_history > 0 &&
              std::isfinite(options.energy_tolerance) && options.energy_tolerance > 0.0 &&
              std::isfinite(options.density_tolerance) && options.density_tolerance > 0.0,
          "invalid COSX SCF convergence controls");
}

Matrix rhf_fock(PreparedCosxFockPlan& plan, const Matrix& density) {
  return scf::assemble_fock(plan.strategy(), plan.one_electron().hcore, plan.build(density)).alpha;
}

std::pair<Matrix, Matrix> uhf_focks(PreparedCosxFockPlan& plan, const Matrix& alpha,
                                    const Matrix& beta) {
  auto fock =
      scf::assemble_fock(plan.strategy(), plan.one_electron().hcore, plan.build(alpha, beta));
  return {std::move(fock.alpha), std::move(fock.beta)};
}

double residual_gate(const scf::ScfOptions& options) {
  return std::min(1.0e-9, options.density_tolerance);
}

void finalize_rhf(PreparedCosxFockPlan& plan, const scf::ScfOptions& options, std::size_t occupied,
                  const Matrix& x, Matrix& density, scf::ScfResult& result) {
  const auto& ints = plan.one_electron();
  const auto n = ints.nbf;
  Matrix fock = rhf_fock(plan, density);
  EigenResult orbitals = generalized_eigen(fock, x, n);
  Matrix projected = density_from_orbitals(orbitals.vectors, n, occupied);
  result.density_rms = density_rms(projected, density);
  density = std::move(projected);
  fock = rhf_fock(plan, density);
  const double final_energy = electronic_energy(density, ints.hcore, fock) + ints.nuclear_repulsion;
  const auto residual = commutator_residual(fock, density, ints.overlap, n);
  result.physical_residual_rms = residual_rms(residual);
  result.energy_change = std::abs(final_energy - result.energy);
  result.energy = final_energy;
  if (options.compute_forces) {
    const Matrix weighted = energy_weighted_density(orbitals.vectors, orbitals.values, n, occupied);
    result.forces =
        scf::gradient::analytic_forces(ints, density, weighted, plan.energy_derivative(density));
  }
  result.converged = result.energy_change < options.energy_tolerance &&
                     result.density_rms < options.density_tolerance &&
                     result.physical_residual_rms < residual_gate(options);
  result.density = density;
}

void finalize_uhf(PreparedCosxFockPlan& plan, const scf::ScfOptions& options,
                  std::size_t alpha_occupied, std::size_t beta_occupied, const Matrix& x,
                  Matrix& alpha, Matrix& beta, scf::ScfResult& result) {
  const auto& ints = plan.one_electron();
  const auto n = ints.nbf;
  auto [fa, fb] = uhf_focks(plan, alpha, beta);
  const auto ca = generalized_eigen(fa, x, n);
  const auto cb = generalized_eigen(fb, x, n);
  Matrix projected_a = density_from_orbitals(ca.vectors, n, alpha_occupied, 1.0);
  Matrix projected_b = density_from_orbitals(cb.vectors, n, beta_occupied, 1.0);
  result.density_rms = density_rms(concatenate(projected_a, projected_b), concatenate(alpha, beta));
  alpha = std::move(projected_a);
  beta = std::move(projected_b);
  std::tie(fa, fb) = uhf_focks(plan, alpha, beta);
  const double final_energy =
      uhf_electronic_energy(alpha, beta, ints.hcore, fa, fb) + ints.nuclear_repulsion;
  const auto ra = commutator_residual(fa, alpha, ints.overlap, n);
  const auto rb = commutator_residual(fb, beta, ints.overlap, n);
  result.physical_residual_rms = std::hypot(residual_rms(ra), residual_rms(rb)) / std::sqrt(2.0);
  result.energy_change = std::abs(final_energy - result.energy);
  result.energy = final_energy;
  if (options.compute_forces) {
    const Matrix alpha_weighted =
        energy_weighted_density(ca.vectors, ca.values, n, alpha_occupied, 1.0);
    const Matrix beta_weighted =
        energy_weighted_density(cb.vectors, cb.values, n, beta_occupied, 1.0);
    result.forces = scf::gradient::analytic_uhf_forces(
        ints, alpha, beta, alpha_weighted, beta_weighted, plan.energy_derivative(alpha, beta));
  }
  result.converged = result.energy_change < options.energy_tolerance &&
                     result.density_rms < options.density_tolerance &&
                     std::max(residual_rms(ra), residual_rms(rb)) < residual_gate(options);
  result.density = concatenate(alpha, beta);
}

}  // namespace

scf::ScfResult run_cosx_rhf(PreparedCosxFockPlan& plan, const scf::ScfOptions& options,
                            const std::vector<double>* initial_density) {
  validate_options(plan, options, scf::FockSpin::Restricted);
  const auto& system = plan.system();
  const auto& ints = plan.one_electron();
  const std::size_t n = ints.nbf;
  require(system.electron_count > 0 && (system.electron_count & 1) == 0,
          "COSX RHF requires a positive even electron count");
  const std::size_t occupied = static_cast<std::size_t>(system.electron_count / 2);
  require(occupied <= n, "COSX RHF occupations exceed the orbital space");

  const Matrix x = symmetric_orthogonalizer(ints.overlap, n);
  std::optional<EigenResult> initial_orbitals;
  Matrix density = scf::initial_guess::prepare_initial_density(system, ints, x, occupied,
                                                               initial_density, initial_orbitals);
  if (options.strict_initial_density && initial_density) {
    scf::solver::validate_seed(ints.overlap, *initial_density, n,
                               {static_cast<unsigned>(system.electron_count)}, 2.0);
    density = *initial_density;
  }

  scf::solver::Diis diis(options.diis_history);
  scf::ScfResult result;
  result.initial_density_used = initial_density != nullptr;
  double previous_energy = std::numeric_limits<double>::infinity();

  for (unsigned iteration = 1; iteration <= options.max_iterations; ++iteration) {
    ++result.fock_builds;
    const Matrix fock = rhf_fock(plan, density);
    const double energy = electronic_energy(density, ints.hcore, fock) + ints.nuclear_repulsion;
    const Matrix residual = commutator_residual(fock, density, ints.overlap, n);
    const double physical_residual = residual_rms(residual);
    const Matrix effective = diis.update(fock, residual);
    const auto orbitals = generalized_eigen(effective, x, n);
    Matrix next_density = density_from_orbitals(orbitals.vectors, n, occupied);

    result.iterations = iteration;
    result.energy = energy;
    result.energy_change = std::isfinite(previous_energy) ? std::abs(energy - previous_energy)
                                                          : std::numeric_limits<double>::infinity();
    result.density_rms = density_rms(next_density, density);
    result.physical_residual_rms = physical_residual;
    const bool terminal = iteration > 1 && result.energy_change < options.energy_tolerance &&
                          result.density_rms < options.density_tolerance &&
                          physical_residual < residual_gate(options);
    if (terminal) {
      density = std::move(next_density);
      result.converged = true;
      break;
    }
    // The published nonconverged state must be the density whose energy and
    // physical residual were actually evaluated above. Do not advance to an
    // unmeasured proposal when the iteration budget is exhausted.
    if (iteration == options.max_iterations) break;
    previous_energy = energy;
    density = std::move(next_density);
  }

  if (!result.converged) {
    result.density = std::move(density);
    return result;
  }
  result.fock_builds += 2;
  finalize_rhf(plan, options, occupied, x, density, result);
  return result;
}

scf::ScfResult run_cosx_uhf(PreparedCosxFockPlan& plan, const scf::ScfOptions& options,
                            const std::vector<double>* initial_density) {
  validate_options(plan, options, scf::FockSpin::Unrestricted);
  const auto& system = plan.system();
  const auto& ints = plan.one_electron();
  const std::size_t n = ints.nbf;
  const auto [alpha_occupied, beta_occupied] = scf::initial_guess::spin_occupations(system);
  require(alpha_occupied <= n && beta_occupied <= n,
          "COSX UHF occupations exceed the orbital space");

  const Matrix x = symmetric_orthogonalizer(ints.overlap, n);
  std::optional<EigenResult> initial_alpha, initial_beta;
  auto [alpha, beta] = scf::initial_guess::prepare_initial_uhf_density(
      ints, x, alpha_occupied, beta_occupied, initial_density, initial_alpha, initial_beta);
  if (options.strict_initial_density && initial_density) {
    scf::solver::validate_seed(
        ints.overlap, *initial_density, n,
        {static_cast<unsigned>(alpha_occupied), static_cast<unsigned>(beta_occupied)}, 1.0);
    std::tie(alpha, beta) = split_spin_matrices(*initial_density, n * n);
  }

  scf::solver::Diis diis(options.diis_history);
  scf::ScfResult result;
  result.initial_density_used = initial_density != nullptr;
  double previous_energy = std::numeric_limits<double>::infinity();

  for (unsigned iteration = 1; iteration <= options.max_iterations; ++iteration) {
    ++result.fock_builds;
    auto [fa, fb] = uhf_focks(plan, alpha, beta);
    const double energy =
        uhf_electronic_energy(alpha, beta, ints.hcore, fa, fb) + ints.nuclear_repulsion;
    const Matrix ra = commutator_residual(fa, alpha, ints.overlap, n);
    const Matrix rb = commutator_residual(fb, beta, ints.overlap, n);
    const double residual_a = residual_rms(ra), residual_b = residual_rms(rb);
    const Matrix effective = diis.update(concatenate(fa, fb), concatenate(ra, rb));
    std::tie(fa, fb) = split_spin_matrices(effective, n * n);
    const auto ca = generalized_eigen(fa, x, n);
    const auto cb = generalized_eigen(fb, x, n);
    Matrix next_alpha = density_from_orbitals(ca.vectors, n, alpha_occupied, 1.0);
    Matrix next_beta = density_from_orbitals(cb.vectors, n, beta_occupied, 1.0);

    result.iterations = iteration;
    result.energy = energy;
    result.energy_change = std::isfinite(previous_energy) ? std::abs(energy - previous_energy)
                                                          : std::numeric_limits<double>::infinity();
    result.density_rms = density_rms(concatenate(next_alpha, next_beta), concatenate(alpha, beta));
    result.physical_residual_rms = std::hypot(residual_a, residual_b) / std::sqrt(2.0);
    const bool terminal = iteration > 1 && result.energy_change < options.energy_tolerance &&
                          result.density_rms < options.density_tolerance &&
                          std::max(residual_a, residual_b) < residual_gate(options);
    if (terminal) {
      alpha = std::move(next_alpha);
      beta = std::move(next_beta);
      result.converged = true;
      break;
    }
    // As in RHF, keep the last evaluated alpha/beta pair on exhaustion so
    // returned density, energy and physical residual describe one state.
    if (iteration == options.max_iterations) break;
    previous_energy = energy;
    alpha = std::move(next_alpha);
    beta = std::move(next_beta);
  }

  if (!result.converged) {
    result.density = concatenate(alpha, beta);
    return result;
  }
  result.fock_builds += 2;
  finalize_uhf(plan, options, alpha_occupied, beta_occupied, x, alpha, beta, result);
  return result;
}

}  // namespace vibeqc::dft
