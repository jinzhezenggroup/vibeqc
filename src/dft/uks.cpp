#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>

#include "dft/ao_grid.hpp"
#include "dft/grid.hpp"
#include "dft/xc.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/initial_guess/density.hpp"
#include "scf/mean_field.hpp"
#include "scf/reference/mean_field.hpp"
#include "scf/solver/diis.hpp"
#include "scf/solver/proposal_control.hpp"

namespace vibeqc::scf {
namespace {
using namespace reference;

struct SpinEvaluation {
  FockMatrices fock;
  dft::EnergyComponents components;
};

/** The physical operator is independent of extrapolation and occupations.
 * The #202 strategy owns J dispatch; semilocal methods never request K. */
SpinEvaluation evaluate(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                        const dft::MolecularGrid& grid, const Matrix& alpha, const Matrix& beta,
                        bool pbe, std::size_t tile) {
  const auto& ints = plan.one_electron();
  const auto jk = plan.build(alpha, beta);
  SpinEvaluation out;
  out.fock = assemble_fock(plan.strategy(), ints.hcore, jk);
  const auto xc = pbe ? dft::integrate_pbe_uks(basis, grid, alpha, beta, tile)
                      : dft::integrate_lda_xc_pw_uks(basis, grid, alpha, beta, tile);
  for (std::size_t i = 0; i < alpha.size(); ++i) {
    out.fock.alpha[i] += xc.potential[0][i];
    out.fock.beta[i] += xc.potential[1][i];
  }
  out.components = {ints.nuclear_repulsion, dot(alpha, ints.hcore) + dot(beta, ints.hcore),
                    contract_fock_energy(plan.strategy(), jk, alpha, beta), xc.energy};
  if (!std::isfinite(out.components.total())) throw std::runtime_error("nonfinite UKS energy");
  return out;
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

}  // namespace

ScfResult run_uks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                  const dft::MolecularGrid& grid, const ScfOptions& options, bool pbe,
                  const std::vector<double>* initial_density) {
  using namespace reference;
  const auto& strategy = plan.strategy();
  validate_resolved_fock_build(strategy);
  if (options.compute_forces) throw std::invalid_argument("UKS gradients require issue #163");
  if (strategy.backend != FockBackend::Cpu || strategy.spec.spin != FockSpin::Unrestricted ||
      strategy.spec.derivative_order != 0 || !strategy.spec.coulomb.present ||
      strategy.spec.coulomb.coefficient != 1.0 || strategy.spec.exchange.present)
    throw std::invalid_argument("UKS requires a CPU Coulomb-only Fock strategy");
  if (options.xc_density_route != dft::XcDensityRoute::DensityMatrix)
    throw std::invalid_argument("UKS occupied-factor XC has not been implemented");
  const auto& system = plan.system();
  const auto& ints = plan.one_electron();
  const std::size_t n = ints.nbf;
  if (basis.nao != n || basis.natom != system.atoms.size() || grid.point_count() == 0)
    throw std::invalid_argument("UKS grid/basis binding is inconsistent");
  if (!plan.matches(grid.system(), nullptr, strategy, -1, 0) ||
      basis.packed != dft::AoBasis(system).packed)
    throw std::invalid_argument("UKS refuses a stale geometry, basis, charge or spin binding");
  const auto [na, nb] = initial_guess::spin_occupations(system);
  if (na > n || nb > n || system.electron_count <= 0)
    throw std::invalid_argument("UKS occupations exceed the orbital space");
  const Matrix x = symmetric_orthogonalizer(ints.overlap, n);
  std::optional<EigenResult> initial_alpha, initial_beta;
  auto [alpha, beta] = initial_guess::prepare_initial_uhf_density(ints, x, na, nb, initial_density,
                                                                  initial_alpha, initial_beta);
  // Warm seeds omit the unused core solve; each spin obtains its first frame
  // from the physical target Fock before the proposal consumes its orbitals.
  EigenResult ca = std::move(initial_alpha).value_or(EigenResult{});
  EigenResult cb = std::move(initial_beta).value_or(EigenResult{});
  if (options.strict_initial_density && initial_density) {
    solver::validate_seed(ints.overlap, *initial_density, n,
                          {static_cast<unsigned>(na), static_cast<unsigned>(nb)}, 1.0);
    std::tie(alpha, beta) = split_spin_matrices(*initial_density, n * n);
  }
  solver::Diis diis(options.diis_history, true);
  ScfResult result;
  result.initial_density_used = initial_density != nullptr;
  auto& diagnostic = result.dft_diagnostic;
  diagnostic.occupations = {na, nb};
  diagnostic.grid_points = grid.point_count();
  diagnostic.tile_points = std::min(options.xc_tile_points, grid.point_count());
  diagnostic.ao_order = pbe ? 1 : 0;
  double previous_energy = std::numeric_limits<double>::infinity();
  const double residual_gate = std::min(1.0e-9, options.density_tolerance);
  bool stabilize_occupations = false;
  for (unsigned iteration = 1; iteration <= options.max_iterations; ++iteration) {
    const auto physical = evaluate(plan, basis, grid, alpha, beta, pbe, options.xc_tile_points);
    ++result.fock_builds;
    const Matrix ra = commutator_residual(physical.fock.alpha, alpha, ints.overlap, n);
    const Matrix rb = commutator_residual(physical.fock.beta, beta, ints.overlap, n);
    const double residual_a = residual_rms(ra), residual_b = residual_rms(rb);
    diagnostic.physical_residual = std::max(residual_a, residual_b);
    // Preserve the public joined-spin RMS while gating each spin separately.
    result.physical_residual_rms = std::hypot(residual_a, residual_b) / std::sqrt(2.0);
    diagnostic.components = physical.components;
    const auto effective = split_spin_matrices(
        diis.update(concatenate(physical.fock.alpha, physical.fock.beta), concatenate(ra, rb)),
        n * n);
    ca = stabilize_occupations ? stabilized_uks_orbitals(effective.first, alpha, ints.overlap, x, n)
                               : generalized_eigen(effective.first, x, n);
    cb = stabilize_occupations ? stabilized_uks_orbitals(effective.second, beta, ints.overlap, x, n)
                               : generalized_eigen(effective.second, x, n);
    Matrix next_a = density_from_orbitals(ca.vectors, n, na, 1.0);
    Matrix next_b = density_from_orbitals(cb.vectors, n, nb, 1.0);
    result.energy = physical.components.total();
    result.iterations = iteration;
    result.energy_change = std::abs(result.energy - previous_energy);
    const double change_a = density_rms(next_a, alpha), change_b = density_rms(next_b, beta);
    const double density_change = std::max(change_a, change_b);
    result.density_rms = std::hypot(change_a, change_b) / std::sqrt(2.0);
    diagnostic.electrons = {dot(alpha, ints.overlap), dot(beta, ints.overlap)};
    diagnostic.density_change = density_change;
    diagnostic.history.push_back({iteration, physical.components, result.energy_change,
                                  density_change, diagnostic.physical_residual,
                                  diagnostic.electrons, stabilize_occupations});
    // Sample actual coexisting capacities. Recurrence, XC tile and solver
    // temporaries have already retired here and remain outside this sample.
    runtime::sample_cpu_capacity(runtime::add_capacity(
        runtime::add_capacity(plan.cpu_observation_capacity(), diis.numeric_capacity()),
        runtime::vector_capacities(basis.packed, grid.points(), grid.weights(), grid.owners(), x,
                                   alpha, beta, ca.values, ca.vectors, cb.values, cb.vectors,
                                   physical.fock.alpha, physical.fock.beta, ra, rb, effective.first,
                                   effective.second, next_a, next_b, diagnostic.history)));
    // Preserve #305's stationary occupation-cycle fix when routing its public
    // wrappers through this driver. Only subsequent proposals are shifted;
    // every physical gate and the integer-occupation density stay unchanged.
    if (iteration > 1 && result.energy_change < options.energy_tolerance &&
        diagnostic.physical_residual < residual_gate && density_change >= options.density_tolerance)
      stabilize_occupations = true;
    if (iteration > 1 && result.energy_change < options.energy_tolerance &&
        density_change < options.density_tolerance &&
        diagnostic.physical_residual < residual_gate) {
      // Return the CURRENT physical state: E, V, residual and D refer to one
      // generation. The DIIS proposal is a convergence check, not final state.
      result.converged = true;
      break;
    }
    if (iteration == options.max_iterations) break;
    previous_energy = result.energy;
    alpha = std::move(next_a);
    beta = std::move(next_b);
  }
  if (!result.converged) {
    result.density = concatenate(alpha, beta);
    return result;
  }

  // Close the successful UKS state on the actual unshifted physical operator.
  // DIIS/stabilized proposal orbitals are only a convergence device and must
  // never become the derivative-state proof. A small bounded fixed-point
  // correction mirrors the shared final-state policy without another SCF loop.
  auto final = evaluate(plan, basis, grid, alpha, beta, pbe, options.xc_tile_points);
  ++result.fock_builds;
  double previous_physical_energy = result.energy;
  result.converged = false;
  constexpr unsigned maximum_final_corrections = 4;
  for (unsigned correction = 0; correction < maximum_final_corrections; ++correction) {
    // A degenerate occupied/virtual boundary must retain the already qualified
    // occupation choice. Only the proposal is shifted; next remains F[D].
    ca = stabilize_occupations
             ? stabilized_uks_orbitals(final.fock.alpha, alpha, ints.overlap, x, n)
             : generalized_eigen(final.fock.alpha, x, n);
    cb = stabilize_occupations ? stabilized_uks_orbitals(final.fock.beta, beta, ints.overlap, x, n)
                               : generalized_eigen(final.fock.beta, x, n);
    Matrix projected_a = density_from_orbitals(ca.vectors, n, na, 1.0);
    Matrix projected_b = density_from_orbitals(cb.vectors, n, nb, 1.0);
    const double change_a = density_rms(projected_a, alpha);
    const double change_b = density_rms(projected_b, beta);
    const double density_change = std::max(change_a, change_b);
    result.density_rms = std::hypot(change_a, change_b) / std::sqrt(2.0);
    alpha = std::move(projected_a);
    beta = std::move(projected_b);

    auto next = evaluate(plan, basis, grid, alpha, beta, pbe, options.xc_tile_points);
    ++result.fock_builds;
    const Matrix ra = commutator_residual(next.fock.alpha, alpha, ints.overlap, n);
    const Matrix rb = commutator_residual(next.fock.beta, beta, ints.overlap, n);
    const double residual_a = residual_rms(ra), residual_b = residual_rms(rb);
    diagnostic.physical_residual = std::max(residual_a, residual_b);
    result.physical_residual_rms = std::hypot(residual_a, residual_b) / std::sqrt(2.0);
    diagnostic.components = next.components;
    diagnostic.electrons = {dot(alpha, ints.overlap), dot(beta, ints.overlap)};
    diagnostic.density_change = density_change;
    result.energy = next.components.total();
    result.energy_change = std::abs(result.energy - previous_physical_energy);
    final = std::move(next);
    if (result.energy_change < options.energy_tolerance &&
        density_change < options.density_tolerance &&
        diagnostic.physical_residual < residual_gate) {
      result.converged = true;
      break;
    }
    previous_physical_energy = result.energy;
  }
  if (result.converged && options.retain_ks_state)
    result.ks_physical_fock = concatenate(final.fock.alpha, final.fock.beta);
  result.density = concatenate(alpha, beta);
  return result;
}
// The registered CPU slice and the extended CPU/CUDA adapter share one UKS
// driver; compatibility entry points do not retain a second iteration loop.
ScfResult run_lda_uks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                      const dft::MolecularGrid& grid, const ScfOptions& options,
                      const std::vector<double>* initial_density) {
  return run_uks(plan, basis, grid, options, false, initial_density);
}
ScfResult run_pbe_uks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                      const dft::MolecularGrid& grid, const ScfOptions& options,
                      const std::vector<double>* initial_density) {
  return run_uks(plan, basis, grid, options, true, initial_density);
}
}  // namespace vibeqc::scf
