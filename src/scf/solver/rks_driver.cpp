#include "scf/mean_field.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "dft/ao_grid.hpp"
#include "dft/grid.hpp"
#include "dft/xc.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/fock_build.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/initial_guess/density.hpp"
#include "scf/reference/mean_field.hpp"
#include "scf/solver/diis.hpp"
#include "scf/solver/proposal_control.hpp"

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

template <class... Vectors>
void sample_rks_buffers(const PreparedFockPlan& plan, const Diis& diis,
                        const Vectors&... vectors) noexcept {
  if (!runtime::cpu_resource_observation.active) return;
  runtime::sample_cpu_capacity(runtime::add_capacity(
      plan.cpu_observation_capacity(),
      runtime::add_capacity(diis.numeric_capacity(), runtime::vector_capacities(vectors...))));
}

struct RksEvaluation {
  Matrix fock;
  double energy{};
};

using RksXcEvaluator = dft::XcIntegral (*)(const dft::AoBasis&, const dft::MolecularGrid&,
                                           const Matrix&);

dft::XcIntegral evaluate_lda_xc_rks(const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                                     const Matrix& density) {
  return dft::integrate_lda_xc_pw_rks(basis, grid, density);
}

dft::XcIntegral evaluate_pbe_xc_rks(const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                                     const Matrix& density) {
  return dft::integrate_pbe_rks_with_tail(basis, grid, density);
}

RksEvaluation evaluate_rks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                           const dft::MolecularGrid& grid, const Matrix& density,
                           RksXcEvaluator evaluate_xc, const char* method_name) {
  const auto& strategy = plan.strategy();
  const auto& ints = plan.one_electron();
  const auto jk = plan.build(density);
  RksEvaluation result;
  result.fock = assemble_fock(strategy, ints.hcore, jk).alpha;
  const auto xc = evaluate_xc(basis, grid, density);
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

ScfResult run_rks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                  const dft::MolecularGrid& grid, const ScfOptions& options,
                  const std::vector<double>* initial_density, RksXcEvaluator evaluate_xc,
                  const char* method_name) {
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
  if (basis.nao != ints.nbf || basis.natom != system.atoms.size() ||
      grid.point_count() == 0 || grid.system().atoms.size() != system.atoms.size())
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
    validate_seed(ints.overlap, *initial_density, n,
                  {static_cast<unsigned>(system.electron_count)}, 2.0);
    density = *initial_density;
  }
  Diis diis(options.diis_history);
  ScfResult result;
  result.initial_density_used = initial_density != nullptr;
  double previous_energy = std::numeric_limits<double>::infinity();
  for (unsigned iteration = 1; iteration <= options.max_iterations; ++iteration) {
    ++result.fock_builds;
    const auto physical = evaluate_rks(plan, basis, grid, density, evaluate_xc, method_name);
    const Matrix residual = commutator_residual(physical.fock, density, ints.overlap, n);
    const Matrix effective_fock = diis.update(physical.fock, residual);
    orbitals = generalized_eigen(effective_fock, orthogonalizer, n);
    Matrix next_density = density_from_orbitals(orbitals.vectors, n, occupied);

    sample_rks_buffers(plan, diis, orthogonalizer, density, physical.fock, residual,
                       effective_fock, orbitals.values, orbitals.vectors, next_density);
    result.iterations = iteration;
    result.energy = physical.energy;
    result.energy_change = std::isfinite(previous_energy)
                               ? std::abs(physical.energy - previous_energy)
                               : std::numeric_limits<double>::infinity();
    result.density_rms = density_rms(next_density, density);
    if (iteration > 1 && result.energy_change < options.energy_tolerance &&
        result.density_rms < options.density_tolerance &&
        residual_rms(residual) < options.density_tolerance) {
      density = std::move(next_density);
      result.converged = true;
      break;
    }
    previous_energy = physical.energy;
    density = std::move(next_density);
  }
  if (!result.converged) {
    result.density = std::move(density);
    return result;
  }

  result.fock_builds += 2;
  auto final = evaluate_rks(plan, basis, grid, density, evaluate_xc, method_name);
  orbitals = generalized_eigen(final.fock, orthogonalizer, n);
  density = density_from_orbitals(orbitals.vectors, n, occupied);
  final = evaluate_rks(plan, basis, grid, density, evaluate_xc, method_name);
  result.energy = final.energy;
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
