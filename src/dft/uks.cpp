#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <tuple>

#include "dft/ao_grid.hpp"
#include "dft/grid.hpp"
#include "dft/nonlocal_correlation/vv10_integration.hpp"
#include "dft/nonlocal_correlation/vv10_runtime.hpp"
#include "dft/xc.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/initial_guess/density.hpp"
#include "scf/mean_field.hpp"
#include "scf/reference/mean_field.hpp"
#include "scf/solver/diis.hpp"
#include "scf/solver/proposal_control.hpp"
#include "scf/solver/self_consistent.hpp"
#include "xc_cpu_generated.hpp"

namespace vibeqc::scf {
namespace {
using namespace reference;

struct SpinEvaluation {
  FockMatrices fock;
  dft::EnergyComponents components;
};

struct SpinXcEvaluator {
  using Direct = dft::SpinXcIntegral (*)(const dft::AoBasis&, const dft::MolecularGrid&,
                                         const Matrix&, const Matrix&, std::size_t, double, double);
  Direct direct{};
  const dft::SemilocalPointProgram* program{};

  SpinXcEvaluator(Direct value) : direct(value) {}
  SpinXcEvaluator(const dft::SemilocalPointProgram& value) : program(&value) {}

  dft::SpinXcIntegral operator()(const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                                 const Matrix& alpha, const Matrix& beta, std::size_t tile,
                                 double exchange_scale, double correlation_scale) const {
    if (program) {
      if (exchange_scale != 1.0 || correlation_scale != 1.0)
        throw std::invalid_argument("generic semilocal UKS does not accept legacy XC scaling");
      return dft::integrate_semilocal_uks(basis, grid, alpha, beta, *program, tile);
    }
    if (!direct) throw std::logic_error("UKS XC evaluator is empty");
    return direct(basis, grid, alpha, beta, tile, exchange_scale, correlation_scale);
  }
};

dft::SpinXcIntegral evaluate_lda_xc_uks(const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                                        const Matrix& alpha, const Matrix& beta, std::size_t tile,
                                        double exchange_scale, double correlation_scale) {
  if (exchange_scale != 1.0 || correlation_scale != 1.0)
    throw std::invalid_argument("scaled LDA UKS is not qualified");
  return dft::integrate_lda_xc_pw_uks(basis, grid, alpha, beta, tile);
}

dft::SpinXcIntegral evaluate_pbe_xc_uks(const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                                        const Matrix& alpha, const Matrix& beta, std::size_t tile,
                                        double exchange_scale, double correlation_scale) {
  return dft::integrate_pbe_uks_scaled(basis, grid, alpha, beta, tile, exchange_scale,
                                       correlation_scale);
}

dft::SpinXcIntegral evaluate_r2scan_xc_uks(const dft::AoBasis& basis,
                                           const dft::MolecularGrid& grid, const Matrix& alpha,
                                           const Matrix& beta, std::size_t tile,
                                           double exchange_scale, double correlation_scale) {
  if (exchange_scale != 1.0 || correlation_scale != 1.0)
    throw std::invalid_argument("scaled r2SCAN UKS is not qualified");
  return dft::integrate_r2scan_uks(basis, grid, alpha, beta, tile);
}

dft::SpinXcIntegral evaluate_b3lyp_xc_uks(const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                                          const Matrix& alpha, const Matrix& beta, std::size_t tile,
                                          double exchange_scale, double correlation_scale) {
  if (exchange_scale != 1.0 || correlation_scale != 1.0)
    throw std::invalid_argument("scaled B3LYP UKS is not qualified");
  return dft::integrate_b3lyp_uks(basis, grid, alpha, beta, tile);
}

dft::SpinXcIntegral evaluate_wb97mv_xc_uks(const dft::AoBasis& basis,
                                           const dft::MolecularGrid& grid, const Matrix& alpha,
                                           const Matrix& beta, std::size_t tile,
                                           double exchange_scale, double correlation_scale) {
  if (exchange_scale != 1.0 || correlation_scale != 1.0)
    throw std::invalid_argument("scaled WB97M-V semilocal execution is not qualified");
  return dft::integrate_wb97mv_uks(basis, grid, alpha, beta, tile);
}

dft::SpinXcIntegral evaluate_cam_b3lyp_xc_uks(const dft::AoBasis& basis,
                                              const dft::MolecularGrid& grid, const Matrix& alpha,
                                              const Matrix& beta, std::size_t tile,
                                              double exchange_scale, double correlation_scale) {
  if (exchange_scale != 1.0 || correlation_scale != 1.0)
    throw std::invalid_argument("scaled CAM-B3LYP UKS is not qualified");
  return dft::integrate_cam_b3lyp_uks(basis, grid, alpha, beta, tile);
}

/** The physical operator is independent of extrapolation and occupations.
 * The common Fock plans own J/K dispatch; RSH adds a structurally separate
 * long-range exchange correction without changing proposal semantics. */
SpinEvaluation evaluate(const PreparedFockPlan& plan, const PreparedFockPlan* long_range_correction,
                        const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                        const Matrix& alpha, const Matrix& beta, SpinXcEvaluator evaluate_xc,
                        const ScfOptions& options, dft::nlc::Vv10Plan* nonlocal_correlation,
                        dft::nlc::Vv10DensityDomain nonlocal_domain) {
  const auto& ints = plan.one_electron();
  const auto jk = plan.build(alpha, beta);
  SpinEvaluation out;
  out.fock = assemble_fock(plan.strategy(), ints.hcore, jk);
  const auto primary_energy = contract_fock_energy_components(plan.strategy(), jk, alpha, beta);
  double exact_exchange = primary_energy.exchange;
  if (long_range_correction) {
    const auto correction_jk = long_range_correction->build(alpha, beta);
    const auto& correction_strategy = long_range_correction->strategy();
    if (correction_jk.exchange_alpha.size() != out.fock.alpha.size() ||
        correction_jk.exchange_beta.size() != out.fock.beta.size())
      throw std::runtime_error(
          "RSH spin correction exchange dimensions do not match the Fock matrix");
    for (std::size_t i = 0; i < alpha.size(); ++i) {
      out.fock.alpha[i] +=
          correction_strategy.spec.exchange.coefficient * correction_jk.exchange_alpha[i];
      out.fock.beta[i] +=
          correction_strategy.spec.exchange.coefficient * correction_jk.exchange_beta[i];
    }
    exact_exchange +=
        contract_fock_energy_components(correction_strategy, correction_jk, alpha, beta).exchange;
  }
  const auto xc =
      evaluate_xc(basis, grid, alpha, beta, options.xc_tile_points,
                  options.semilocal_exchange_scale, options.semilocal_correlation_scale);
  dft::nlc::SpinVv10Integral nonlocal;
  if (nonlocal_correlation)
    nonlocal = dft::nlc::integrate_vv10_uks(basis, grid, alpha, beta, *nonlocal_correlation,
                                            options.xc_tile_points, nonlocal_domain);
  if (xc.potential[0].size() != out.fock.alpha.size() ||
      xc.potential[1].size() != out.fock.beta.size() ||
      (nonlocal_correlation && (nonlocal.potential[0].size() != out.fock.alpha.size() ||
                                nonlocal.potential[1].size() != out.fock.beta.size())))
    throw std::runtime_error("UKS XC potential dimensions do not match the Fock matrix");
  for (std::size_t i = 0; i < alpha.size(); ++i) {
    out.fock.alpha[i] += xc.potential[0][i];
    out.fock.beta[i] += xc.potential[1][i];
    if (nonlocal_correlation) {
      out.fock.alpha[i] += nonlocal.potential[0][i];
      out.fock.beta[i] += nonlocal.potential[1][i];
    }
  }
  if (nonlocal_correlation)
    runtime::sample_cpu_capacity(runtime::add_capacity(
        nonlocal.owned_numeric_bytes, runtime::vector_capacities(out.fock.alpha, out.fock.beta)));
  out.components = {ints.nuclear_repulsion, dot(alpha, ints.hcore) + dot(beta, ints.hcore),
                    primary_energy.coulomb,
                    xc.energy + (nonlocal_correlation ? nonlocal.energy : 0.0), exact_exchange};
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

ScfResult run_uks_impl(
    const PreparedFockPlan& plan, const PreparedFockPlan* long_range_correction,
    const dft::AoBasis& basis, const dft::MolecularGrid& grid, const ScfOptions& options,
    SpinXcEvaluator evaluate_xc, const char* method_name,
    const std::vector<double>* initial_density, dft::nlc::Vv10Plan* nonlocal_correlation,
    dft::nlc::Vv10DensityDomain nonlocal_domain = dft::nlc::Vv10DensityDomain::StrictPositive) {
  using namespace reference;
  const auto& strategy = plan.strategy();
  validate_resolved_fock_build(strategy);
  if (options.compute_forces)
    throw std::invalid_argument(std::string(method_name) + " UKS forces are not implemented");
  if (strategy.backend != FockBackend::Cpu || strategy.spec.spin != FockSpin::Unrestricted ||
      strategy.spec.derivative_order != 0 || !strategy.spec.coulomb.present ||
      strategy.spec.coulomb.coefficient != 1.0 ||
      (strategy.spec.exchange.present &&
       (strategy.spec.exchange.op != FockOperator::FullRange ||
        (strategy.spec.exchange.approximation != FockApproximation::Exact &&
         strategy.spec.exchange.approximation != FockApproximation::DensityFitted))))
    throw std::invalid_argument("UKS requires a CPU full-range exact or fitted J/K Fock strategy");
  if (long_range_correction) {
    const auto& correction = long_range_correction->strategy();
    validate_resolved_fock_build(correction);
    const bool primary_exchange =
        strategy.spec.exchange.present &&
        strategy.spec.exchange.approximation == FockApproximation::Exact &&
        strategy.spec.exchange.op == FockOperator::FullRange && strategy.spec.exchange.omega == 0.0;
    const bool correction_exchange =
        correction.backend == FockBackend::Cpu && correction.spec.spin == FockSpin::Unrestricted &&
        correction.spec.derivative_order == 0 && !correction.spec.coulomb.present &&
        correction.spec.exchange.present &&
        correction.spec.exchange.approximation == FockApproximation::Exact &&
        correction.spec.exchange.op == FockOperator::LongRange;
    if (!primary_exchange || !correction_exchange)
      throw std::invalid_argument("RSH UKS requires full-range primary K plus direct long-range K");
  }
  if (options.xc_density_route != dft::XcDensityRoute::DensityMatrix)
    throw std::invalid_argument("UKS occupied-factor XC has not been implemented");
  const auto& system = plan.system();
  const auto& ints = plan.one_electron();
  const std::size_t n = ints.nbf;
  if (basis.nao != n || basis.natom != system.atoms.size() || grid.point_count() == 0)
    throw std::invalid_argument("UKS grid/basis binding is inconsistent");
  // The source already owns its auxiliary basis. Comparing against a null
  // auxiliary request would incorrectly replace it with the orbital basis.
  if (!plan.matches_system(grid.system()) || basis.packed != dft::AoBasis(system).packed)
    throw std::invalid_argument("UKS refuses a stale geometry, basis, charge or spin binding");
  if (long_range_correction &&
      (!long_range_correction->matches(system, nullptr, long_range_correction->strategy(), -1, 0) ||
       long_range_correction->one_electron().nbf != ints.nbf))
    throw std::invalid_argument("RSH correction refuses a stale or incompatible source binding");
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
  diagnostic.ao_order = evaluate_xc.program
                            ? (evaluate_xc.program->ingredient_mask == 1U ? 0U : 1U)
                            : (std::string_view(method_name) == "LDA" ? 0U : 1U);
  diagnostic.scf_domain_version =
      evaluate_xc.program ? evaluate_xc.program->domain_version
                          : (std::string_view(method_name) == "WB97M-V"
                                 ? 3U
                                 : (std::string_view(method_name) == "B3LYP" ? 2U : 1U));
  const double residual_gate = std::min(1.0e-9, options.density_tolerance);
  bool stabilize_occupations = false;

  struct UksState {
    Matrix alpha;
    Matrix beta;
  };
  struct UksLoopEvaluation {
    FockMatrices physical_fock;
    Matrix alpha_residual;
    Matrix beta_residual;
    Matrix effective_alpha;
    Matrix effective_beta;
    Matrix next_alpha;
    Matrix next_beta;
    dft::EnergyComponents components;
    double energy{};
    double state_rms{};
    double residual_rms{};
    double joined_density_rms{};
    double joined_residual_rms{};
    double alpha_electrons{};
    double beta_electrons{};
    bool stabilized{};
  };

  const solver::SelfConsistentPolicy policy{options.max_iterations, options.energy_tolerance,
                                            options.density_tolerance, residual_gate, true};
  auto outcome = solver::run_self_consistent(
      UksState{std::move(alpha), std::move(beta)}, policy,
      [&](const UksState& state, unsigned) {
        const bool stabilized = stabilize_occupations;
        auto physical = evaluate(plan, long_range_correction, basis, grid, state.alpha, state.beta,
                                 evaluate_xc, options, nonlocal_correlation, nonlocal_domain);
        ++result.fock_builds;
        Matrix ra = commutator_residual(physical.fock.alpha, state.alpha, ints.overlap, n);
        Matrix rb = commutator_residual(physical.fock.beta, state.beta, ints.overlap, n);
        const double residual_a = residual_rms(ra), residual_b = residual_rms(rb);
        auto effective = split_spin_matrices(
            diis.update(concatenate(physical.fock.alpha, physical.fock.beta), concatenate(ra, rb)),
            n * n);
        ca = stabilized ? stabilized_uks_orbitals(effective.first, state.alpha, ints.overlap, x, n)
                        : generalized_eigen(effective.first, x, n);
        cb = stabilized ? stabilized_uks_orbitals(effective.second, state.beta, ints.overlap, x, n)
                        : generalized_eigen(effective.second, x, n);
        Matrix next_a = density_from_orbitals(ca.vectors, n, na, 1.0);
        Matrix next_b = density_from_orbitals(cb.vectors, n, nb, 1.0);
        const double change_a = density_rms(next_a, state.alpha);
        const double change_b = density_rms(next_b, state.beta);
        return UksLoopEvaluation{std::move(physical.fock),
                                 std::move(ra),
                                 std::move(rb),
                                 std::move(effective.first),
                                 std::move(effective.second),
                                 std::move(next_a),
                                 std::move(next_b),
                                 physical.components,
                                 physical.components.total(),
                                 std::max(change_a, change_b),
                                 std::max(residual_a, residual_b),
                                 std::hypot(change_a, change_b) / std::sqrt(2.0),
                                 std::hypot(residual_a, residual_b) / std::sqrt(2.0),
                                 dot(state.alpha, ints.overlap),
                                 dot(state.beta, ints.overlap),
                                 stabilized};
      },
      [&](UksState& state, UksLoopEvaluation evaluation,
          const solver::SelfConsistentProgress& progress) {
        runtime::sample_cpu_capacity(runtime::add_capacity(
            runtime::add_capacity(
                runtime::add_capacity(
                    plan.cpu_observation_capacity(),
                    long_range_correction ? long_range_correction->cpu_observation_capacity() : 0),
                diis.numeric_capacity()),
            runtime::vector_capacities(basis.packed, grid.points(), grid.weights(), grid.owners(),
                                       x, state.alpha, state.beta, ca.values, ca.vectors, cb.values,
                                       cb.vectors, evaluation.physical_fock.alpha,
                                       evaluation.physical_fock.beta, evaluation.alpha_residual,
                                       evaluation.beta_residual, evaluation.effective_alpha,
                                       evaluation.effective_beta, evaluation.next_alpha,
                                       evaluation.next_beta, diagnostic.history)));

        // Preserve #305's occupation-cycle policy: only subsequent proposals
        // are shifted, while every physical convergence gate remains unshifted.
        if (!progress.converged && progress.iteration > 1 &&
            progress.energy_change < options.energy_tolerance &&
            progress.residual_rms < residual_gate &&
            progress.state_rms >= options.density_tolerance)
          stabilize_occupations = true;

        // UKS convergence certifies the CURRENT physical state. The DIIS/level-
        // shifted proposal is only a check and must never replace that state.
        if (progress.converged || progress.iteration == options.max_iterations)
          return UksState{std::move(state.alpha), std::move(state.beta)};
        return UksState{std::move(evaluation.next_alpha), std::move(evaluation.next_beta)};
      },
      [&](const solver::SelfConsistentProgress& progress, const UksLoopEvaluation& evaluation) {
        result.energy = progress.energy;
        result.iterations = progress.iteration;
        result.energy_change = progress.energy_change;
        result.density_rms = evaluation.joined_density_rms;
        diagnostic.physical_residual = progress.residual_rms;
        result.physical_residual_rms = evaluation.joined_residual_rms;
        diagnostic.components = evaluation.components;
        diagnostic.electrons = {evaluation.alpha_electrons, evaluation.beta_electrons};
        diagnostic.density_change = progress.state_rms;
        diagnostic.history.push_back(
            {progress.iteration, evaluation.components, progress.energy_change, progress.state_rms,
             progress.residual_rms, diagnostic.electrons, evaluation.stabilized});
      });
  alpha = std::move(outcome.state.alpha);
  beta = std::move(outcome.state.beta);
  result.converged = outcome.converged;

  if (!result.converged) {
    result.density = concatenate(alpha, beta);
    return result;
  }

  // Close the successful UKS state on the actual unshifted physical operator.
  // DIIS/stabilized proposal orbitals are only a convergence device and must
  // never become the derivative-state proof. A small bounded fixed-point
  // correction mirrors the shared final-state policy without another SCF loop.
  auto final = evaluate(plan, long_range_correction, basis, grid, alpha, beta, evaluate_xc, options,
                        nonlocal_correlation, nonlocal_domain);
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

    auto next = evaluate(plan, long_range_correction, basis, grid, alpha, beta, evaluate_xc,
                         options, nonlocal_correlation, nonlocal_domain);
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
ScfResult run_uks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                  const dft::MolecularGrid& grid, const ScfOptions& options, bool pbe,
                  const std::vector<double>* initial_density) {
  return run_uks_impl(plan, nullptr, basis, grid, options,
                      pbe ? evaluate_pbe_xc_uks : evaluate_lda_xc_uks, pbe ? "PBE" : "LDA",
                      initial_density, nullptr);
}
ScfResult run_semilocal_uks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                            const dft::MolecularGrid& grid, const ScfOptions& options,
                            const dft::SemilocalPointProgram& program,
                            const std::vector<double>* initial_density) {
  dft::validate_semilocal_point_program(program);
  return run_uks_impl(plan, nullptr, basis, grid, options, SpinXcEvaluator(program),
                      program.identifier, initial_density, nullptr);
}

ScfResult run_lda_uks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                      const dft::MolecularGrid& grid, const ScfOptions& options,
                      const std::vector<double>* initial_density) {
  return run_uks_impl(plan, nullptr, basis, grid, options, evaluate_lda_xc_uks, "LDA",
                      initial_density, nullptr);
}
ScfResult run_pbe_uks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                      const dft::MolecularGrid& grid, const ScfOptions& options,
                      const std::vector<double>* initial_density) {
  return run_uks_impl(plan, nullptr, basis, grid, options, evaluate_pbe_xc_uks, "PBE",
                      initial_density, nullptr);
}
ScfResult run_pbe_uks_nonlocal(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                               const dft::MolecularGrid& grid, const ScfOptions& options,
                               const std::vector<double>* initial_density,
                               dft::nlc::Vv10Plan& nonlocal_correlation) {
  return run_uks_impl(plan, nullptr, basis, grid, options, evaluate_pbe_xc_uks, "PBE",
                      initial_density, &nonlocal_correlation);
}

ScfResult run_pbe_rsh_uks(const PreparedFockPlan& primary,
                          const PreparedFockPlan& long_range_correction, const dft::AoBasis& basis,
                          const dft::MolecularGrid& grid, const ScfOptions& options,
                          const std::vector<double>* initial_density,
                          dft::nlc::Vv10Plan* nonlocal_correlation) {
  return run_uks_impl(primary, &long_range_correction, basis, grid, options, evaluate_pbe_xc_uks,
                      "PBE-RSH", initial_density, nonlocal_correlation);
}
ScfResult run_r2scan_uks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                         const dft::MolecularGrid& grid, const ScfOptions& options,
                         const std::vector<double>* initial_density) {
  return run_uks_impl(plan, nullptr, basis, grid, options, evaluate_r2scan_xc_uks, "R2SCAN",
                      initial_density, nullptr);
}
ScfResult run_b3lyp_uks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                        const dft::MolecularGrid& grid, const ScfOptions& options,
                        const std::vector<double>* initial_density) {
  auto spec =
      make_global_hybrid_fock_spec(FockSpin::Unrestricted, dft::generated::kB3lypExactExchange);
  if (options.density_fitting_mode != VIBEQC_DENSITY_FITTING_NONE) {
    spec.coulomb.approximation = FockApproximation::DensityFitted;
    spec.exchange.approximation = FockApproximation::DensityFitted;
  }
  const auto expected = resolve_fock_build(spec, FockBackend::Cpu, options.screening_tolerance,
                                           options.density_fitting_relative_threshold);
  if (plan.strategy() != expected)
    throw std::invalid_argument("B3LYP plan does not match the generated MethodIR composition");
  return run_uks_impl(plan, nullptr, basis, grid, options, evaluate_b3lyp_xc_uks, "B3LYP",
                      initial_density, nullptr);
}

ScfResult run_wb97mv_uks(const PreparedFockPlan& primary, const PreparedFockPlan& correction,
                         const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                         const ScfOptions& options, dft::nlc::Vv10Plan& nonlocal,
                         const std::vector<double>* initial_density) {
  require_wb97mv_composition(primary.strategy(), correction.strategy(), nonlocal.parameters());
  if (nonlocal.backend() != VIBEQC_BACKEND_CPU_REFERENCE ||
      nonlocal.resources().point_count != grid.point_count())
    throw std::invalid_argument("WB97M-V nonlocal owner is incompatible with the KS grid/backend");
  return run_uks_impl(primary, &correction, basis, grid, options, evaluate_wb97mv_xc_uks, "WB97M-V",
                      initial_density, &nonlocal, dft::nlc::Vv10DensityDomain::MolecularV1);
}

ScfResult run_cam_b3lyp_uks(const PreparedFockPlan& primary,
                            const PreparedFockPlan& long_range_correction,
                            const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                            const ScfOptions& options, const std::vector<double>* initial_density) {
  const auto expected_primary = resolve_fock_build(
      make_rsh_primary_fock_spec(FockSpin::Unrestricted, dft::generated::kCamB3lypShortExchange),
      FockBackend::Cpu);
  const auto expected_correction =
      resolve_fock_build(make_rsh_correction_fock_spec(
                             FockSpin::Unrestricted, dft::generated::kCamB3lypShortExchange,
                             dft::generated::kCamB3lypLongExchange, dft::generated::kCamB3lypOmega),
                         FockBackend::Cpu);
  if (primary.strategy() != expected_primary ||
      long_range_correction.strategy() != expected_correction)
    throw std::invalid_argument("CAM-B3LYP plans do not match the generated MethodIR composition");
  return run_uks_impl(primary, &long_range_correction, basis, grid, options,
                      evaluate_cam_b3lyp_xc_uks, "CAM-B3LYP", initial_density, nullptr);
}
}  // namespace vibeqc::scf
