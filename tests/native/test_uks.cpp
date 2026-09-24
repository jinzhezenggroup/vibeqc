#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "dft/xc.hpp"
#include "molecule/basis.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/mean_field.hpp"
#include "scf/reference/mean_field.hpp"
#include "xc_cpu_generated.hpp"

namespace {
using namespace vibeqc;
using scf::reference::Matrix;

void require(bool passed, const char* message) {
  if (!passed) throw std::runtime_error(message);
}

dft::SemilocalPointValue pw91_program_point(const double rho[2],
                                            const double (&gradient)[2][3],
                                            const double[2]) {
  return dft::evaluate_pw91_point(rho, gradient);
}

const dft::SemilocalPointProgram kPw91QualificationProgram{
    "PW91 qualification program", dft::generated::kPw91SemilocalExpressionIdentity,
    7U, 1U, pw91_program_point};

core::System closed_shell_h2(double displacement = 0.0) {
  core::System system;
  system.atoms = {{1, {0.0, 0.0, 0.0}}, {1, {0.1, 0.2, 1.4 + displacement}}};
  system.shells = {
      {0,
       0,
       {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}},
      {1,
       0,
       {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}}};
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          "invalid closed-shell RSH fixture");
  return system;
}

core::System hydrogens(unsigned count, double shift = 0) {
  core::System system;
  system.multiplicity = 2;
  system.charge = count == 2 ? 1 : 0;
  for (unsigned i = 0; i < count; ++i) {
    system.atoms.push_back({1, {0.15 * i * i, 0.13 * i, 1.5 * i + shift * i}});
    system.shells.push_back(
        {i,
         0,
         {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}});
  }
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          "invalid UKS fixture");
  return system;
}

void run_case(unsigned atoms, bool pbe) {
  const auto system = hydrogens(atoms);
  scf::FockBuildSpec spec;
  spec.spin = scf::FockSpin::Unrestricted;
  spec.exchange.present = false;
  spec.derivative_order = 0;
  const auto strategy = scf::resolve_fock_build(spec, scf::FockBackend::Cpu);
  const scf::PreparedFockPlan plan(system, nullptr, strategy);
  const dft::AoBasis basis(system);
  const dft::GridSpec grid_spec{1, 24, 12, 24, 3, 1e-12};
  const dft::MolecularGrid grid(system, grid_spec);
  scf::ScfOptions options;
  options.compute_forces = false;
  options.max_iterations = 150;
  options.energy_tolerance = 1e-12;
  options.density_tolerance = 1e-10;
  const auto result = scf::run_uks(plan, basis, grid, options, pbe);
  require(result.converged && result.dft_diagnostic.physical_residual < 1e-9,
          "UKS cold endpoint failed");
  const auto& diagnostic = result.dft_diagnostic;
  const auto na = (system.electron_count + 1) / 2, nb = system.electron_count - na;
  require(diagnostic.occupations[0] == static_cast<std::size_t>(na) &&
              diagnostic.occupations[1] == static_cast<std::size_t>(nb),
          "UKS occupations are not independent");
  for (const auto& item : diagnostic.history) {
    require(std::abs(item.electrons[0] - na) < 1e-11 && std::abs(item.electrons[1] - nb) < 1e-11,
            "UKS iteration changed spin populations");
  }
  const auto [alpha, beta] =
      scf::reference::split_spin_matrices(result.density, static_cast<std::size_t>(atoms) * atoms);
  const auto xc = pbe ? dft::integrate_pbe_uks(basis, grid, alpha, beta)
                      : dft::integrate_lda_xc_pw_uks(basis, grid, alpha, beta);
  const auto jk = plan.build(alpha, beta);
  Matrix total(alpha.size());
  for (std::size_t i = 0; i < total.size(); ++i) total[i] = alpha[i] + beta[i];
  const auto& ints = plan.one_electron();
  const double hartree = 0.5 * scf::reference::dot(total, jk.coulomb);
  const double independent =
      ints.nuclear_repulsion + scf::reference::dot(total, ints.hcore) + hartree + xc.energy;
  require(std::abs(independent - result.energy) < 1e-12 &&
              std::abs(hartree - diagnostic.components.hartree) < 1e-12,
          "UKS physical E/D components do not describe one generation");
  require(std::abs(result.energy - (independent + xc.energy)) > 0.1,
          "UKS component gate cannot detect XC double counting");

  auto seed = result.density;
  for (std::size_t i = 0; i < seed.size(); ++i) seed[i] *= i < alpha.size() ? 1.1 : 0.7;
  const auto warm = scf::run_uks(plan, basis, grid, options, pbe, &seed);
  require(
      warm.converged && warm.initial_density_used && std::abs(warm.energy - result.energy) < 1e-10,
      "UKS warm normalization changed spin state or endpoint");
  options.strict_initial_density = true;
  bool rejected = false;
  try {
    (void)scf::run_uks(plan, basis, grid, options, pbe, &seed);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "strict UKS seed accepted wrong populations");
  options.strict_initial_density = false;

  const auto changed = hydrogens(atoms, 0.2);
  const scf::PreparedFockPlan changed_plan(changed, nullptr, strategy);
  const dft::AoBasis changed_basis(changed);
  const dft::MolecularGrid changed_grid(changed, grid_spec);
  if (atoms > 1) {
    rejected = false;
    try {
      (void)scf::run_uks(changed_plan, changed_basis, grid, options, pbe);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "UKS accepted a stale grid with the same atom count");
    rejected = false;
    try {
      (void)scf::run_uks(changed_plan, basis, changed_grid, options, pbe);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "UKS accepted stale AO centers with the same AO count");
  }
  const auto moved =
      scf::run_uks(changed_plan, changed_basis, changed_grid, options, pbe, &result.density);
  const auto cold_moved = scf::run_uks(changed_plan, changed_basis, changed_grid, options, pbe);
  require(
      moved.converged && cold_moved.converged && std::abs(moved.energy - cold_moved.energy) < 1e-10,
      "changed-geometry UKS warm state failed");

  options.max_iterations = 1;
  const auto failed = scf::run_uks(plan, basis, grid, options, pbe);
  require(!failed.converged && failed.iterations == 1 && failed.fock_builds == 1 &&
              failed.energy == failed.dft_diagnostic.components.total(),
          "failed UKS run published converged or inconsistent state");
  if (atoms == 3) {
    options.max_iterations = 2;
    options.energy_tolerance = options.density_tolerance = 10.0;
    const auto loose = scf::run_uks(plan, basis, grid, options, pbe);
    require(!loose.converged && loose.dft_diagnostic.physical_residual > 1e-9,
            "stable/loose energy and density gates replaced physical stationarity");
  }
}
void run_b3lyp_global_case() {
  const auto system = closed_shell_h2();
  const dft::AoBasis basis(system);
  const dft::GridSpec grid_spec{1, 1, 2, 4, 3, 1e-12};
  const dft::MolecularGrid grid(system, grid_spec);
  scf::ScfOptions options;
  options.compute_forces = false;
  options.max_iterations = 150;
  options.energy_tolerance = 1e-12;
  options.density_tolerance = 1e-10;

  const auto rks_strategy = scf::resolve_fock_build(
      scf::make_global_hybrid_fock_spec(scf::FockSpin::Restricted, 0.2), scf::FockBackend::Cpu);
  const scf::PreparedFockPlan rks_plan(system, nullptr, rks_strategy);
  const auto rks = scf::run_b3lyp_rks(rks_plan, basis, grid, options);
  require(rks.converged && std::isfinite(rks.energy) && rks.physical_residual_rms < 1e-9,
          "B3LYP native RKS did not converge on the audited interior grid");
  const auto rks_jk = rks_plan.build(rks.density);
  const auto rks_jk_energy =
      scf::contract_fock_energy_components(rks_strategy, rks_jk, rks.density);
  require(
      std::abs(rks.dft_diagnostic.components.hartree - rks_jk_energy.coulomb) < 1e-12 &&
          std::abs(rks.dft_diagnostic.components.exact_exchange - rks_jk_energy.exchange) < 1e-12,
      "B3LYP diagnostics disagree with the common full-range J/K provider");

  const auto warm = scf::run_b3lyp_rks(rks_plan, basis, grid, options, &rks.density);
  require(warm.converged && warm.initial_density_used && std::abs(warm.energy - rks.energy) < 1e-10,
          "B3LYP RKS warm replay changed the physical endpoint");

  const auto uks_strategy = scf::resolve_fock_build(
      scf::make_global_hybrid_fock_spec(scf::FockSpin::Unrestricted, 0.2), scf::FockBackend::Cpu);
  const scf::PreparedFockPlan uks_plan(system, nullptr, uks_strategy);
  const std::size_t n2 = basis.nao * basis.nao;
  std::vector<double> uks_seed(2 * n2);
  for (std::size_t i = 0; i < n2; ++i) uks_seed[i] = uks_seed[n2 + i] = 0.5 * rks.density[i];
  const auto uks = scf::run_b3lyp_uks(uks_plan, basis, grid, options, &uks_seed);
  require(uks.converged && std::isfinite(uks.energy) && uks.physical_residual_rms < 1e-9,
          "B3LYP native UKS did not converge on the audited interior grid");
  require(std::abs(uks.energy - rks.energy) < 2e-10,
          "B3LYP closed-shell RKS/UKS spin accounting disagrees");

  const auto wrong_strategy = scf::resolve_fock_build(
      scf::make_global_hybrid_fock_spec(scf::FockSpin::Restricted, 0.25), scf::FockBackend::Cpu);
  const scf::PreparedFockPlan wrong_plan(system, nullptr, wrong_strategy);
  bool rejected = false;
  try {
    (void)scf::run_b3lyp_rks(wrong_plan, basis, grid, options);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "B3LYP accepted a stale/wrong exact-exchange coefficient");

  options.compute_forces = true;
  rejected = false;
  try {
    (void)scf::run_b3lyp_rks(rks_plan, basis, grid, options);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "B3LYP interior value slice incorrectly advertised complete forces");
}

void run_generic_semilocal_scf_case() {
  const auto system = closed_shell_h2();
  const dft::AoBasis basis(system);
  const dft::GridSpec grid_spec{1, 1, 2, 4, 3, 1e-12};
  const dft::MolecularGrid grid(system, grid_spec);
  scf::ScfOptions options;
  options.compute_forces = false;
  options.max_iterations = 150;
  options.energy_tolerance = 1e-12;
  options.density_tolerance = 1e-10;

  scf::FockBuildSpec rks_spec;
  rks_spec.spin = scf::FockSpin::Restricted;
  rks_spec.exchange.present = false;
  rks_spec.derivative_order = 0;
  const auto rks_strategy = scf::resolve_fock_build(rks_spec, scf::FockBackend::Cpu);
  const scf::PreparedFockPlan rks_plan(system, nullptr, rks_strategy);
  const auto rks =
      scf::run_semilocal_rks(rks_plan, basis, grid, options, kPw91QualificationProgram);
  require(rks.converged && std::isfinite(rks.energy) && rks.physical_residual_rms < 1e-9,
          "generic semilocal RKS qualification path did not converge");
  const auto rks_xc = dft::integrate_pw91_rks(basis, grid, rks.density);
  const auto rks_jk = rks_plan.build(rks.density);
  const auto& ints = rks_plan.one_electron();
  const double independent =
      ints.nuclear_repulsion + scf::reference::dot(rks.density, ints.hcore) +
      0.5 * scf::reference::dot(rks.density, rks_jk.coulomb) + rks_xc.energy;
  require(std::abs(independent - rks.energy) < 2e-11,
          "generic semilocal RKS endpoint disagrees with component rebuild");
  require(rks.dft_diagnostic.ao_order == 1 && rks.dft_diagnostic.scf_domain_version == 1,
          "generic semilocal RKS lost program feature/domain metadata");

  scf::FockBuildSpec uks_spec = rks_spec;
  uks_spec.spin = scf::FockSpin::Unrestricted;
  const auto uks_strategy = scf::resolve_fock_build(uks_spec, scf::FockBackend::Cpu);
  const scf::PreparedFockPlan uks_plan(system, nullptr, uks_strategy);
  const std::size_t n2 = basis.nao * basis.nao;
  std::vector<double> seed(2 * n2);
  for (std::size_t i = 0; i < n2; ++i) seed[i] = seed[n2 + i] = 0.5 * rks.density[i];
  const auto uks =
      scf::run_semilocal_uks(uks_plan, basis, grid, options, kPw91QualificationProgram, &seed);
  require(uks.converged && std::isfinite(uks.energy) && uks.physical_residual_rms < 1e-9,
          "generic semilocal UKS qualification path did not converge");
  require(std::abs(uks.energy - rks.energy) < 2e-10,
          "generic semilocal closed-shell RKS/UKS endpoints disagree");

  auto invalid = kPw91QualificationProgram;
  invalid.expression_identity = "";
  bool rejected = false;
  try {
    (void)scf::run_semilocal_rks(rks_plan, basis, grid, options, invalid);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "generic semilocal SCF accepted an unbound program identity");
}

void run_cam_rsh_case() {
  const auto system = closed_shell_h2();
  const dft::AoBasis basis(system);
  const dft::GridSpec grid_spec{1, 1, 2, 4, 3, 1e-12};
  const dft::MolecularGrid grid(system, grid_spec);
  scf::ScfOptions options;
  options.compute_forces = false;
  options.max_iterations = 150;
  options.energy_tolerance = 1e-12;
  options.density_tolerance = 1e-10;

  const auto rks_primary_strategy = scf::resolve_fock_build(
      scf::make_rsh_primary_fock_spec(scf::FockSpin::Restricted, 0.19), scf::FockBackend::Cpu);
  const auto rks_correction_strategy = scf::resolve_fock_build(
      scf::make_rsh_correction_fock_spec(scf::FockSpin::Restricted, 0.19, 0.65, 0.33),
      scf::FockBackend::Cpu);
  const scf::PreparedFockPlan rks_primary(system, nullptr, rks_primary_strategy);
  const scf::PreparedFockPlan rks_correction(system, nullptr, rks_correction_strategy);
  const auto rks = scf::run_cam_b3lyp_rks(rks_primary, rks_correction, basis, grid, options);
  require(rks.converged && std::isfinite(rks.energy) && rks.physical_residual_rms < 1e-9,
          "CAM-B3LYP native RKS did not converge on the audited interior grid");
  const auto rks_primary_jk = rks_primary.build(rks.density);
  const auto rks_correction_jk = rks_correction.build(rks.density);
  const auto rks_primary_energy =
      scf::contract_fock_energy_components(rks_primary_strategy, rks_primary_jk, rks.density);
  const auto rks_correction_energy =
      scf::contract_fock_energy_components(rks_correction_strategy, rks_correction_jk, rks.density);
  require(std::abs(rks.dft_diagnostic.components.hartree - rks_primary_energy.coulomb) < 1e-12 &&
              std::abs(rks.dft_diagnostic.components.exact_exchange -
                       (rks_primary_energy.exchange + rks_correction_energy.exchange)) < 1e-12,
          "CAM-B3LYP diagnostics mixed Hartree and range-separated exact exchange");
  const auto warm =
      scf::run_cam_b3lyp_rks(rks_primary, rks_correction, basis, grid, options, &rks.density);
  require(warm.converged && warm.initial_density_used && std::abs(warm.energy - rks.energy) < 1e-10,
          "CAM-B3LYP RKS warm replay changed the physical endpoint");

  const auto uks_primary_strategy = scf::resolve_fock_build(
      scf::make_rsh_primary_fock_spec(scf::FockSpin::Unrestricted, 0.19), scf::FockBackend::Cpu);
  const auto uks_correction_strategy = scf::resolve_fock_build(
      scf::make_rsh_correction_fock_spec(scf::FockSpin::Unrestricted, 0.19, 0.65, 0.33),
      scf::FockBackend::Cpu);
  const scf::PreparedFockPlan uks_primary(system, nullptr, uks_primary_strategy);
  const scf::PreparedFockPlan uks_correction(system, nullptr, uks_correction_strategy);
  const std::size_t n2 = basis.nao * basis.nao;
  std::vector<double> uks_seed(2 * n2);
  for (std::size_t i = 0; i < n2; ++i) uks_seed[i] = uks_seed[n2 + i] = 0.5 * rks.density[i];
  const auto uks =
      scf::run_cam_b3lyp_uks(uks_primary, uks_correction, basis, grid, options, &uks_seed);
  require(uks.converged && std::isfinite(uks.energy) && uks.physical_residual_rms < 1e-9,
          "CAM-B3LYP native UKS did not converge on the audited interior grid");
  require(std::abs(uks.energy - rks.energy) < 2e-10,
          "CAM-B3LYP closed-shell RKS/UKS spin accounting disagrees");

  const auto wrong_omega_strategy = scf::resolve_fock_build(
      scf::make_rsh_correction_fock_spec(scf::FockSpin::Restricted, 0.19, 0.65, 0.34),
      scf::FockBackend::Cpu);
  const scf::PreparedFockPlan wrong_omega(system, nullptr, wrong_omega_strategy);
  bool rejected = false;
  try {
    (void)scf::run_cam_b3lyp_rks(rks_primary, wrong_omega, basis, grid, options);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "CAM-B3LYP RSH accepted a correction plan with stale omega identity");

  const auto moved_system = closed_shell_h2(0.08);
  const dft::AoBasis moved_basis(moved_system);
  const dft::MolecularGrid moved_grid(moved_system, grid_spec);
  const scf::PreparedFockPlan moved_primary(moved_system, nullptr, rks_primary_strategy);
  const scf::PreparedFockPlan moved_correction(moved_system, nullptr, rks_correction_strategy);
  rejected = false;
  try {
    (void)scf::run_cam_b3lyp_rks(moved_primary, moved_correction, moved_basis, grid, options);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "CAM-B3LYP RSH accepted a stale grid after geometry change");
  const auto moved_warm = scf::run_cam_b3lyp_rks(moved_primary, moved_correction, moved_basis,
                                                 moved_grid, options, &rks.density);
  const auto moved_cold =
      scf::run_cam_b3lyp_rks(moved_primary, moved_correction, moved_basis, moved_grid, options);
  require(moved_warm.converged && moved_cold.converged &&
              std::abs(moved_warm.energy - moved_cold.energy) < 1e-10,
          "CAM-B3LYP RSH changed-geometry warm replay changed the endpoint");

  options.compute_forces = true;
  rejected = false;
  try {
    (void)scf::run_cam_b3lyp_rks(rks_primary, rks_correction, basis, grid, options);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "CAM-B3LYP value slice incorrectly advertised analytic forces");
}

}  // namespace

int main() {
  try {
    for (unsigned atoms : {1U, 2U, 3U})
      for (bool pbe : {false, true}) run_case(atoms, pbe);
    run_b3lyp_global_case();
    run_generic_semilocal_scf_case();
    run_cam_rsh_case();
    std::cout << "UKS physical state, spin, warm and stale-input gates passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
