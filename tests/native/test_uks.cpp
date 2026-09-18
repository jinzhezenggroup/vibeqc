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

namespace {
using namespace vibeqc;
using scf::reference::Matrix;

void require(bool passed, const char* message) {
  if (!passed) throw std::runtime_error(message);
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
}  // namespace

int main() {
  try {
    for (unsigned atoms : {1U, 2U, 3U})
      for (bool pbe : {false, true}) run_case(atoms, pbe);
    std::cout << "UKS physical state, spin, warm and stale-input gates passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
