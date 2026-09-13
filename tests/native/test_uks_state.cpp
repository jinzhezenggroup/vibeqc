#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "dft/xc.hpp"
#include "molecule/basis.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/mean_field.hpp"
#include "scf/reference/mean_field.hpp"

namespace {
using namespace vibeqc;

void require(bool passed, const char* message) {
  if (!passed) throw std::runtime_error(message);
}

void check_state(bool pbe, unsigned max_iterations, bool hydroxyl = false) {
  // Asymmetric H3 has two occupied alpha orbitals and one beta orbital.
  // Its initial core density is not stationary, unlike a one-AO atom.
  core::System system;
  system.multiplicity = 2;
  for (unsigned i = 0; i < 3; ++i) {
    system.atoms.push_back({1, {0.15 * i * i, 0.13 * i, 1.5 * i}});
    system.shells.push_back(
        {i,
         0,
         {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}});
  }
  if (hydroxyl) {
    // Linear OH has symmetry-related pi occupations. LDA previously flipped
    // the beta determinant indefinitely despite stationary energy/residual.
    system.atoms = {{8, {0, 0, 0}}, {1, {0, 0, 1.8}}};
    system.shells = {
        {0,
         0,
         {{130.7093214, 0.1543289673}, {23.80886605, 0.5353281423}, {6.443608313, 0.4446345422}}},
        {0,
         0,
         {{5.033151319, -0.09996722919}, {1.169596125, 0.3995128261}, {0.38038896, 0.7001154689}}},
        {0,
         1,
         {{5.033151319, 0.155916275}, {1.169596125, 0.6076837186}, {0.38038896, 0.3919573931}}},
        {1,
         0,
         {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}}};
  }
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          "invalid UKS state fixture");
  scf::FockBuildSpec spec;
  spec.spin = scf::FockSpin::Unrestricted;
  spec.exchange.present = false;
  spec.derivative_order = 0;
  const auto strategy = scf::resolve_fock_build(spec, scf::FockBackend::Cpu);
  const scf::PreparedFockPlan plan(system, nullptr, strategy);
  const dft::AoBasis basis(system);
  const dft::MolecularGrid grid(system,
                                hydroxyl ? dft::GridSpec{} : dft::GridSpec{1, 12, 6, 12, 3, 1e-12});
  scf::ScfOptions options;
  options.compute_forces = false;
  options.max_iterations = max_iterations;
  options.energy_tolerance = 1e-10;
  options.density_tolerance = 1e-7;
  if (hydroxyl) {
    options.energy_tolerance = 1e-12;
    options.density_tolerance = 1e-10;
  }
  const auto result = pbe ? scf::run_pbe_uks(plan, basis, grid, options)
                          : scf::run_lda_uks(plan, basis, grid, options);
  require(result.converged == (max_iterations > 1), "unexpected UKS convergence status");
  const std::size_t n = plan.one_electron().nbf;
  const auto [alpha, beta] = scf::reference::split_spin_matrices(result.density, n * n);
  const auto xc = pbe ? dft::integrate_pbe_uks_with_tail(basis, grid, alpha, beta)
                      : dft::integrate_lda_xc_pw_uks(basis, grid, alpha, beta);
  const auto jk = plan.build(alpha, beta);
  const auto& ints = plan.one_electron();
  const auto total = [&] {
    auto density = alpha;
    for (std::size_t i = 0; i < density.size(); ++i) density[i] += beta[i];
    return density;
  }();
  const double energy = ints.nuclear_repulsion + scf::reference::dot(total, ints.hcore) +
                        0.5 * scf::reference::dot(total, jk.coulomb) + xc.energy;
  require(std::abs(result.energy - energy) < 1e-12,
          "UKS returned energy and density from different iterations");
  auto fock = scf::assemble_fock(strategy, ints.hcore, jk);
  for (std::size_t i = 0; i < alpha.size(); ++i) {
    fock.alpha[i] += xc.potential[0][i];
    fock.beta[i] += xc.potential[1][i];
  }
  const auto residual = scf::reference::concatenate(
      scf::reference::commutator_residual(fock.alpha, alpha, ints.overlap, n),
      scf::reference::commutator_residual(fock.beta, beta, ints.overlap, n));
  const double physical_rms = scf::reference::residual_rms(residual);
  require(std::abs(physical_rms - result.physical_residual_rms) < 1e-13,
          "UKS residual does not describe the returned density");
  require(std::abs(scf::reference::dot(alpha, ints.overlap) - (hydroxyl ? 5.0 : 2.0)) < 1e-12 &&
              std::abs(scf::reference::dot(beta, ints.overlap) - (hydroxyl ? 4.0 : 1.0)) < 1e-12,
          "UKS changed spin populations");
  for (const auto* spin : {&alpha, &beta}) {
    const auto projected =
        scf::reference::multiply(scf::reference::multiply(*spin, ints.overlap, n), *spin, n);
    require(scf::reference::density_rms(projected, *spin) < 1e-12,
            "UKS occupation stabilization lost the integer-occupation projector");
  }
  if (result.converged)
    require(physical_rms < options.density_tolerance &&
                result.density_rms < options.density_tolerance &&
                result.energy_change < options.energy_tolerance,
            "UKS returned success without passing all physical-state gates");
}
}  // namespace

int main() {
  try {
    for (bool pbe : {false, true}) {
      for (unsigned iterations : {1U, 150U}) check_state(pbe, iterations);
      check_state(pbe, 200, true);
    }
    std::cout << "UKS accepted and exhausted-budget states are physically consistent\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
