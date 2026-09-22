#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/cosx_reference.hpp"
#include "dft/cosx_scf.hpp"
#include "molecule/basis.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/reference/mean_field.hpp"

namespace {

using vibeqc::scf::reference::Matrix;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

vibeqc::core::System hydrogen_dimer(int charge, int multiplicity) {
  vibeqc::core::System system;
  system.atoms = {{1, {0.0, 0.0, -0.7}}, {1, {0.0, 0.0, 0.7}}};
  system.shells = {
      {0,
       0,
       {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}},
      {1,
       0,
       {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}},
  };
  system.charge = charge;
  system.multiplicity = multiplicity;
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          "COSX SCF hydrogen fixture normalization failed");
  return system;
}

vibeqc::scf::ResolvedFockBuild mixed_strategy(vibeqc::scf::FockSpin spin,
                                              unsigned derivative_order = 0) {
  using namespace vibeqc::scf;
  auto spec = make_hf_fock_spec(spin);
  spec.derivative_order = derivative_order;
  spec.coulomb.approximation = FockApproximation::DensityFitted;
  spec.exchange.approximation = FockApproximation::SeminumericalCosx;
  spec.exchange.cosx = make_cosx_v1_spec(6, 4, 8, 3, 1.0e-12);
  return resolve_fock_build(spec, FockBackend::Cuda, 1.0e-12, 1.0e-10);
}

vibeqc::scf::ResolvedFockBuild cpu_j_strategy(const vibeqc::scf::ResolvedFockBuild& mixed) {
  auto spec = mixed.spec;
  spec.exchange.present = false;
  return vibeqc::scf::resolve_fock_build(spec, vibeqc::scf::FockBackend::Cpu,
                                         mixed.screening_tolerance,
                                         mixed.metric_relative_threshold);
}

struct PhysicalCheck {
  double energy{};
  double residual{};
};

PhysicalCheck independent_rhf(const vibeqc::core::System& system,
                              const vibeqc::dft::PreparedCosxFockPlan& gpu,
                              const std::vector<double>& density) {
  using namespace vibeqc;
  scf::PreparedFockPlan cpu_j(system, &system, cpu_j_strategy(gpu.strategy()));
  auto jk = cpu_j.build(density);
  jk.exchange_alpha =
      dft::build_cosx_reference(system, gpu.grid().points(), gpu.grid().weights(), density,
                                dft::CosxDensityConvention::rhf_spin_summed)
          .exchange;
  const auto fock = scf::assemble_fock(gpu.strategy(), gpu.one_electron().hcore, jk).alpha;
  const auto residual = scf::reference::commutator_residual(
      fock, density, gpu.one_electron().overlap, gpu.one_electron().nbf);
  return {scf::reference::electronic_energy(density, gpu.one_electron().hcore, fock) +
              gpu.one_electron().nuclear_repulsion,
          scf::reference::residual_rms(residual)};
}

PhysicalCheck independent_uhf(const vibeqc::core::System& system,
                              const vibeqc::dft::PreparedCosxFockPlan& gpu,
                              const std::vector<double>& joined) {
  using namespace vibeqc;
  const auto n = gpu.one_electron().nbf;
  auto [alpha, beta] = scf::reference::split_spin_matrices(joined, n * n);
  scf::PreparedFockPlan cpu_j(system, &system, cpu_j_strategy(gpu.strategy()));
  auto jk = cpu_j.build(alpha, beta);
  jk.exchange_alpha = dft::build_cosx_reference(system, gpu.grid().points(), gpu.grid().weights(),
                                                alpha, dft::CosxDensityConvention::spin_resolved)
                          .exchange;
  jk.exchange_beta = dft::build_cosx_reference(system, gpu.grid().points(), gpu.grid().weights(),
                                               beta, dft::CosxDensityConvention::spin_resolved)
                         .exchange;
  const auto fock = scf::assemble_fock(gpu.strategy(), gpu.one_electron().hcore, jk);
  const auto ra =
      scf::reference::commutator_residual(fock.alpha, alpha, gpu.one_electron().overlap, n);
  const auto rb =
      scf::reference::commutator_residual(fock.beta, beta, gpu.one_electron().overlap, n);
  const double residual_a = scf::reference::residual_rms(ra);
  const double residual_b = scf::reference::residual_rms(rb);
  return {scf::reference::uhf_electronic_energy(alpha, beta, gpu.one_electron().hcore, fock.alpha,
                                                fock.beta) +
              gpu.one_electron().nuclear_repulsion,
          std::hypot(residual_a, residual_b) / std::sqrt(2.0)};
}

vibeqc::scf::ScfOptions options() {
  vibeqc::scf::ScfOptions out;
  out.compute_forces = false;
  out.max_iterations = 100;
  out.diis_history = 8;
  out.energy_tolerance = 1.0e-10;
  out.density_tolerance = 1.0e-8;
  return out;
}

void verify_rhf(int device) {
  using namespace vibeqc;
  const auto system = hydrogen_dimer(0, 1);
  const auto strategy = mixed_strategy(scf::FockSpin::Restricted);
  dft::PreparedCosxFockPlan plan(system, &system, strategy, 16, device);

  auto control = options();
  const auto cold = dft::run_cosx_rhf(plan, control);
  require(cold.converged && cold.forces.empty() && cold.iterations > 1 &&
              cold.fock_builds == cold.iterations + 2 && cold.physical_residual_rms < 1.0e-8,
          "COSX RHF cold SCF did not return a converged physical state");
  const auto check = independent_rhf(system, plan, cold.density);
  require(std::abs(check.energy - cold.energy) < 5.0e-9 && check.residual < 1.0e-8,
          "COSX RHF returned E/D do not match independent DF-J/COSX-K physics");

  const auto warm = dft::run_cosx_rhf(plan, control, &cold.density);
  require(
      warm.converged && warm.initial_density_used && std::abs(warm.energy - cold.energy) < 5.0e-9,
      "COSX RHF warm replay changed the physical endpoint");

  auto strict = control;
  strict.strict_initial_density = true;
  const auto strict_warm = dft::run_cosx_rhf(plan, strict, &cold.density);
  require(strict_warm.converged && std::abs(strict_warm.energy - cold.energy) < 5.0e-9,
          "COSX RHF strict warm replay rejected its own determinant");

  auto forces = control;
  forces.compute_forces = true;
  const auto force_strategy = mixed_strategy(scf::FockSpin::Restricted, 1);
  dft::PreparedCosxFockPlan force_plan(system, &system, force_strategy, 16, device);
  const auto forced = dft::run_cosx_rhf(force_plan, forces, &cold.density);
  require(forced.converged && forced.forces.size() == 3 * system.atoms.size() &&
              std::abs(forced.energy - cold.energy) < 5.0e-9,
          "COSX RHF force endpoint changed the value semantics");
  for (unsigned axis = 0; axis < 3; ++axis) {
    double translation = 0.0;
    for (std::size_t atom = 0; atom < system.atoms.size(); ++atom)
      translation += forced.forces[3 * atom + axis];
    require(std::abs(translation) < 2.0e-7,
            "COSX RHF analytic force violates translational invariance");
  }
  const auto displaced_energy = [&](double displacement) {
    auto displaced = system;
    displaced.atoms[1].position[2] += displacement;
    dft::PreparedCosxFockPlan displaced_plan(displaced, &displaced,
                                             mixed_strategy(scf::FockSpin::Restricted), 16, device);
    const auto endpoint = dft::run_cosx_rhf(displaced_plan, control);
    require(endpoint.converged, "displaced COSX RHF finite-difference endpoint did not converge");
    return endpoint.energy;
  };
  const auto central = [&](double step) {
    return (displaced_energy(step) - displaced_energy(-step)) / (2.0 * step);
  };
  const double coarse = central(2.0e-4), fine = central(1.0e-4);
  const double extrapolated = (4.0 * fine - coarse) / 3.0;
  require(std::abs(forced.forces[5] + extrapolated) < 4.0e-5,
          "COSX RHF analytic force disagrees with multi-step SCF energy finite differences");

  auto one = control;
  one.max_iterations = 1;
  const auto failed = dft::run_cosx_rhf(plan, one);
  require(!failed.converged && failed.iterations == 1 && failed.fock_builds == 1 &&
              failed.density.size() == plan.one_electron().nbf * plan.one_electron().nbf,
          "failed COSX RHF published a converged/finalized state");
  const auto failed_check = independent_rhf(system, plan, failed.density);
  require(std::abs(failed_check.energy - failed.energy) < 5.0e-9 &&
              std::abs(failed_check.residual - failed.physical_residual_rms) < 5.0e-9,
          "failed COSX RHF returned diagnostics from a different density");
}

void verify_uhf(int device) {
  using namespace vibeqc;
  const auto system = hydrogen_dimer(1, 2);
  const auto strategy = mixed_strategy(scf::FockSpin::Unrestricted);
  dft::PreparedCosxFockPlan plan(system, &system, strategy, 16, device);

  auto control = options();
  const auto cold = dft::run_cosx_uhf(plan, control);
  require(cold.converged && cold.forces.empty() && cold.iterations > 1 &&
              cold.fock_builds == cold.iterations + 2 && cold.physical_residual_rms < 1.0e-8,
          "COSX UHF cold SCF did not return a converged physical state");
  const auto check = independent_uhf(system, plan, cold.density);
  require(std::abs(check.energy - cold.energy) < 5.0e-9 && check.residual < 1.0e-8,
          "COSX UHF returned E/D do not match independent DF-J/COSX-K physics");

  const auto warm = dft::run_cosx_uhf(plan, control, &cold.density);
  require(
      warm.converged && warm.initial_density_used && std::abs(warm.energy - cold.energy) < 5.0e-9,
      "COSX UHF warm replay changed the physical endpoint");

  auto forces = control;
  forces.compute_forces = true;
  const auto force_strategy = mixed_strategy(scf::FockSpin::Unrestricted, 1);
  dft::PreparedCosxFockPlan force_plan(system, &system, force_strategy, 16, device);
  const auto forced = dft::run_cosx_uhf(force_plan, forces, &cold.density);
  require(forced.converged && forced.forces.size() == 3 * system.atoms.size() &&
              std::abs(forced.energy - cold.energy) < 5.0e-9,
          "COSX UHF force endpoint changed the value semantics");
  for (unsigned axis = 0; axis < 3; ++axis) {
    double translation = 0.0;
    for (std::size_t atom = 0; atom < system.atoms.size(); ++atom)
      translation += forced.forces[3 * atom + axis];
    require(std::abs(translation) < 2.0e-7,
            "COSX UHF analytic force violates translational invariance");
  }
  const auto displaced_energy = [&](double displacement) {
    auto displaced = system;
    displaced.atoms[1].position[2] += displacement;
    dft::PreparedCosxFockPlan displaced_plan(
        displaced, &displaced, mixed_strategy(scf::FockSpin::Unrestricted), 16, device);
    const auto endpoint = dft::run_cosx_uhf(displaced_plan, control);
    require(endpoint.converged, "displaced COSX UHF finite-difference endpoint did not converge");
    return endpoint.energy;
  };
  const auto central = [&](double step) {
    return (displaced_energy(step) - displaced_energy(-step)) / (2.0 * step);
  };
  const double coarse = central(2.0e-4), fine = central(1.0e-4);
  const double extrapolated = (4.0 * fine - coarse) / 3.0;
  require(std::abs(forced.forces[5] + extrapolated) < 4.0e-5,
          "COSX UHF analytic force disagrees with multi-step SCF energy finite differences");

  auto one = control;
  one.max_iterations = 1;
  const auto failed = dft::run_cosx_uhf(plan, one);
  require(!failed.converged && failed.iterations == 1 && failed.fock_builds == 1 &&
              failed.density.size() == 2 * plan.one_electron().nbf * plan.one_electron().nbf,
          "failed COSX UHF published a converged/finalized state");
  const auto failed_check = independent_uhf(system, plan, failed.density);
  require(std::abs(failed_check.energy - failed.energy) < 5.0e-9 &&
              std::abs(failed_check.residual - failed.physical_residual_rms) < 5.0e-9,
          "failed COSX UHF returned diagnostics from a different density");
}

}  // namespace

int main() {
  try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    verify_rhf(0);
    verify_uhf(0);
    std::cout << "prepared COSX RHF/UHF value and analytic-force SCF PASS\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
