#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/cosx_fock_provider.hpp"
#include "dft/cosx_reference.hpp"
#include "molecule/basis.hpp"
#include "scf/fock_build.hpp"
#include "scf/fock_prepared.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

vibeqc::core::System h2() {
  vibeqc::core::System system;
  system.atoms = {{1, {0.0, 0.0, 0.0}}, {1, {0.1, 0.2, 1.4}}};
  system.shells = {
      {0,
       0,
       {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}},
      {1,
       0,
       {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}},
  };
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          "COSX provider H2 normalization failed");
  return system;
}

double max_error(const std::vector<double>& first, const std::vector<double>& second) {
  require(first.size() == second.size(), "COSX provider comparison size mismatch");
  double error = 0.0;
  for (std::size_t i = 0; i < first.size(); ++i)
    error = std::max(error, std::abs(first[i] - second[i]));
  return error;
}

vibeqc::scf::ResolvedFockBuild mixed_strategy(vibeqc::scf::FockSpin spin,
                                              unsigned derivative_order = 0) {
  using namespace vibeqc::scf;
  auto spec = make_hf_fock_spec(spin);
  spec.derivative_order = derivative_order;
  spec.coulomb.approximation = FockApproximation::DensityFitted;
  spec.exchange.approximation = FockApproximation::SeminumericalCosx;
  spec.exchange.cosx = make_cosx_v1_spec(12, 8, 16, 3, 1.0e-12);
  return resolve_fock_build(spec, FockBackend::Cuda, 1.0e-12, 1.0e-10);
}

vibeqc::scf::ResolvedFockBuild cpu_j_strategy(const vibeqc::scf::ResolvedFockBuild& mixed) {
  auto spec = mixed.spec;
  spec.exchange.present = false;
  return vibeqc::scf::resolve_fock_build(spec, vibeqc::scf::FockBackend::Cpu,
                                         mixed.screening_tolerance,
                                         mixed.metric_relative_threshold);
}

void verify_restricted(const vibeqc::core::System& system, int device) {
  using namespace vibeqc;
  const auto strategy = mixed_strategy(scf::FockSpin::Restricted);
  bool legacy_rejected = false;
  try {
    scf::PreparedFockPlan invalid(system, &system, strategy, device);
  } catch (const std::invalid_argument&) {
    legacy_rejected = true;
  }
  require(legacy_rejected,
          "legacy PreparedFockPlan silently routed COSX through an exact/DF provider");

  dft::PreparedCosxFockPlan gpu(system, &system, strategy, 7, device);

  const std::vector<double> density{0.8, 0.2, 0.2, 0.6};
  const auto actual = gpu.build(density);

  scf::PreparedFockPlan cpu_j(system, &system, cpu_j_strategy(strategy));
  auto expected = cpu_j.build(density);
  const auto reference_k =
      dft::build_cosx_reference(system, gpu.grid().points(), gpu.grid().weights(), density,
                                dft::CosxDensityConvention::rhf_spin_summed);
  expected.exchange_alpha = reference_k.exchange;

  require(max_error(actual.coulomb, expected.coulomb) < 3.0e-10,
          "prepared RI-J differs from the CPU DF oracle");
  require(max_error(actual.exchange_alpha, expected.exchange_alpha) < 3.0e-12 &&
              actual.exchange_beta.empty(),
          "prepared COSX-K differs from the CPU RHF oracle");

  const double actual_energy = scf::contract_fock_energy(strategy, actual, density);
  const double expected_energy = scf::contract_fock_energy(strategy, expected, density);
  require(std::abs(actual_energy - expected_energy) < 3.0e-10,
          "prepared RI-J/COSX-K two-electron energy differs from the independent oracles");

  const auto actual_fock = scf::assemble_fock(strategy, gpu.one_electron().hcore, actual);
  const auto expected_fock = scf::assemble_fock(strategy, gpu.one_electron().hcore, expected);
  require(max_error(actual_fock.alpha, expected_fock.alpha) < 3.0e-10 && actual_fock.beta.empty(),
          "prepared RI-J/COSX-K RHF assembly differs from the independent oracles");

  const auto& diagnostic = gpu.diagnostic();
  require(diagnostic.strategy == strategy &&
              diagnostic.coulomb.strategy.spec.exchange.present == false &&
              diagnostic.exchange.esp_on_device && diagnostic.exchange.assembly_on_device &&
              diagnostic.device_bytes <= diagnostic.device_budget_bytes &&
              diagnostic.tile_points == 7,
          "prepared COSX diagnostic lost provider identity or bounded resources");
  const auto& model = strategy.spec.exchange.cosx;
  require(gpu.grid().spec().version == model.grid_version &&
              gpu.grid().spec().radial_points == model.radial_points &&
              gpu.grid().spec().angular_polar == model.angular_polar &&
              gpu.grid().spec().angular_azimuth == model.angular_azimuth &&
              gpu.grid().spec().partition_iterations == model.partition_iterations &&
              gpu.grid().spec().coincident_tolerance == model.coincident_tolerance &&
              gpu.grid().spec().element_radii == model.element_radii,
          "prepared COSX grid does not reproduce the resolved mathematical identity");

  const auto cosx_only = dft::cuda_cosx_staging_diagnostic(system, gpu.grid().point_count(), 7);
  bool budget_rejected = false;
  try {
    dft::PreparedCosxFockPlan too_small(system, &system, strategy, 7, device,
                                        cosx_only.device_bytes);
  } catch (const std::bad_alloc&) {
    budget_rejected = true;
  }
  require(budget_rejected,
          "prepared RI-J/COSX-K accepted a budget with no capacity for the J provider");

  const auto force_strategy = mixed_strategy(scf::FockSpin::Restricted, 1);
  dft::PreparedCosxFockPlan force_gpu(system, &system, force_strategy, 7, device);
  scf::PreparedFockPlan force_cpu_j(system, &system, cpu_j_strategy(force_strategy));
  auto expected_derivative = force_cpu_j.energy_derivative(density);
  const auto reference_derivative = dft::build_cosx_molecular_derivative_reference(
      force_gpu.grid(), density, dft::CosxDensityConvention::rhf_spin_summed);
  for (std::size_t coordinate = 0; coordinate < expected_derivative.size(); ++coordinate)
    expected_derivative[coordinate] += reference_derivative.nuclear_gradient[coordinate];
  const auto actual_derivative = force_gpu.energy_derivative(density);
  require(max_error(actual_derivative, expected_derivative) < 3.0e-8,
          "prepared RI-J/COSX-K RHF derivative differs from independent J/K oracles");
  require(force_gpu.diagnostic().derivative.bounded_tiling &&
              force_gpu.diagnostic().derivative_peak_device_bytes <=
                  force_gpu.diagnostic().device_budget_bytes,
          "prepared COSX force provider lost bounded peak-resource accounting");

  auto scaled_spec = force_strategy.spec;
  scaled_spec.exchange.coefficient *= 0.5;
  const auto scaled_strategy = scf::resolve_fock_build(scaled_spec, scf::FockBackend::Cuda,
                                                       force_strategy.screening_tolerance,
                                                       force_strategy.metric_relative_threshold);
  dft::PreparedCosxFockPlan scaled_gpu(system, &system, scaled_strategy, 7, device);
  scf::PreparedFockPlan scaled_cpu_j(system, &system, cpu_j_strategy(scaled_strategy));
  auto scaled_expected = scaled_cpu_j.energy_derivative(density);
  for (std::size_t coordinate = 0; coordinate < scaled_expected.size(); ++coordinate)
    scaled_expected[coordinate] += 0.5 * reference_derivative.nuclear_gradient[coordinate];
  require(max_error(scaled_gpu.energy_derivative(density), scaled_expected) < 3.0e-8,
          "prepared COSX derivative ignored the resolved arbitrary exchange coefficient");
}

void verify_unrestricted(const vibeqc::core::System& system, int device) {
  using namespace vibeqc;
  const auto strategy = mixed_strategy(scf::FockSpin::Unrestricted);
  dft::PreparedCosxFockPlan gpu(system, &system, strategy, 5, device);

  const std::vector<double> alpha{0.45, 0.10, 0.10, 0.35};
  const std::vector<double> beta{0.25, 0.04, 0.04, 0.18};
  const auto actual = gpu.build(alpha, beta);

  scf::PreparedFockPlan cpu_j(system, &system, cpu_j_strategy(strategy));
  auto expected = cpu_j.build(alpha, beta);
  expected.exchange_alpha =
      dft::build_cosx_reference(system, gpu.grid().points(), gpu.grid().weights(), alpha,
                                dft::CosxDensityConvention::spin_resolved)
          .exchange;
  expected.exchange_beta =
      dft::build_cosx_reference(system, gpu.grid().points(), gpu.grid().weights(), beta,
                                dft::CosxDensityConvention::spin_resolved)
          .exchange;

  require(max_error(actual.coulomb, expected.coulomb) < 3.0e-10 &&
              max_error(actual.exchange_alpha, expected.exchange_alpha) < 3.0e-12 &&
              max_error(actual.exchange_beta, expected.exchange_beta) < 3.0e-12,
          "prepared UHF RI-J/COSX-K matrices differ from the independent oracles");
  require(std::abs(scf::contract_fock_energy(strategy, actual, alpha, beta) -
                   scf::contract_fock_energy(strategy, expected, alpha, beta)) < 3.0e-10,
          "prepared UHF RI-J/COSX-K energy differs from the independent oracles");

  const auto actual_fock = scf::assemble_fock(strategy, gpu.one_electron().hcore, actual);
  const auto expected_fock = scf::assemble_fock(strategy, gpu.one_electron().hcore, expected);
  require(max_error(actual_fock.alpha, expected_fock.alpha) < 3.0e-10 &&
              max_error(actual_fock.beta, expected_fock.beta) < 3.0e-10,
          "prepared UHF RI-J/COSX-K Fock differs from the independent oracles");

  const auto force_strategy = mixed_strategy(scf::FockSpin::Unrestricted, 1);
  dft::PreparedCosxFockPlan force_gpu(system, &system, force_strategy, 5, device);
  scf::PreparedFockPlan force_cpu_j(system, &system, cpu_j_strategy(force_strategy));
  auto expected_derivative = force_cpu_j.energy_derivative(alpha, beta);
  const auto alpha_reference = dft::build_cosx_molecular_derivative_reference(
      force_gpu.grid(), alpha, dft::CosxDensityConvention::spin_resolved);
  const auto beta_reference = dft::build_cosx_molecular_derivative_reference(
      force_gpu.grid(), beta, dft::CosxDensityConvention::spin_resolved);
  for (std::size_t coordinate = 0; coordinate < expected_derivative.size(); ++coordinate)
    expected_derivative[coordinate] +=
        alpha_reference.nuclear_gradient[coordinate] + beta_reference.nuclear_gradient[coordinate];
  require(max_error(force_gpu.energy_derivative(alpha, beta), expected_derivative) < 3.0e-8,
          "prepared RI-J/COSX-K UHF derivative differs from independent J/K oracles");
}

}  // namespace

int main() {
  try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    verify_restricted(h2(), 0);
    verify_unrestricted(h2(), 0);
    std::cout << "prepared RI-J/COSX-K fixed-density RHF/UHF provider PASS\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
