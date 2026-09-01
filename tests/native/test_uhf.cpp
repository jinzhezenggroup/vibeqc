#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "molecule/basis.hpp"
#include "scf/fleet.hpp"
#include "scf/rhf.hpp"
#include "vibeqc/vibeqc.h"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

vibeqc::core::System hydrogen_system(double distance, int charge, int multiplicity) {
  vibeqc::core::System system;
  system.atoms = {
      {1, {0.0, 0.0, -0.5 * distance}},
      {1, {0.0, 0.0, 0.5 * distance}},
  };
  system.shells = {
      {0, 0, {{3.42525091, 0.15432897}, {0.62391373, 0.53532814}, {0.16885540, 0.44463454}}},
      {1, 0, {{3.42525091, 0.15432897}, {0.62391373, 0.53532814}, {0.16885540, 0.44463454}}},
  };
  system.charge = charge;
  system.multiplicity = multiplicity;
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          "H2 UHF system normalization failed");
  return system;
}

vibeqc::core::System hydrogen_molecular_ion(double distance) {
  return hydrogen_system(distance, 1, 2);
}

vibeqc::core::System hydrogen_molecular_singlet(double distance) {
  // A neutral H2 UHF state has one occupied orbital in each spin channel.
  // It is the smallest public workload that can expose rejection of either
  // alpha or beta external-density trace independently.
  return hydrogen_system(distance, 0, 1);
}

vibeqc::scf::ScfResult evaluate(double distance, const std::vector<double>* warm = nullptr) {
  vibeqc::scf::ScfOptions options;
  options.max_iterations = 100;
  options.energy_tolerance = 1.0e-12;
  options.density_tolerance = 1.0e-10;
  return vibeqc::scf::run_uhf(hydrogen_molecular_ion(distance), options, warm);
}

void require_cpu_rejects_invalid_spin_trace(const vibeqc::core::System& system,
                                            const vibeqc::scf::ScfOptions& options,
                                            const std::vector<double>& density,
                                            std::size_t spin_offset, const char* message) {
  const std::size_t matrix_size = density.size() / 2;
  std::vector<double> invalid = density;
  for (std::size_t element = 0; element < matrix_size; ++element) {
    invalid[spin_offset + element] = -invalid[spin_offset + element];
  }
  try {
    (void)vibeqc::scf::run_uhf(system, options, &invalid);
  } catch (const std::invalid_argument&) {
    return;
  }
  require(false, message);
}

void verify_cpu_external_spin_trace_validation() {
  const vibeqc::core::System system = hydrogen_molecular_singlet(1.4);
  vibeqc::scf::ScfOptions options;
  options.max_iterations = 100;
  options.energy_tolerance = 1.0e-12;
  options.density_tolerance = 1.0e-10;
  const vibeqc::scf::ScfResult cold = vibeqc::scf::run_uhf(system, options);
  require(cold.converged && cold.density.size() == 8,
          "CPU neutral-H2 UHF trace fixture did not converge");
  const vibeqc::scf::ScfResult warm = vibeqc::scf::run_uhf(system, options, &cold.density);
  require(warm.converged && warm.initial_density_used,
          "CPU neutral-H2 UHF valid external density was rejected");
  require_cpu_rejects_invalid_spin_trace(system, options, cold.density, 0,
                                         "CPU accepted an invalid alpha-spin electron trace");
  require_cpu_rejects_invalid_spin_trace(system, options, cold.density, cold.density.size() / 2,
                                         "CPU accepted an invalid beta-spin electron trace");
}

#if VIBEQC_HAS_CUDA
bool cuda_device_available() {
  // A CUDA-enabled build may still run on a login node without an allocated
  // device. Probe through the public runtime contract so the CPU UHF oracle
  // remains testable there without hiding CUDA failures on allocated workers.
  vibeqc_context_descriptor descriptor{sizeof(vibeqc_context_descriptor), VIBEQC_ABI_VERSION, 0,
                                       VIBEQC_BACKEND_CUDA};
  vibeqc_context* context = nullptr;
  const vibeqc_status status = vibeqc_context_create(&descriptor, &context);
  if (context != nullptr) vibeqc_context_destroy(context);
  return status == VIBEQC_STATUS_SUCCESS;
}

vibeqc::core::System large_ao_atom(bool unrestricted) {
  vibeqc::core::System system;
  system.atoms = {{unrestricted ? 1 : 2, {0.0, 0.0, 0.0}}};
  // Seventeen even-tempered s shells cross the cuBLAS matrix-product
  // threshold while keeping the direct integral/force regression cheap.
  // A factor-three radial spacing avoids exact same-center dependence.
  double exponent = 1.0e-6;
  for (int shell = 0; shell < 17; ++shell) {
    system.shells.push_back({0, 0, {{exponent, 1.0}}});
    exponent *= 3.0;
  }
  system.charge = 0;
  system.multiplicity = unrestricted ? 2 : 1;
  system.basis_representation = VIBEQC_BASIS_SPHERICAL;
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          "large-AO atomic test system normalization failed");
  return system;
}

void verify_cuda_external_spin_trace_validation() {
  const vibeqc::core::System system = hydrogen_molecular_singlet(1.4);
  vibeqc::scf::ScfOptions options;
  options.max_iterations = 100;
  options.energy_tolerance = 1.0e-12;
  options.density_tolerance = 1.0e-10;
  options.screening_tolerance = 1.0e-14;

  vibeqc::scf::CudaRhfBucketPlan* plan = nullptr;
  const std::vector<vibeqc::core::System> systems{system};
  const std::vector<const std::vector<double>*> cold_input{nullptr};
  const auto run_cached = [&](const std::vector<const std::vector<double>*>& density) {
    return vibeqc::scf::run_uhf_cuda_bucket_cached(&plan, systems, options, density, 0, false);
  };

  const std::vector<vibeqc::scf::RhfBucketItem> cold = run_cached(cold_input);
  require(cold.size() == 1 && cold[0].status == VIBEQC_STATUS_SUCCESS && cold[0].scf.converged &&
              cold[0].scf.density.size() == 8,
          "CUDA neutral-H2 UHF trace fixture did not converge");
  const std::vector<double> valid_density = cold[0].scf.density;
  const std::vector<const std::vector<double>*> valid_input{&valid_density};
  const std::vector<vibeqc::scf::RhfBucketItem> warm = run_cached(valid_input);
  require(warm.size() == 1 && warm[0].status == VIBEQC_STATUS_SUCCESS && warm[0].scf.converged &&
              warm[0].scf.initial_density_used,
          "CUDA neutral-H2 UHF valid external density was rejected");

  const std::size_t matrix_size = valid_density.size() / 2;
  std::vector<double> invalid_alpha = valid_density;
  for (std::size_t element = 0; element < matrix_size; ++element) {
    invalid_alpha[element] = -invalid_alpha[element];
  }
  const std::vector<const std::vector<double>*> alpha_input{&invalid_alpha};
  const std::vector<vibeqc::scf::RhfBucketItem> alpha_result = run_cached(alpha_input);
  require(alpha_result.size() == 1 && alpha_result[0].status == VIBEQC_STATUS_INVALID_ARGUMENT,
          "CUDA accepted an invalid alpha-spin electron trace");

  std::vector<double> invalid_beta = valid_density;
  for (std::size_t element = 0; element < matrix_size; ++element) {
    invalid_beta[matrix_size + element] = -invalid_beta[matrix_size + element];
  }
  const std::vector<const std::vector<double>*> beta_input{&invalid_beta};
  const std::vector<vibeqc::scf::RhfBucketItem> beta_result = run_cached(beta_input);
  require(beta_result.size() == 1 && beta_result[0].status == VIBEQC_STATUS_INVALID_ARGUMENT,
          "CUDA accepted an invalid beta-spin electron trace");
  vibeqc::scf::destroy_rhf_cuda_bucket_plan(plan);
}

void verify_cached_cuda_warm_density_sequence(bool unrestricted) {
  const vibeqc::core::System system = large_ao_atom(unrestricted);
  vibeqc::scf::ScfOptions options;
  options.max_iterations = 100;
  options.energy_tolerance = 1.0e-12;
  options.density_tolerance = 1.0e-10;
  options.screening_tolerance = 1.0e-14;

  const vibeqc::scf::ScfResult cpu =
      unrestricted ? vibeqc::scf::run_uhf(system, options) : vibeqc::scf::run_rhf(system, options);
  require(cpu.converged, "large-AO CPU oracle did not converge");

  vibeqc::scf::CudaRhfBucketPlan* plan = nullptr;
  const std::vector<vibeqc::core::System> systems{system};
  const std::vector<const std::vector<double>*> cold_density{nullptr};
  const auto run_cached = [&](const std::vector<const std::vector<double>*>& dm0) {
    return unrestricted
               ? vibeqc::scf::run_uhf_cuda_bucket_cached(&plan, systems, options, dm0, 0, false)
               : vibeqc::scf::run_rhf_cuda_bucket_cached(&plan, systems, options, dm0, 0, false);
  };

  std::vector<vibeqc::scf::RhfBucketItem> cold = run_cached(cold_density);
  require(cold.size() == 1 && cold[0].status == VIBEQC_STATUS_SUCCESS && cold[0].scf.converged,
          "large-AO cached CUDA cold execution failed");
  require(std::abs(cold[0].scf.energy - cpu.energy) < 5.0e-9,
          "large-AO CUDA energy differs from the CPU oracle");

  const std::vector<double> valid_density = cold[0].scf.density;
  const std::vector<const std::vector<double>*> valid_input{&valid_density};
  vibeqc::scf::set_rhf_cuda_bucket_warm_start_updates(plan, false);
  std::vector<vibeqc::scf::RhfBucketItem> first_fixed = run_cached(valid_input);
  std::vector<vibeqc::scf::RhfBucketItem> second_fixed = run_cached(valid_input);
  require(first_fixed[0].status == VIBEQC_STATUS_SUCCESS && first_fixed[0].scf.converged &&
              first_fixed[0].scf.iterations == 1 &&
              second_fixed[0].status == VIBEQC_STATUS_SUCCESS && second_fixed[0].scf.converged &&
              second_fixed[0].scf.iterations == 1,
          "fixed CUDA dm0 did not retain its one-iteration energy baseline");
  double advanced_density_change = 0.0;
  for (std::size_t element = 0; element < valid_density.size(); ++element) {
    advanced_density_change =
        std::max(advanced_density_change,
                 std::abs(first_fixed[0].scf.density[element] - valid_density[element]));
  }
  require(advanced_density_change > 0.0,
          "fixed-dm0 regression did not exercise an advanced resident density");

  std::vector<double> invalid_density = valid_density;
  for (double& value : invalid_density) value = -value;
  const std::vector<const std::vector<double>*> invalid_input{&invalid_density};
  std::vector<vibeqc::scf::RhfBucketItem> invalid = run_cached(invalid_input);
  require(invalid[0].status == VIBEQC_STATUS_INVALID_ARGUMENT,
          "CUDA accepted a non-positive warm-density electron trace");

  // The rejected normalization overwrites device scratch before reporting
  // INVALID_ARGUMENT. Replaying the frozen valid dm0 proves that the resident
  // cache was invalidated without corrupting its separate frozen baseline.
  std::vector<vibeqc::scf::RhfBucketItem> recovered = run_cached(valid_input);
  require(recovered[0].status == VIBEQC_STATUS_SUCCESS && recovered[0].scf.converged &&
              recovered[0].scf.iterations == 1 &&
              std::abs(recovered[0].scf.energy - first_fixed[0].scf.energy) < 5.0e-10,
          "cached CUDA plan did not recover after rejecting a warm density");

  // Re-enabling updates must discard the fixed baseline. Because the current
  // resident density has advanced past valid_density, replaying that old dm0
  // once more must take the ordinary baseline-free (at least two iteration)
  // path. Clearing then invalidates even an exact resident replay.
  vibeqc::scf::set_rhf_cuda_bucket_warm_start_updates(plan, true);
  std::vector<vibeqc::scf::RhfBucketItem> after_enable = run_cached(valid_input);
  require(after_enable[0].status == VIBEQC_STATUS_SUCCESS && after_enable[0].scf.converged &&
              after_enable[0].scf.iterations > 1,
          "re-enabled CUDA warm updates retained a stale frozen baseline");
  const std::vector<double> resident_density = after_enable[0].scf.density;
  const std::vector<const std::vector<double>*> resident_input{&resident_density};
  vibeqc::scf::clear_rhf_cuda_bucket_warm_starts(plan);
  std::vector<vibeqc::scf::RhfBucketItem> after_clear = run_cached(resident_input);
  require(after_clear[0].status == VIBEQC_STATUS_SUCCESS && after_clear[0].scf.converged &&
              after_clear[0].scf.iterations > 1,
          "cleared CUDA warm cache reused a resident energy baseline");
  vibeqc::scf::destroy_rhf_cuda_bucket_plan(plan);
}

void verify_fleet_fixed_warm_start() {
  vibeqc::scf::ScfOptions options;
  options.max_iterations = 100;
  options.energy_tolerance = 1.0e-12;
  options.density_tolerance = 1.0e-10;
  options.screening_tolerance = 1.0e-14;
  vibeqc::scf::FleetPlan fleet({large_ao_atom(false)}, VIBEQC_METHOD_RHF, options, true, true,
                               false, false, 0);

  const std::vector<vibeqc::scf::FleetItemResult> cold = fleet.execute({});
  require(cold.size() == 1 && cold[0].status == VIBEQC_STATUS_SUCCESS && cold[0].scf.converged,
          "CUDA Fleet cold execution for fixed warm-start test failed");
  fleet.set_warm_start_updates(false);
  const std::vector<vibeqc::scf::FleetItemResult> first_fixed = fleet.execute({});
  const std::vector<vibeqc::scf::FleetItemResult> second_fixed = fleet.execute({});
  require(first_fixed[0].warm_start_used && second_fixed[0].warm_start_used &&
              first_fixed[0].scf.iterations == 1 && second_fixed[0].scf.iterations == 1,
          "Fleet did not propagate its fixed dm0/energy baseline to CUDA");

  fleet.set_warm_start_updates(true);
  const std::vector<vibeqc::scf::FleetItemResult> resumed = fleet.execute({});
  const std::vector<vibeqc::scf::FleetItemResult> updated = fleet.execute({});
  require(resumed[0].status == VIBEQC_STATUS_SUCCESS && resumed[0].scf.iterations > 1 &&
              updated[0].status == VIBEQC_STATUS_SUCCESS && updated[0].scf.iterations == 1,
          "Fleet did not resume advancing CUDA warm-start state");

  fleet.clear_warm_starts();
  const std::vector<vibeqc::scf::FleetItemResult> cleared = fleet.execute({});
  require(cleared[0].status == VIBEQC_STATUS_SUCCESS && !cleared[0].warm_start_used &&
              cleared[0].scf.iterations > 1,
          "Fleet clear retained CUDA warm-start state");
}
#endif

}  // namespace

int main() {
  try {
    const vibeqc::scf::ScfResult center = evaluate(1.4);
    require(center.converged, "H2+ UHF did not converge");
    require(std::abs(center.energy - (-0.53851134755010321)) < 2.0e-10,
            "H2+ UHF energy differs from PySCF");
    require(center.density.size() == 8, "UHF warm state does not contain alpha and beta matrices");
    require(std::abs(center.forces[2] - (-0.19038408605055362)) < 3.0e-9 &&
                std::abs(center.forces[5] - 0.19038408605055368) < 3.0e-9,
            "H2+ analytic UHF force differs from PySCF");

    const double step = 1.0e-4;
    const vibeqc::scf::ScfResult plus = evaluate(1.4 + step);
    const vibeqc::scf::ScfResult minus = evaluate(1.4 - step);
    require(plus.converged && minus.converged, "finite-difference UHF points did not converge");
    const double derivative = (plus.energy - minus.energy) / (2.0 * step);
    require(std::abs(center.forces[5] + derivative) < 2.0e-6,
            "analytic UHF force disagrees with energy finite differences");

    const vibeqc::scf::ScfResult warm = evaluate(1.4, &center.density);
    require(warm.converged && warm.initial_density_used,
            "UHF packed spin density was not accepted as a warm start");
    require(std::abs(warm.energy - center.energy) < 2.0e-12,
            "warm UHF energy changed the converged state");
    verify_cpu_external_spin_trace_validation();

#if VIBEQC_HAS_CUDA
    if (cuda_device_available()) {
      vibeqc::scf::ScfOptions cuda_options;
      cuda_options.max_iterations = 100;
      cuda_options.energy_tolerance = 1.0e-12;
      cuda_options.density_tolerance = 1.0e-10;
      const vibeqc::scf::ScfResult cuda =
          vibeqc::scf::run_uhf_cuda(hydrogen_molecular_ion(1.4), cuda_options, 0);
      require(cuda.converged, "CUDA H2+ UHF did not converge");
      require(std::abs(cuda.energy - center.energy) < 2.0e-10,
              "CUDA H2+ UHF energy differs from the CPU oracle");
      require(cuda.density.size() == center.density.size(),
              "CUDA UHF did not retain both spin-density matrices");
      for (std::size_t coordinate = 0; coordinate < center.forces.size(); ++coordinate) {
        require(std::abs(cuda.forces[coordinate] - center.forces[coordinate]) < 3.0e-9,
                "CUDA analytic UHF force differs from the CPU oracle");
      }
      const vibeqc::scf::ScfResult cuda_warm =
          vibeqc::scf::run_uhf_cuda(hydrogen_molecular_ion(1.4), cuda_options, 0, &cuda.density);
      require(cuda_warm.converged && cuda_warm.initial_density_used,
              "CUDA UHF packed spin density was not accepted as a warm start");
      verify_cuda_external_spin_trace_validation();
      verify_cached_cuda_warm_density_sequence(false);
      verify_cached_cuda_warm_density_sequence(true);
      verify_fleet_fixed_warm_start();
    } else {
      std::cout << "CUDA UHF checks skipped: no allocated CUDA device\n";
    }
#endif

    std::cout << "H2+ UHF energy: " << center.energy << '\n';
    std::cout << "H2+ UHF force z(atom 1): " << center.forces[5] << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
