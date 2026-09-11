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

bool cuda_device_available() {
  vibeqc_context_descriptor descriptor{sizeof(vibeqc_context_descriptor), VIBEQC_ABI_VERSION, 0,
                                       VIBEQC_BACKEND_CUDA};
  vibeqc_context* context = nullptr;
  const vibeqc_status status = vibeqc_context_create(&descriptor, &context);
  if (context != nullptr) vibeqc_context_destroy(context);
  return status == VIBEQC_STATUS_SUCCESS;
}

/**
 * Compact s/p/d system above the persistent-ERI boundary.
 *
 * Two radial s functions and one p/d shell per center keep the overlap well
 * conditioned while guaranteeing angular-order-three-and-higher direct Fock
 * tasks. RHF uses the two-electron cation; UHF uses its one-electron doublet.
 */
vibeqc::core::System mixed_precision_system(bool unrestricted) {
  vibeqc::core::System system;
  system.atoms = {{2, {0.0, 0.0, -0.8}}, {1, {0.0, 0.0, 0.8}}};
  system.shells = {
      {0, 0, {{4.0, 1.0}}}, {0, 0, {{0.7, 1.0}}}, {0, 1, {{1.2, 1.0}}}, {0, 2, {{0.6, 1.0}}},
      {1, 0, {{2.0, 1.0}}}, {1, 0, {{0.4, 1.0}}}, {1, 1, {{0.8, 1.0}}}, {1, 2, {{0.45, 1.0}}},
  };
  system.charge = unrestricted ? 2 : 1;
  system.multiplicity = unrestricted ? 2 : 1;
  system.basis_representation = VIBEQC_BASIS_SPHERICAL;
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          "mixed-precision system normalization failed");
  require(vibeqc::molecule::ao_count(system) > 16,
          "mixed-precision fixture did not enter direct Fock");
  return system;
}

double maximum_difference(const std::vector<double>& first, const std::vector<double>& second) {
  require(first.size() == second.size(), "mixed-precision result shape changed");
  double maximum = 0.0;
  for (std::size_t index = 0; index < first.size(); ++index) {
    maximum = std::max(maximum, std::abs(first[index] - second[index]));
  }
  return maximum;
}

void verify_mode(bool unrestricted) {
  const vibeqc::core::System system = mixed_precision_system(unrestricted);
  vibeqc::scf::ScfOptions options;
  options.max_iterations = 100;
  options.energy_tolerance = 1.0e-11;
  options.density_tolerance = 1.0e-9;
  options.screening_tolerance = 1.0e-12;

  // Select only dpps so this regression exercises its generated FP64/FP32
  // workers while avoiding the independent generated-ssss baseline failure.
  // Other high-order classes continue through the generic mixed evaluator.
  setenv("VIBEQC_AOT_FOCK_SHELL_CLASSES", "dpps", 1);
  unsetenv("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD");

  vibeqc::scf::CudaRhfBucketPlan* plan = nullptr;
  const std::vector<vibeqc::core::System> systems{system};
  const std::vector<const std::vector<double>*> cold_density{nullptr};
  const auto run_cached = [&](const std::vector<const std::vector<double>*>& dm0) {
    return unrestricted
               ? vibeqc::scf::run_uhf_cuda_bucket_cached(&plan, systems, options, dm0, 0, false)
               : vibeqc::scf::run_rhf_cuda_bucket_cached(&plan, systems, options, dm0, 0, false);
  };

  const std::vector<vibeqc::scf::RhfBucketItem> fp64 = run_cached(cold_density);
  require(fp64.size() == 1 && fp64[0].status == VIBEQC_STATUS_SUCCESS && fp64[0].scf.converged,
          "FP64 direct-Fock reference did not converge");
  const std::vector<const std::vector<double>*> warm_density{&fp64[0].scf.density};

  // Output selection belongs to the cached-plan identity. Reusing a topology
  // after switching to energy-only must omit every force result while keeping
  // the exact final energy; switching back below must restore force work.
  options.compute_forces = false;
  const std::vector<vibeqc::scf::RhfBucketItem> energy_only = run_cached(warm_density);
  require(energy_only.size() == 1 && energy_only[0].status == VIBEQC_STATUS_SUCCESS &&
              energy_only[0].scf.converged && energy_only[0].scf.forces.empty(),
          "energy-only CUDA execution produced an analytic-force result");
  require(std::abs(energy_only[0].scf.energy - fp64[0].scf.energy) < 2.0e-9,
          "energy-only CUDA execution changed the FP64 energy");
  options.compute_forces = true;

  setenv("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD", "auto", 1);
  const std::vector<vibeqc::scf::RhfBucketItem> mixed = run_cached(warm_density);
  require(mixed.size() == 1 && mixed[0].status == VIBEQC_STATUS_SUCCESS && mixed[0].scf.converged &&
              mixed[0].scf.initial_density_used,
          "mixed direct-Fock execution did not converge");
  require(std::abs(mixed[0].scf.energy - fp64[0].scf.energy) < 2.0e-8,
          "mixed direct-Fock energy exceeded its regression tolerance");
  require(maximum_difference(mixed[0].scf.forces, fp64[0].scf.forces) < 2.0e-7,
          "mixed direct-Fock force exceeded its regression tolerance");
  require(maximum_difference(mixed[0].scf.density, fp64[0].scf.density) < 2.0e-5,
          "mixed direct-Fock density exceeded its regression tolerance");

  // Invalid values conservatively disable mixed precision. Changing the
  // setting on a live cached topology must rebuild the plan and recover the
  // ordinary FP64 result instead of replaying stale queue pointers.
  setenv("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD", "invalid", 1);
  const std::vector<vibeqc::scf::RhfBucketItem> invalid = run_cached(warm_density);
  require(
      invalid.size() == 1 && invalid[0].status == VIBEQC_STATUS_SUCCESS && invalid[0].scf.converged,
      "invalid mixed threshold did not fall back to FP64");
  require(std::abs(invalid[0].scf.energy - fp64[0].scf.energy) < 2.0e-9 &&
              maximum_difference(invalid[0].scf.forces, fp64[0].scf.forces) < 2.0e-8 &&
              maximum_difference(invalid[0].scf.density, fp64[0].scf.density) < 2.0e-7,
          "invalid mixed threshold changed the FP64 result");

  // The diagnostic switches alter the captured Graph. Enter and leave the
  // mode on a live plan without changing arithmetic, then toggle only its
  // Fock-only switch; neither transition may reuse an incompatible Graph.
  unsetenv("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD");
  // The isolated streaming route requires complete shell-class coverage.
  // The earlier direct test deliberately restricts its AOT registry to DPPS.
  unsetenv("VIBEQC_AOT_FOCK_SHELL_CLASSES");
  setenv("VIBEQC_BOUNDED_DIRECT_STREAMING", "force", 1);
  setenv("VIBEQC_BOUNDED_DIRECT_FOCK_CLASS_PROFILE", "1", 1);
  setenv("VIBEQC_BOUNDED_DIRECT_FOCK_ONLY_DIAGNOSTIC", "1", 1);
  const auto diagnostic = run_cached(warm_density);
  require(diagnostic.size() == 1 && diagnostic[0].fock_only_diagnostic &&
              diagnostic[0].status == VIBEQC_STATUS_NOT_CONVERGED,
          (std::string("cached plan did not enter isolated Fock mode: ") +
           (diagnostic.empty() ? "empty result" : vibeqc_status_message(diagnostic[0].status)))
              .c_str());
  unsetenv("VIBEQC_BOUNDED_DIRECT_FOCK_ONLY_DIAGNOSTIC");
  const auto restored = run_cached(warm_density);
  require(restored.size() == 1 && !restored[0].fock_only_diagnostic &&
              restored[0].status == VIBEQC_STATUS_SUCCESS && restored[0].scf.converged,
          "cached plan retained its isolated Fock Graph");
  unsetenv("VIBEQC_BOUNDED_DIRECT_FOCK_CLASS_PROFILE");
  unsetenv("VIBEQC_BOUNDED_DIRECT_STREAMING");

  // A diagnostic intentionally reports nonconvergence. The fleet must not
  // interpret that completion as a failed warm SCF and measure a second,
  // cold-density operation. Preserve the same snapshot across every mode.
  vibeqc::scf::FleetPlan fleet(systems, unrestricted ? VIBEQC_METHOD_UHF : VIBEQC_METHOD_RHF,
                               options, true, true, false, false, 0);
  const auto cold = fleet.execute({});
  require(cold[0].status == VIBEQC_STATUS_SUCCESS, "diagnostic fleet setup failed");
  fleet.set_warm_start_updates(false);
  const auto frozen_density = fleet.warm_state(0)->density;
  setenv("VIBEQC_BOUNDED_DIRECT_STREAMING", "force", 1);
  setenv("VIBEQC_BOUNDED_DIRECT_FOCK_CLASS_PROFILE", "1", 1);
  setenv("VIBEQC_BOUNDED_DIRECT_FOCK_ONLY_DIAGNOSTIC", "1", 1);
  for (const char* threshold : {"invalid", "auto", "1e300"}) {
    setenv("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD", threshold, 1);
    for (int replay = 0; replay < 2; ++replay) {
      const auto measured = fleet.execute({});
      require(measured.size() == 1 && measured[0].status == VIBEQC_STATUS_NOT_CONVERGED &&
                  measured[0].warm_start_used && !measured[0].warm_start_fallback,
              "fixed-density diagnostic retried a cold solve");
      require(fleet.warm_state(0)->density == frozen_density,
              "fixed-density diagnostic replaced the frozen snapshot");
    }
  }
  unsetenv("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD");
  unsetenv("VIBEQC_BOUNDED_DIRECT_FOCK_CLASS_PROFILE");
  unsetenv("VIBEQC_BOUNDED_DIRECT_STREAMING");
  unsetenv("VIBEQC_BOUNDED_DIRECT_FOCK_ONLY_DIAGNOSTIC");
  const auto replayed = fleet.execute({});
  require(replayed[0].status == VIBEQC_STATUS_SUCCESS &&
              std::abs(replayed[0].scf.energy - cold[0].scf.energy) < 2.0e-9,
          "diagnostic exit did not restore the full SCF Graph");

  vibeqc::scf::destroy_rhf_cuda_bucket_plan(plan);
}

/**
 * The public \p auto policy must be budget-gated and must finish with an exact
 * FP64 target refinement: the reported convergence, energy, and forces come from
 * continued FP64 iterations after the mixed stage, and the provenance records
 * the certified reserved budget and the refinement cost. A refusal must keep the
 * pure FP64 operator and report it honestly.
 */
void verify_public_auto_policy(bool unrestricted) {
  const vibeqc::core::System system = mixed_precision_system(unrestricted);
  vibeqc::scf::ScfOptions options;
  options.max_iterations = 100;
  options.energy_tolerance = 1.0e-10;
  options.density_tolerance = 1.0e-8;
  options.screening_tolerance = 1.0e-12;
  setenv("VIBEQC_AOT_FOCK_SHELL_CLASSES", "dpps", 1);
  unsetenv("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD");
  vibeqc::scf::CudaRhfBucketPlan* plan = nullptr;
  const std::vector<vibeqc::core::System> systems{system};
  const std::vector<const std::vector<double>*> cold_density{nullptr};
  const auto run_cached = [&](const std::vector<const std::vector<double>*>& dm0) {
    return unrestricted
               ? vibeqc::scf::run_uhf_cuda_bucket_cached(&plan, systems, options, dm0, 0, false)
               : vibeqc::scf::run_rhf_cuda_bucket_cached(&plan, systems, options, dm0, 0, false);
  };
  options.precision_mode = VIBEQC_PRECISION_FP64;
  const std::vector<vibeqc::scf::RhfBucketItem> fp64 = run_cached(cold_density);
  require(fp64.size() == 1 && fp64[0].status == VIBEQC_STATUS_SUCCESS && fp64[0].scf.converged,
          "public FP64 policy did not converge");
  require(fp64[0].scf.precision.requested_mode == VIBEQC_PRECISION_FP64 &&
              fp64[0].scf.precision.effective_bits == 64U &&
              !fp64[0].scf.precision.strict_refinement_applied &&
              fp64[0].scf.precision.refinement_iterations == 0U &&
              fp64[0].scf.precision.mixed_precision_reserved_error == 0.0,
          "explicit FP64 provenance is not honest");
  // Changing the policy on a live prepared bucket must rebuild instead of
  // replaying the FP64 plan.
  const std::vector<const std::vector<double>*> warm_density{&fp64[0].scf.density};
  options.precision_mode = VIBEQC_PRECISION_AUTO;
  const std::vector<vibeqc::scf::RhfBucketItem> automatic = run_cached(warm_density);
  require(automatic.size() == 1 && automatic[0].status == VIBEQC_STATUS_SUCCESS &&
              automatic[0].scf.converged,
          "public auto policy did not converge");
  const vibeqc::scf::PrecisionProvenance& provenance = automatic[0].scf.precision;
  require(provenance.requested_mode == VIBEQC_PRECISION_AUTO,
          "auto request is not reported as auto");
  if (provenance.effective_bits == 32U) {
    // The budget certified a mixed cutoff, so the exact FP64 refinement must
    // have continued the run: convergence is a target-operator statement.
    require(provenance.strict_refinement_applied, "mixed run skipped the FP64 refinement");
    require(provenance.refinement_iterations >= 1U, "mixed run reported no refinement iterations");
    require(provenance.mixed_precision_fock_threshold > 0.0, "mixed run resolved no cutoff");
    require(provenance.mixed_precision_reserved_error > 0.0,
            "mixed run reported no reserved error budget");
  } else {
    require(provenance.effective_bits == 64U && !provenance.strict_refinement_applied &&
                provenance.refinement_iterations == 0U &&
                provenance.mixed_precision_reserved_error == 0.0,
            "FP64 fallback provenance is not honest");
  }
  // The refinement must not change the converged observable: the auto solve
  // starts from the FP64 density, so the refined result must still agree.
  require(std::abs(automatic[0].scf.energy - fp64[0].scf.energy) < 2.0e-8,
          "public auto energy diverged from FP64");
  require(maximum_difference(automatic[0].scf.forces, fp64[0].scf.forces) < 2.0e-7,
          "public auto forces diverged from FP64");
  vibeqc::scf::destroy_rhf_cuda_bucket_plan(plan);
}
/**
 * The public auto policy is per item: one batch keeps a cold item on the exact
 * operator while a warm item uses the mixed route and refines it in FP64. This
 * pins per-item admission, staging and provenance on a real device.
 */
void verify_per_item_auto_policy(bool unrestricted) {
  const vibeqc::core::System system = mixed_precision_system(unrestricted);
  vibeqc::scf::ScfOptions options;
  options.max_iterations = 100;
  options.energy_tolerance = 1.0e-10;
  options.density_tolerance = 1.0e-8;
  options.screening_tolerance = 1.0e-12;
  setenv("VIBEQC_AOT_FOCK_SHELL_CLASSES", "dpps", 1);
  unsetenv("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD");
  vibeqc::scf::CudaRhfBucketPlan* plan = nullptr;
  const std::vector<vibeqc::core::System> systems{system, system};
  const std::vector<const std::vector<double>*> cold{nullptr, nullptr};
  const auto run_cached = [&](const std::vector<const std::vector<double>*>& dm0) {
    return unrestricted
               ? vibeqc::scf::run_uhf_cuda_bucket_cached(&plan, systems, options, dm0, 0, false)
               : vibeqc::scf::run_rhf_cuda_bucket_cached(&plan, systems, options, dm0, 0, false);
  };
  options.precision_mode = VIBEQC_PRECISION_FP64;
  const std::vector<vibeqc::scf::RhfBucketItem> reference = run_cached(cold);
  require(reference.size() == 2 && reference[0].status == VIBEQC_STATUS_SUCCESS &&
              reference[1].status == VIBEQC_STATUS_SUCCESS && reference[0].scf.converged &&
              reference[1].scf.converged,
          "per-item FP64 reference did not converge");
  // Item 0 stays cold; item 1 starts from its validated converged density.
  const std::vector<const std::vector<double>*> ragged{nullptr, &reference[1].scf.density};
  options.precision_mode = VIBEQC_PRECISION_AUTO;
  const std::vector<vibeqc::scf::RhfBucketItem> per_item = run_cached(ragged);
  require(per_item.size() == 2 && per_item[0].status == VIBEQC_STATUS_SUCCESS &&
              per_item[1].status == VIBEQC_STATUS_SUCCESS && per_item[0].scf.converged &&
              per_item[1].scf.converged,
          "per-item auto policy did not converge");
  require(!per_item[0].scf.initial_density_used && per_item[1].scf.initial_density_used,
          "per-item starting states are not distinguishable");
  const vibeqc::scf::PrecisionProvenance& cold_item = per_item[0].scf.precision;
  const vibeqc::scf::PrecisionProvenance& warm_item = per_item[1].scf.precision;
  require(cold_item.requested_mode == VIBEQC_PRECISION_AUTO &&
              warm_item.requested_mode == VIBEQC_PRECISION_AUTO,
          "per-item provenance lost the requested policy");
  require(cold_item.effective_bits == 64U && !cold_item.strict_refinement_applied &&
              cold_item.refinement_iterations == 0U &&
              cold_item.mixed_precision_fock_threshold == 0.0 &&
              cold_item.mixed_precision_reserved_error == 0.0,
          "a cold item must keep the exact FP64 operator");
  require(warm_item.effective_bits == 32U && warm_item.strict_refinement_applied &&
              warm_item.refinement_iterations >= 1U &&
              warm_item.mixed_precision_fock_threshold > 0.0 &&
              warm_item.mixed_precision_reserved_error > 0.0,
          "a warm item must use the mixed route and refine it in FP64");
  for (std::size_t index = 0; index < 2; ++index) {
    require(std::abs(per_item[index].scf.energy - reference[index].scf.energy) < 2.0e-8,
            "per-item auto energy diverged from FP64");
    require(maximum_difference(per_item[index].scf.forces, reference[index].scf.forces) < 2.0e-7,
            "per-item auto forces diverged from FP64");
  }
  vibeqc::scf::destroy_rhf_cuda_bucket_plan(plan);
}
}  // namespace

int main() {
  try {
#if VIBEQC_HAS_CUDA
    if (!cuda_device_available()) {
      std::cout << "mixed-precision checks skipped: no allocated CUDA device\n";
      return EXIT_SUCCESS;
    }
    verify_mode(false);
    verify_mode(true);
    verify_public_auto_policy(false);
    verify_public_auto_policy(true);
    verify_per_item_auto_policy(false);
    verify_per_item_auto_policy(true);
    std::cout << "validated RHF/UHF mixed direct-Fock, public auto and per-item policies\n";
#else
    std::cout << "mixed-precision checks skipped: CUDA disabled\n";
#endif
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
