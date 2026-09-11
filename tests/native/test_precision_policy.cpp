#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "molecule/basis.hpp"
#include "scf/cuda/rhf_policy.hpp"
#include "scf/direct_task_layout.hpp"
#include "scf/rhf.hpp"
#include "vibeqc/vibeqc.h"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void require_close(double actual, double expected, double tolerance, const char* message) {
  require(std::abs(actual - expected) <= tolerance, message);
}

/** RAII environment setter that restores the previous value on scope exit. */
class ScopedEnv {
 public:
  explicit ScopedEnv(const char* name, const char* value) : name_(name) {
    if (value == nullptr) {
      previous_ = std::getenv(name_);
      unsetenv(name_);
    } else {
      previous_ = std::getenv(name_);
      setenv(name_, value, 1);
    }
  }
  ~ScopedEnv() {
    if (previous_ == nullptr)
      unsetenv(name_);
    else
      setenv(name_, previous_, 1);
  }
  const char* name_;
  const char* previous_;
};

using Threshold = std::optional<double>;
using Admission = vibeqc::scf::cuda_policy::AutoMixedPrecisionAdmission;
/** Binary32 unit roundoff, mirrored from the policy constant. */
constexpr double kTestFloat32UnitRoundoff = 5.9604644775390625e-08;
/**
 * The \p auto cutoff is an accumulated-error budget, not a per-tile constant:
 * `eps32 * cutoff * census` must fit the error reserved for the iterative
 * operator, so a census of individually small contributions tightens the cutoff
 * and eventually refuses the mixed route entirely.
 */
void verify_auto_budget_admission() {
  const double screen = 1.0e-12;
  const double energy = 1.0e-10;
  const double reserved = energy / 16.0;
  // Small census: the legacy measured anchor is tighter than the budget, so it
  // still bounds the cutoff.
  const Admission small =
      vibeqc::scf::cuda_policy::admit_auto_mixed_precision_fock(energy, screen, 100.0);
  require(small.admitted, "a 100-tile mixed-capable census fits the budget");
  require_close(small.threshold, 1.0e-6, 1.0e-18, "the anchor caps a small census");
  require_close(small.reserved_error, reserved, 1.0e-24, "the admission reports its budget");
  // Budget-dominated census: many individually eligible tiles collectively
  // exceed the per-tile anchor, so the certified cutoff must shrink.
  const Admission crowded =
      vibeqc::scf::cuda_policy::admit_auto_mixed_precision_fock(energy, screen, 1000.0);
  require(crowded.admitted, "a budget-dominated census is still admissible");
  require_close(crowded.threshold, reserved / (kTestFloat32UnitRoundoff * 1000.0), 1.0e-18,
                "the budget resolves the cutoff for the actual census");
  require(crowded.threshold < small.threshold,
          "a larger census resolves a tighter cutoff than the per-tile anchor");
  // The certified bound is what the cutoff promises: no census may let the
  // accumulated FP32 rounding exceed the reserved error.
  require(kTestFloat32UnitRoundoff * crowded.threshold * crowded.eligible_tiles <=
              crowded.reserved_error * (1.0 + 1.0e-12),
          "the resolved cutoff certifies the accumulated bound");
  // Many individually small contributions, collectively over budget: the
  // certified cutoff falls to the screening floor and the mixed route is
  // refused, leaving the FP64 operator rather than an unbounded accumulation.
  const double floor_census = reserved / (kTestFloat32UnitRoundoff * screen);
  require(
      !vibeqc::scf::cuda_policy::admit_auto_mixed_precision_fock(energy, screen, floor_census * 2.0)
           .admitted,
      "a census past the budget floor refuses the mixed route");
  require(!vibeqc::scf::cuda_policy::admit_auto_mixed_precision_fock(energy, screen,
                                                                     floor_census * 10.0)
               .admitted,
          "further past the budget floor the mixed route stays refused");
  // An absent census cannot be bounded, so it is never admitted by default.
  require(!vibeqc::scf::cuda_policy::admit_auto_mixed_precision_fock(energy, screen, 0.0).admitted,
          "an unknown census is refused");
  require(!vibeqc::scf::cuda_policy::admit_auto_mixed_precision_fock(0.0, screen, 100.0).admitted,
          "a non-positive target is refused");
  require(!vibeqc::scf::cuda_policy::admit_auto_mixed_precision_fock(
               std::numeric_limits<double>::infinity(), screen, 100.0)
               .admitted,
          "a non-finite target is refused");
  // A looser target both raises the anchor and enlarges the reserved budget.
  const Admission looser =
      vibeqc::scf::cuda_policy::admit_auto_mixed_precision_fock(1.0e-6, screen, 100.0);
  require(looser.admitted && looser.threshold > small.threshold,
          "a looser target certifies a larger cutoff");
  require_close(looser.threshold, 1.0e-2, 1.0e-14, "1e-6 target anchors a 1e-2 cutoff");
}
/** Explicit FP64 keeps the pure double path; even a numeric legacy env cannot relax it. */
void verify_fp64_strict() {
  const double screen = 1.0e-12;
  const double energy = 1.0e-10;
  const vibeqc::scf::cuda_policy::MixedPrecisionFockPolicy clean =
      vibeqc::scf::cuda_policy::resolve_mixed_precision_fock_policy(
          std::optional<vibeqc_precision_mode>(VIBEQC_PRECISION_FP64), energy, screen, 100.0);
  require(!clean.threshold.has_value(), "explicit FP64 resolves to the pure double path");
  require(!clean.budget_certified, "explicit FP64 certifies no mixed budget");
  // A numeric diagnostic override is present but FP64 must ignore it.
  ScopedEnv override("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD", "5e-7");
  const vibeqc::scf::cuda_policy::MixedPrecisionFockPolicy with_override =
      vibeqc::scf::cuda_policy::resolve_mixed_precision_fock_policy(
          std::optional<vibeqc_precision_mode>(VIBEQC_PRECISION_FP64), energy, screen, 100.0);
  require(!with_override.threshold.has_value(),
          "FP64 is not relaxed by the legacy diagnostic override");
}
/** AUTO derives from tolerances, but an explicit numeric env acts as a hard override. */
void verify_auto_with_legacy_override() {
  const double screen = 1.0e-12;
  const double energy = 1.0e-10;

  ScopedEnv numeric("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD", "2e-7");
  const vibeqc::scf::cuda_policy::MixedPrecisionFockPolicy overridden =
      vibeqc::scf::cuda_policy::resolve_mixed_precision_fock_policy(
          std::optional<vibeqc_precision_mode>(VIBEQC_PRECISION_AUTO), energy, screen, 100.0);
  require(overridden.threshold.has_value(), "AUTO with a numeric override resolves a mixed route");
  require_close(*overridden.threshold, 2.0e-7, 1.0e-14,
                "numeric override wins over the derived value");
  require(!overridden.budget_certified,
          "a diagnostic override is reported as uncertified, not as budgeted evidence");

  // The 'auto' / '0' / 'none' spellings are not numeric overrides; they fall back
  // to the derived threshold.
  ScopedEnv auto_spelling("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD", "auto");
  // The 'auto' spelling is not a numeric override: the budget decides again.
  const vibeqc::scf::cuda_policy::MixedPrecisionFockPolicy derived =
      vibeqc::scf::cuda_policy::resolve_mixed_precision_fock_policy(
          std::optional<vibeqc_precision_mode>(VIBEQC_PRECISION_AUTO), energy, screen, 100.0);
  require(derived.threshold.has_value() && std::abs(*derived.threshold - 1.0e-6) <= 1.0e-12,
          "AUTO with the 'auto' spelling derives from the budget");
  require(derived.budget_certified, "the derived cutoff is reported as budget-certified");
  ScopedEnv below_floor("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD", "5e-13");
  const vibeqc::scf::cuda_policy::MixedPrecisionFockPolicy below =
      vibeqc::scf::cuda_policy::resolve_mixed_precision_fock_policy(
          std::optional<vibeqc_precision_mode>(VIBEQC_PRECISION_AUTO), energy, screen, 100.0);
  require(below.threshold.has_value() && std::abs(*below.threshold - 1.0e-6) <= 1.0e-12,
          "a sub-floor numeric override falls back to the budgeted cutoff");
  require(below.budget_certified, "the budgeted fallback is certified");
  // A diagnostic cutoff is deliberately item agnostic: it is not a budget, so
  // an item without a census and without a validated warm state still runs the
  // same diagnostic route instead of silently dropping out of it.
  const vibeqc::scf::cuda_policy::MixedPrecisionItemPolicy diagnostic_item =
      vibeqc::scf::cuda_policy::resolve_mixed_precision_item(overridden, false, 0, screen);
  require(diagnostic_item.admitted && diagnostic_item.census == 1U,
          "a diagnostic cutoff stays item agnostic");
}
/**
 * The accumulated-error budget is per item. A batch keeps each system's own
 * mixed-capable census, and every item resolves its own cutoff from it, so a
 * cold item or an item without mixed-capable tiles stays on the exact operator
 * while a warm high-order item in the same batch runs the mixed route.
 */
void verify_per_item_budget() {
  // System 0 owns two s shells only, so no mixed-capable order exists. System 1
  // owns the same s shells plus a d shell, so high-order tiles do exist. Both
  // systems share one batch.
  const std::vector<std::int64_t> shell_ao_offsets{0, 1, 2, 3, 4, 9};
  const std::vector<std::uint8_t> shell_angular{0, 0, 0, 0, 2};
  const std::vector<std::int64_t> system_shell_pair_offsets{0, 3, 9};
  const std::vector<std::int32_t> shell_pair_first{0, 0, 1, 2, 2, 2, 3, 3, 4};
  const std::vector<std::int32_t> shell_pair_second{0, 1, 1, 2, 3, 4, 3, 4, 4};
  vibeqc::scf::detail::DirectQuartetTaskLayout layout;
  require(vibeqc::scf::detail::make_direct_quartet_task_layout(
              shell_ao_offsets, shell_angular, system_shell_pair_offsets, shell_pair_first,
              shell_pair_second, vibeqc::scf::detail::kDirectQuartetMixedFockMinimumAngularOrder,
              layout),
          "per-item census topology was rejected");
  require(layout.system_mixed_capable_tile_counts.size() == 2,
          "the layout must keep one mixed-capable census per system");
  require(layout.system_mixed_capable_tile_counts[0] == 0,
          "an s-only system must report no mixed-capable tile");
  require(layout.system_mixed_capable_tile_counts[1] > 0,
          "a d-shell system must report mixed-capable tiles");
  const double screen = 1.0e-12;
  const double energy = 1.0e-10;
  const std::size_t ceiling = std::max(layout.system_mixed_capable_tile_counts[0],
                                       layout.system_mixed_capable_tile_counts[1]);
  const vibeqc::scf::cuda_policy::MixedPrecisionFockPolicy policy =
      vibeqc::scf::cuda_policy::resolve_mixed_precision_fock_policy(
          std::optional<vibeqc_precision_mode>(VIBEQC_PRECISION_AUTO), energy, screen,
          static_cast<double>(ceiling));
  require(policy.threshold.has_value() && policy.budget_certified,
          "the batch ceiling must certify the route");
  // One batch, three legitimate resolutions: cold, census-free and warm.
  require(!vibeqc::scf::cuda_policy::resolve_mixed_precision_item(
               policy, false, layout.system_mixed_capable_tile_counts[1], screen)
               .admitted,
          "a cold item must keep the exact FP64 operator");
  require(!vibeqc::scf::cuda_policy::resolve_mixed_precision_item(
               policy, true, layout.system_mixed_capable_tile_counts[0], screen)
               .admitted,
          "an item without mixed-capable tiles must keep the exact operator");
  const vibeqc::scf::cuda_policy::MixedPrecisionItemPolicy warm =
      vibeqc::scf::cuda_policy::resolve_mixed_precision_item(
          policy, true, layout.system_mixed_capable_tile_counts[1], screen);
  require(warm.admitted, "a warm mixed-capable item must be admitted");
  require(warm.census == layout.system_mixed_capable_tile_counts[1],
          "the item must report its own census");
  // The item cutoff is the per-item accumulated bound: an item whose census
  // makes that bound the binding term resolves a tighter cutoff than the
  // tolerance anchor, while a small census keeps the anchor.
  const vibeqc::scf::cuda_policy::MixedPrecisionItemPolicy crowded =
      vibeqc::scf::cuda_policy::resolve_mixed_precision_item(policy, true, 1000U, screen);
  const vibeqc::scf::cuda_policy::MixedPrecisionItemPolicy tight =
      vibeqc::scf::cuda_policy::resolve_mixed_precision_item(policy, true, 200U, screen);
  const vibeqc::scf::cuda_policy::MixedPrecisionItemPolicy anchored =
      vibeqc::scf::cuda_policy::resolve_mixed_precision_item(policy, true, 4U, screen);
  require(crowded.admitted && tight.admitted && anchored.admitted,
          "budget-certified items must be admitted");
  require_close(anchored.threshold, *policy.threshold, 1.0e-18,
                "a small census keeps the batch anchor");
  require(tight.threshold < anchored.threshold,
          "a budget-dominated census tightens the item cutoff");
  require(crowded.threshold < tight.threshold, "a larger census resolves an even tighter cutoff");
  const double expected_crowded = policy.item_budget_error / (kTestFloat32UnitRoundoff * 1000.0);
  require_close(crowded.threshold, expected_crowded, std::abs(expected_crowded) * 1.0e-12,
                "the item cutoff is the per-item accumulated bound");
  // An item past the budget floor keeps the exact operator instead of
  // accumulating rounding the requested accuracy cannot certify.
  require(!vibeqc::scf::cuda_policy::resolve_mixed_precision_item(
               policy, true, static_cast<std::size_t>(1.0e9), screen)
               .admitted,
          "an item past the budget floor must be refused");
  // An uncertified batch policy admits no item at all.
  const vibeqc::scf::cuda_policy::MixedPrecisionFockPolicy refused =
      vibeqc::scf::cuda_policy::resolve_mixed_precision_fock_policy(
          std::optional<vibeqc_precision_mode>(VIBEQC_PRECISION_AUTO), energy, screen, 0.0);
  require(!refused.threshold.has_value() &&
              !vibeqc::scf::cuda_policy::resolve_mixed_precision_item(refused, true, 1000U, screen)
                   .admitted,
          "an uncertified batch policy must admit no item");
}
/** A nullopt mode preserves the legacy diagnostic switch verbatim. */
void verify_nullopt_legacy_parity() {
  const double screen = 1.0e-12;
  const double energy = 1.0e-10;
  std::optional<vibeqc_precision_mode> nullopt;

  ScopedEnv absent("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD", "0");
  require(
      !vibeqc::scf::cuda_policy::resolve_mixed_precision_fock_policy(nullopt, energy, screen, 100.0)
           .threshold.has_value(),
      "absent/0 legacy switch keeps the default FP64 path");
  ScopedEnv legacy_auto("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD", "auto");
  const Threshold legacy_auto_threshold =
      vibeqc::scf::cuda_policy::resolve_mixed_precision_fock_policy(nullopt, energy, screen, 100.0)
          .threshold;
  require(legacy_auto_threshold.has_value(), "'auto' legacy switch enables the mixed route");
  require_close(*legacy_auto_threshold, 1.0e-6, 1.0e-12,
                "'auto' legacy switch uses the default 1e-6");
  ScopedEnv legacy_numeric("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD", "3e-7");
  const Threshold legacy_numeric_threshold =
      vibeqc::scf::cuda_policy::resolve_mixed_precision_fock_policy(nullopt, energy, screen, 100.0)
          .threshold;
  require(legacy_numeric_threshold.has_value() &&
              std::abs(*legacy_numeric_threshold - 3.0e-7) <= 1.0e-14,
          "numeric legacy switch passes through verbatim");

  ScopedEnv legacy_too_small("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD", "5e-13");
  require(
      !vibeqc::scf::cuda_policy::resolve_mixed_precision_fock_policy(nullopt, energy, screen, 100.0)
           .threshold.has_value(),
      "a numeric legacy switch at or below the floor keeps the FP64 path");
  ScopedEnv legacy_garbage("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD", "bogus");
  require(
      !vibeqc::scf::cuda_policy::resolve_mixed_precision_fock_policy(nullopt, energy, screen, 100.0)
           .threshold.has_value(),
      "an invalid legacy switch keeps the FP64 path rather than relaxing it");
  // The legacy diagnostic switch is unchanged by the budget: it keeps resolving
  // its own numeric cutoff even when the census would refuse an auto request.
  ScopedEnv legacy_only("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD", "1e-7");
  const Threshold legacy_only_threshold =
      vibeqc::scf::cuda_policy::resolve_mixed_precision_fock_policy(nullopt, energy, screen, 0.0)
          .threshold;
  require(legacy_only_threshold.has_value() && std::abs(*legacy_only_threshold - 1.0e-7) <= 1.0e-19,
          "the legacy diagnostic switch does not consult the budget census");
}
/** A converged-fock reuse RMS scales with the density tolerance. */
void verify_converged_fock_reuse_rms() {
  require_close(vibeqc::scf::cuda_policy::converged_fock_reuse_density_rms(1.0e-12), 1.0e-12,
                1.0e-16, "tight density tolerance keeps the tight reuse RMS");
  require_close(vibeqc::scf::cuda_policy::converged_fock_reuse_density_rms(1.0e-9), 2.0e-9, 1.0e-16,
                "expanded density tolerance widens the reuse RMS");
}

/** Minimal closed-shell H2 used to pin CPU provenance end to end. */
vibeqc::core::System h2_system() {
  vibeqc::core::System system;
  system.atoms = {{1, {0.0, 0.0, -0.7}}, {1, {0.0, 0.0, 0.7}}};
  system.shells = {
      {0, 0, {{1.5, 1.0}, {0.4, 0.5}}},
      {1, 0, {{1.5, 1.0}, {0.4, 0.5}}},
  };
  system.charge = 0;
  system.multiplicity = 1;
  system.basis_representation = VIBEQC_BASIS_SPHERICAL;
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          ("H2 system normalization failed: " + detail).c_str());
  return system;
}

/**
 * The host-plan path always runs the FP64 Fock, so effective_bits stays 64 while
 * requested_mode reports the caller's policy. This pins the honest-provenance fix
 * at the native level: an \p auto request that collapses to FP64 on CPU must still
 * say it was asked for \p auto.
 */
vibeqc::scf::ScfResult run_cpu_rhf(std::optional<vibeqc_precision_mode> precision_mode) {
  const vibeqc::core::System system = h2_system();
  vibeqc::scf::ScfOptions options;
  options.max_iterations = 100;
  options.energy_tolerance = 1.0e-10;
  options.density_tolerance = 1.0e-8;
  options.screening_tolerance = 1.0e-12;
  options.precision_mode = precision_mode;
  return vibeqc::scf::run_rhf(system, options, nullptr);
}

void verify_cpu_provenance() {
  const vibeqc::scf::ScfResult fp64 =
      run_cpu_rhf(std::optional<vibeqc_precision_mode>(VIBEQC_PRECISION_FP64));
  require(fp64.converged, "CPU FP64 RHF should converge");
  require(fp64.precision.requested_mode == VIBEQC_PRECISION_FP64,
          "explicit FP64 reports FP64 as the requested mode");
  require(fp64.precision.effective_bits == 64, "CPU FP64 runs at 64 bits");
  require(!fp64.precision.strict_refinement_applied, "CPU FP64 applies no mixed refinement");
  require(fp64.precision.refinement_iterations == 0 &&
              fp64.precision.mixed_precision_reserved_error == 0.0,
          "CPU FP64 reports no mixed budget and no refinement iterations");

  const vibeqc::scf::ScfResult auto_result =
      run_cpu_rhf(std::optional<vibeqc_precision_mode>(VIBEQC_PRECISION_AUTO));
  require(auto_result.converged, "CPU auto RHF should converge");
  require(auto_result.precision.requested_mode == VIBEQC_PRECISION_AUTO,
          "CPU auto reports the requested auto policy (not the collapsed FP64)");
  require(auto_result.precision.effective_bits == 64,
          "CPU auto still runs the FP64 Fock (mixed route is CUDA-only)");

  // The two policies are computationally identical on CPU, so the energies match
  // and only the requested-mode provenance differs.
  require_close(auto_result.energy, fp64.energy, 1.0e-12, "CPU auto and FP64 give the same energy");
  require(auto_result.precision.requested_mode != fp64.precision.requested_mode,
          "provenance distinguishes the explicit fp64 from an auto request");

  // A nullopt mode keeps the struct default (fp64) on the host path.
  const vibeqc::scf::ScfResult nullopt_result = run_cpu_rhf(std::nullopt);
  require(nullopt_result.converged, "CPU nullopt RHF should converge");
  require(nullopt_result.precision.requested_mode == VIBEQC_PRECISION_FP64,
          "absent precision mode keeps the FP64 requested-mode default");
}

}  // namespace

int main() {
  try {
    verify_auto_budget_admission();
    verify_per_item_budget();
    verify_fp64_strict();
    verify_auto_with_legacy_override();
    verify_nullopt_legacy_parity();
    verify_converged_fock_reuse_rms();
    verify_cpu_provenance();
    std::cout << "validated precision policy controller and CPU provenance\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
