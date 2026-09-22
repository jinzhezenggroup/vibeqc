#include "scf/cuda/rhf_policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "generated_one_electron_derivative_policy.cuh"

namespace vibeqc::scf::cuda_policy {
namespace {

constexpr std::size_t kUnknownTargetFixedTopologyArenaBytes = std::size_t{256} << 20;
constexpr std::size_t kFixedTopologyArenaDeviceMemoryDivisor = 32;
constexpr std::size_t kUnknownTargetBoundedStreamingArenaBytes = std::size_t{256} << 20;
constexpr std::size_t kBoundedStreamingArenaDeviceMemoryDivisor = 16;
constexpr unsigned kUnknownTargetPersistentQuartetWarpsPerSm = 4;

constexpr double kDefaultMixedPrecisionFockThreshold = 1.0e-6;
/**
 * Ceiling for the resolved tile cutoff, anchored so the default 1.0e-10 target
 * can recover the legacy measured-accurate 1.0e-6 threshold. It is only an
 * upper bound: the accumulated-error budget below resolves the value used.
 */
constexpr double kAutoPrecisionFactor = 1.0e4;
/**
 * Fraction of the requested energy tolerance reserved for the accumulated FP32
 * rounding of the iterative mixed Fock. The target-precision refinement makes
 * the operator that produces the reported energy, orbitals, and forces exact,
 * so this bound covers only the perturbation the iterative density was converged
 * against.
 */
constexpr double kAutoMixedPrecisionErrorBudgetFraction = 6.25e-02;
constexpr double kTightConvergedFockReuseDensityRms = 1.0e-12;
constexpr double kExpandedConvergedFockReuseDensityTolerance = 1.0e-9;
constexpr double kExpandedConvergedFockReuseDensityRms = 2.0e-9;
using NucleusCooperativeSchedule =
    generated_one_electron_derivative_policy::NucleusCooperativeSchedule;

bool enabled(const char* variable) noexcept {
  const char* selection = std::getenv(variable);
  return selection == nullptr ||
         (std::strcmp(selection, "0") != 0 && std::strcmp(selection, "none") != 0);
}

bool selected(const char* variable, const char* value) noexcept {
  const char* selection = std::getenv(variable);
  return selection != nullptr &&
         (std::strcmp(selection, "1") == 0 || std::strcmp(selection, value) == 0);
}

std::size_t resolve_target_memory_budget(std::size_t total_global_memory,
                                         std::size_t profile_ceiling, std::size_t unknown_fallback,
                                         std::size_t divisor) noexcept {
  if (total_global_memory == 0) return std::min(profile_ceiling, unknown_fallback);
  return std::min(profile_ceiling, total_global_memory / divisor);
}

std::optional<double> parsed_mixed_precision_override(double screening_tolerance) noexcept {
  // A user-supplied explicit numeric threshold from the legacy switch; the
  // absent / 0 / none / auto / invalid spellings yield std::nullopt.
  const char* selection = std::getenv("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD");
  if (selection == nullptr) return std::nullopt;
  if (std::strcmp(selection, "0") == 0 || std::strcmp(selection, "none") == 0 ||
      std::strcmp(selection, "auto") == 0) {
    return std::nullopt;
  }
  char* end = nullptr;
  const double value = std::strtod(selection, &end);
  if (end == selection || end == nullptr || *end != '\0') return std::nullopt;
  if (!std::isfinite(value) || value <= screening_tolerance) return std::nullopt;
  return value;
}
}  // namespace

DirectJkSchedulePolicy resolve_direct_jk_schedule_policy(const runtime::CudaTargetInfo& target,
                                                         DirectJkTuningProfile profile) noexcept {
  DirectJkSchedulePolicy policy;
  policy.cuda_stack_limit_bytes = profile.cuda_stack_limit_bytes;

  // Fixed-topology descriptors are long-lived prepared-state storage. Keep
  // their historical one-GiB profile ceiling under the conservative 1/32
  // target-memory budget.
  policy.fixed_topology.arena_maximum_bytes = resolve_target_memory_budget(
      target.total_global_memory, profile.fixed_topology.maximum_arena_bytes,
      kUnknownTargetFixedTopologyArenaBytes, kFixedTopologyArenaDeviceMemoryDivisor);

  // Bounded streaming is a reusable page scratch arena, not fixed-topology
  // resident storage. Its independently qualified 8M-task page is 1.5 GiB at
  // the current GeneratedShellTask ABI, so it gets a separate resource budget.
  // Never derive this capacity from the fixed-topology arena: doing so shrinks
  // the 768-AO page count and changes endpoint scheduling.
  policy.bounded_streaming.task_capacity_ceiling = profile.bounded_streaming.maximum_task_capacity;
  policy.bounded_streaming.arena_maximum_bytes = resolve_target_memory_budget(
      target.total_global_memory, profile.bounded_streaming.maximum_arena_bytes,
      kUnknownTargetBoundedStreamingArenaBytes, kBoundedStreamingArenaDeviceMemoryDivisor);

  // Persistent workers are one warp per block. Respect both the resident block
  // ceiling and the SM thread ceiling; unknown facts choose a smaller
  // conservative fallback instead of assuming the measured 5090 occupancy.
  const bool complete_occupancy = target.warp_size != 0 && target.maximum_threads_per_sm != 0 &&
                                  target.maximum_blocks_per_sm != 0;
  unsigned legal_warps = kUnknownTargetPersistentQuartetWarpsPerSm;
  if (target.warp_size != 0 && target.maximum_threads_per_sm != 0) {
    const unsigned thread_ceiling = std::max(1U, target.maximum_threads_per_sm / target.warp_size);
    legal_warps = complete_occupancy ? thread_ceiling : std::min(legal_warps, thread_ceiling);
  }
  if (target.maximum_blocks_per_sm != 0) {
    legal_warps = std::min(legal_warps, target.maximum_blocks_per_sm);
  }
  policy.persistent_quartet_warps_per_sm =
      std::max(1U, std::min(profile.maximum_persistent_quartet_warps_per_sm, legal_warps));
  return policy;
}

std::size_t direct_jk_bounded_streaming_task_capacity_limit(
    const DirectJkSchedulePolicy& policy, std::size_t generated_task_bytes) noexcept {
  if (generated_task_bytes == 0) return 0;
  return std::min(policy.bounded_streaming.task_capacity_ceiling,
                  policy.bounded_streaming.arena_maximum_bytes / generated_task_bytes);
}

bool reuse_converged_fock_requested() noexcept {
  const char* force_rebuild = std::getenv("VIBEQC_FINAL_FOCK_REBUILD");
  return force_rebuild == nullptr || std::strcmp(force_rebuild, "0") == 0 ||
         std::strcmp(force_rebuild, "none") == 0;
}

std::optional<double> configured_mixed_precision_fock_threshold(
    double screening_tolerance) noexcept {
  const char* selection = std::getenv("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD");
  if (selection == nullptr || std::strcmp(selection, "0") == 0 ||
      std::strcmp(selection, "none") == 0) {
    return std::nullopt;
  }
  double threshold = 0.0;
  if (std::strcmp(selection, "auto") == 0) {
    threshold = kDefaultMixedPrecisionFockThreshold;
  } else {
    char* end = nullptr;
    threshold = std::strtod(selection, &end);
    if (end == selection || end == nullptr || *end != '\0') {
      return std::nullopt;
    }
  }
  if (!std::isfinite(threshold) || threshold <= screening_tolerance) {
    return std::nullopt;
  }
  return threshold;
}

AutoMixedPrecisionAdmission admit_auto_mixed_precision_fock(double energy_tolerance,
                                                            double screening_tolerance,
                                                            double eligible_tiles) noexcept {
  AutoMixedPrecisionAdmission admission;
  if (!(energy_tolerance > 0.0) || !std::isfinite(energy_tolerance)) return admission;
  // An unknown (or empty) census cannot be budgeted; refusing keeps the FP64
  // operator rather than admitting work whose accumulated error is unbounded.
  if (!(eligible_tiles >= 1.0) || !std::isfinite(eligible_tiles)) return admission;
  // Worst case: every eligible tile is admitted at the cutoff and carries the
  // full relative FP32 rounding, so the accumulated Fock error is bounded by
  // eps32 * cutoff * eligible_tiles. Solve that bound for the largest cutoff the
  // reserved budget certifies, and never exceed the legacy measured anchor.
  const double reserved_error = kAutoMixedPrecisionErrorBudgetFraction * energy_tolerance;
  const double budget_threshold =
      reserved_error / (kMixedPrecisionFloat32UnitRoundoff * eligible_tiles);
  const double anchor_threshold = kAutoPrecisionFactor * energy_tolerance;
  const double threshold = std::min(anchor_threshold, budget_threshold);
  if (!(threshold > screening_tolerance)) {
    // The budget certifies no useful tile at this requested accuracy, so the
    // operator stays FP64 instead of accumulating rounding it cannot bound.
    return admission;
  }
  admission.admitted = true;
  admission.threshold = threshold;
  admission.reserved_error = reserved_error;
  admission.eligible_tiles = eligible_tiles;
  return admission;
}
MixedPrecisionFockPolicy resolve_mixed_precision_fock_policy(
    std::optional<vibeqc_precision_mode> precision_mode, double energy_tolerance,
    double screening_tolerance, double eligible_tiles) noexcept {
  MixedPrecisionFockPolicy policy;
  if (!precision_mode.has_value()) {
    // No explicit public policy: preserve the legacy diagnostic switch exactly.
    // The diagnostic cutoff is item agnostic, so every item shares the ceiling.
    policy.threshold = configured_mixed_precision_fock_threshold(screening_tolerance);
    policy.item_cutoff_ceiling = policy.threshold.value_or(0.0);
    return policy;
  }
  switch (*precision_mode) {
    case VIBEQC_PRECISION_FP64:
      return policy;
    case VIBEQC_PRECISION_AUTO: {
      const std::optional<double> override_value =
          parsed_mixed_precision_override(screening_tolerance);
      if (override_value.has_value()) {
        // An explicit diagnostic cutoff is deliberately unbudgeted, so it is
        // reported as uncertified rather than as budgeted evidence, and it
        // applies to every item without a census.
        policy.threshold = override_value;
        policy.item_cutoff_ceiling = *override_value;
        return policy;
      }
      const AutoMixedPrecisionAdmission admission =
          admit_auto_mixed_precision_fock(energy_tolerance, screening_tolerance, eligible_tiles);
      if (admission.admitted) {
        policy.threshold = admission.threshold;
        policy.budget_certified = true;
        policy.reserved_error = admission.reserved_error;
        policy.eligible_tiles = admission.eligible_tiles;
        // The certified batch ceiling keeps the tolerance anchor; each item
        // tightens it with its own census below.
        policy.item_cutoff_ceiling = kAutoPrecisionFactor * energy_tolerance;
        policy.item_budget_error = admission.reserved_error;
      }
      return policy;
    }
    default:
      // An unrecognized public policy must never relax the default.
      return policy;
  }
}

MixedPrecisionItemPolicy resolve_mixed_precision_item(const MixedPrecisionFockPolicy& policy,
                                                      bool validated_warm_state,
                                                      std::size_t item_tile_census,
                                                      double screening_tolerance) noexcept {
  MixedPrecisionItemPolicy item;
  if (!policy.threshold.has_value()) return item;
  if (!(policy.item_budget_error > 0.0)) {
    // An explicit diagnostic cutoff is not a budget: keep it item agnostic.
    item.admitted = true;
    item.threshold = policy.item_cutoff_ceiling;
    item.census = 1U;
    return item;
  }
  // The accumulated bound needs the item's own census, and the reserved budget
  // only bounds the perturbation of a known state, so a cold item (or one whose
  // census is unavailable) keeps the exact operator for its whole solve.
  if (!validated_warm_state || item_tile_census == 0) return item;
  const double budget_threshold =
      policy.item_budget_error /
      (kMixedPrecisionFloat32UnitRoundoff * static_cast<double>(item_tile_census));
  const double threshold = std::min(policy.item_cutoff_ceiling, budget_threshold);
  if (!(threshold > screening_tolerance)) return item;
  item.admitted = true;
  item.threshold = threshold;
  item.census = static_cast<std::uint32_t>(item_tile_census);
  return item;
}

bool graph_native_eigensolver_override_requested() noexcept {
  const char* selection = std::getenv("VIBEQC_GRAPH_EIGENSOLVER_OVERRIDE");
  return selection != nullptr && std::strcmp(selection, "graph_native") == 0;
}

bool xsyev_probe_skip_diagnostic_requested() noexcept {
  return selected("VIBEQC_XSYEV_PROBE_SKIP_DIAGNOSTIC", "skip");
}

bool bounded_direct_streaming_override_requested() noexcept {
  return selected("VIBEQC_BOUNDED_DIRECT_STREAMING", "force");
}

bool bounded_direct_count_diagnostic_requested() noexcept {
  return selected("VIBEQC_BOUNDED_DIRECT_COUNT_DIAGNOSTIC", "count");
}

bool bounded_direct_aot_only_diagnostic_requested() noexcept {
  return selected("VIBEQC_BOUNDED_DIRECT_AOT_ONLY_DIAGNOSTIC", "aot");
}

bool bounded_direct_fock_only_diagnostic_requested() noexcept {
  return selected("VIBEQC_BOUNDED_DIRECT_FOCK_ONLY_DIAGNOSTIC", "fock");
}

bool bounded_fock_class_timing_requested() noexcept {
  return selected("VIBEQC_BOUNDED_DIRECT_FOCK_CLASS_PROFILE", "profile");
}

bool aot_shell_class_selection_override_requested() noexcept {
  const char* selection = std::getenv("VIBEQC_AOT_SHELL_CLASSES");
  // Match the generated registry: absent, empty, and "all" all mean the full
  // compiled profile. Any other spelling intentionally narrows the force
  // registry and may therefore exercise the generic fallback for diagnostics.
  return selection != nullptr && *selection != '\0' && std::strcmp(selection, "all") != 0;
}

bool direct_tile_validation_requested() noexcept {
  return selected("VIBEQC_DIRECT_TILE_VALIDATION", "validate");
}

double converged_fock_reuse_density_rms(double density_tolerance) noexcept {
  return density_tolerance >= kExpandedConvergedFockReuseDensityTolerance
             ? kExpandedConvergedFockReuseDensityRms
             : kTightConvergedFockReuseDensityRms;
}

bool force_density_product_screening_requested() noexcept {
  return enabled("VIBEQC_FORCE_DENSITY_PRODUCT_SCREENING");
}

bool resident_ppps_bra_requested() noexcept { return enabled("VIBEQC_PPPS_RESIDENT_BRA"); }

bool ppps_signature_bucketing_requested() noexcept {
  return enabled("VIBEQC_PPPS_SIGNATURE_BUCKETING");
}

bool psps_signature_bucketing_requested() noexcept {
  return enabled("VIBEQC_PSPS_SIGNATURE_BUCKETING");
}

bool ppss_signature_bucketing_requested() noexcept {
  return enabled("VIBEQC_PPSS_SIGNATURE_BUCKETING");
}

unsigned ppps_resident_block_threads_requested() noexcept {
  const char* selection = std::getenv("VIBEQC_PPPS_BLOCK_THREADS");
  if (selection == nullptr || std::strcmp(selection, "256") == 0) return 256U;
  if (std::strcmp(selection, "128") == 0) return 128U;
  if (std::strcmp(selection, "64") == 0) return 64U;
  if (std::strcmp(selection, "32") == 0) return 32U;
  return 0U;
}

OneElectronValuePolicy resolve_one_electron_value_policy(
    const runtime::CudaProviderCapabilities& provider) noexcept {
  OneElectronValuePolicy policy;
  const char* selection = std::getenv("VIBEQC_ONE_ELECTRON_VALUE_MAPPING");
  if (selection == nullptr) {
    policy.mapping = provider.templated_shell_warp_one_electron ? 1U : 0U;
    return policy;
  }
  policy.diagnostic_override = true;
  const bool requests_shell_warp =
      std::strcmp(selection, "1") == 0 || std::strcmp(selection, "shell_warp") == 0;
  if (requests_shell_warp && !provider.templated_shell_warp_one_electron) {
    policy.capability_fallback = true;
    return policy;
  }
  policy.mapping = requests_shell_warp ? 1U : 0U;
  return policy;
}

unsigned one_electron_value_mapping_requested() noexcept {
  return resolve_one_electron_value_policy(runtime::active_cuda_provider()).mapping;
}

bool generated_one_electron_derivatives_requested() noexcept {
  const char* selection = std::getenv("VIBEQC_ONE_ELECTRON_DERIVATIVES");
  if (selection == nullptr || std::strcmp(selection, "1") == 0 ||
      std::strcmp(selection, "generated") == 0 || std::strcmp(selection, "auto") == 0)
    return true;
  // The retained native cooperative implementation is an explicit oracle/control,
  // never an automatic production fallback. Typos/unknown values stay on the
  // compiler-owned generated path rather than silently changing scientific owner.
  if (std::strcmp(selection, "0") == 0 || std::strcmp(selection, "reference") == 0 ||
      std::strcmp(selection, "native") == 0 || std::strcmp(selection, "tensor") == 0)
    return false;
  return true;
}

unsigned one_electron_derivative_mapping_requested() noexcept {
  const char* selection = std::getenv("VIBEQC_ONE_ELECTRON_DERIVATIVE_MAPPING");
  if (selection == nullptr) return NucleusCooperativeSchedule::schedule_code;
  if (std::strcmp(selection, "serial") == 0 || std::strcmp(selection, "2") == 0) return 2U;
  if (std::strcmp(selection, "nucleus_cooperative") == 0 ||
      std::strcmp(selection, "cooperative") == 0 || std::strcmp(selection, "3") == 0)
    return NucleusCooperativeSchedule::schedule_code;
  if (std::strcmp(selection, "shell_warp") == 0 || std::strcmp(selection, "1") == 0) return 1U;
  return 0U;
}

bool resident_psss_bra_requested() noexcept { return enabled("VIBEQC_PSSS_RESIDENT_BRA"); }

unsigned df_derivative_mapping_requested() noexcept {
  return selected("VIBEQC_DF_DERIVATIVE_MAPPING", "serial") ? 1U : 0U;
}

bool df_value_math_requested(unsigned& math) noexcept {
  math = 0;
  const char* value = std::getenv("VIBEQC_DF_VALUE_MATH");
  if (!value || selected("VIBEQC_DF_VALUE_MATH", "auto") ||
      selected("VIBEQC_DF_VALUE_MATH", "generic"))
    return true;
  if (selected("VIBEQC_DF_VALUE_MATH", "candidate")) {
    math = 3;
    return true;
  }
  if (selected("VIBEQC_DF_VALUE_MATH", "polynomial")) {
    math = 1;
    return true;
  }
  if (selected("VIBEQC_DF_VALUE_MATH", "rys")) {
    math = 2;
    return true;
  }
  return false;
}

bool df_value_raw_lanes_requested(unsigned& lanes) noexcept {
  lanes = 1;
  const char* value = std::getenv("VIBEQC_DF_VALUE_RAW_MAPPING");
  if (!value || selected("VIBEQC_DF_VALUE_RAW_MAPPING", "auto") ||
      selected("VIBEQC_DF_VALUE_RAW_MAPPING", "scalar"))
    return true;
  if (selected("VIBEQC_DF_VALUE_RAW_MAPPING", "candidate")) {
    lanes = kDfCandidateRawSchedule;
    return true;
  }
  if (selected("VIBEQC_DF_VALUE_RAW_MAPPING", "subgroup")) {
    lanes = 4;
    return true;
  }
  if (selected("VIBEQC_DF_VALUE_RAW_MAPPING", "warp")) {
    lanes = 32;
    return true;
  }
  return false;
}

unsigned df_value_mapping_requested() noexcept {
  // Primitive-oriented warps won the endpoint comparisons at both budgets
  // and batch sizes. Other mappings remain explicit diagnostic candidates.
  if (std::getenv("VIBEQC_DF_VALUE_MAPPING") == nullptr) return 2U;
  if (selected("VIBEQC_DF_VALUE_MAPPING", "component")) return 1U;
  if (selected("VIBEQC_DF_VALUE_MAPPING", "primitive")) return 2U;
  return 0U;
}

}  // namespace vibeqc::scf::cuda_policy
