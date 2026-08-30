#include "scf/cuda/rhf_policy.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace vibeqc::scf::cuda_policy {
namespace {

constexpr double kDefaultMixedPrecisionFockThreshold = 1.0e-6;
constexpr double kTightConvergedFockReuseDensityRms = 1.0e-12;
constexpr double kExpandedConvergedFockReuseDensityTolerance = 1.0e-9;
constexpr double kExpandedConvergedFockReuseDensityRms = 2.0e-9;

bool enabled(const char* variable) noexcept {
  const char* selection = std::getenv(variable);
  return selection == nullptr ||
      (std::strcmp(selection, "0") != 0 &&
       std::strcmp(selection, "none") != 0);
}

bool selected(const char* variable, const char* value) noexcept {
  const char* selection = std::getenv(variable);
  return selection != nullptr &&
      (std::strcmp(selection, "1") == 0 || std::strcmp(selection, value) == 0);
}

}  // namespace

bool reuse_converged_fock_requested() noexcept {
  const char* force_rebuild = std::getenv("VIBEQC_FINAL_FOCK_REBUILD");
  return force_rebuild == nullptr || std::strcmp(force_rebuild, "0") == 0 ||
      std::strcmp(force_rebuild, "none") == 0;
}

std::optional<double> configured_mixed_precision_fock_threshold(
    double screening_tolerance) noexcept {
  const char* selection =
      std::getenv("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD");
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

bool resident_ppps_bra_requested() noexcept {
  return enabled("VIBEQC_PPPS_RESIDENT_BRA");
}

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

bool one_electron_force_scalar_requested() noexcept {
  const char* selection = std::getenv("VIBEQC_ONE_ELECTRON_FORCE_SCALAR");
  return selection == nullptr || std::strcmp(selection, "0") == 0;
}

bool resident_psss_bra_requested() noexcept {
  return enabled("VIBEQC_PSSS_RESIDENT_BRA");
}

}  // namespace vibeqc::scf::cuda_policy
