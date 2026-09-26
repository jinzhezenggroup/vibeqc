#include <climits>
#include <cstdint>
#include <vector>

#include "api/error.hpp"
#include "dft/dispersion/gcp_r2scan3c.hpp"
#include "vibeqc/vibeqc.h"

namespace {

vibeqc_status public_status(vibeqc::dft::dispersion::GCPStatus status) {
  using vibeqc::dft::dispersion::GCPStatus;
  switch (status) {
    case GCPStatus::success:
      return VIBEQC_STATUS_SUCCESS;
    case GCPStatus::invalid_argument:
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    case GCPStatus::unsupported_element:
      return VIBEQC_STATUS_NOT_IMPLEMENTED;
    case GCPStatus::coincident_atoms:
    case GCPStatus::numerical_failure:
      return VIBEQC_STATUS_NUMERICAL_FAILURE;
  }
  return VIBEQC_STATUS_INTERNAL_ERROR;
}

}  // namespace

extern "C" {

const char* vibeqc_r2scan3c_gcp_provider_identity(void) { return "r2scan3c-gcp-cpu-v1"; }

vibeqc_status vibeqc_r2scan3c_gcp_evaluate(const int32_t* atomic_numbers, uint32_t atom_count,
                                           const double* coordinates, uint32_t coordinate_count,
                                           double* energy, double* gradient,
                                           uint32_t gradient_count) {
  // The native kernel indexes xyz/gradient with signed int expressions (3 * n).
  if (!energy || !atomic_numbers || !coordinates || atom_count == 0 ||
      atom_count > static_cast<uint32_t>(INT_MAX / 3) || coordinate_count != 3u * atom_count ||
      ((!gradient && gradient_count != 0) || (gradient && gradient_count != coordinate_count))) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  try {
    std::vector<double> scratch;
    double* target = gradient;
    if (!target) {
      scratch.resize(coordinate_count);
      target = scratch.data();
    }
    return public_status(vibeqc::dft::dispersion::evaluate_r2scan3c_gcp(
        static_cast<int>(atom_count), atomic_numbers, coordinates,
        vibeqc::dft::dispersion::r2scan3c_gcp_parameters(), energy, target));
  } catch (...) {
    return vibeqc::api::map_exception();
  }
}

}  // extern "C"
