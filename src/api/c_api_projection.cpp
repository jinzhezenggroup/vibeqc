#include <span>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "integrals/s_integrals.hpp"
#include "vibeqc/vibeqc.h"

extern "C" vibeqc_status vibeqc_system_cross_overlap_cpu(vibeqc_context* context,
                                                         const vibeqc_system* target,
                                                         const vibeqc_system* source,
                                                         double* output, size_t output_count) {
  if (context == nullptr || target == nullptr || source == nullptr || output == nullptr) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  try {
    vibeqc::integrals::cross_overlap(target->data, source->data, {output, output_count});
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&context->last_detail);
  }
}
