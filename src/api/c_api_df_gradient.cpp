#include <algorithm>
#include <limits>
#include <span>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "molecule/basis.hpp"
#include "scf/cuda_df_gradient.hpp"
#include "vibeqc/vibeqc.h"

extern "C" vibeqc_status vibeqc_system_df_gradient_cuda(
    vibeqc_context* context, const vibeqc_system* orbital, const vibeqc_system* auxiliary,
    const double* bar_a, size_t count_a, const double* bar_m, size_t count_m, unsigned schedule,
    size_t maximum_bytes, size_t maximum_tile_elements, double* gradient, size_t gradient_count,
    vibeqc_df_gradient_resources* resources) {
  if (!context || !orbital || !auxiliary || !gradient || schedule > 1 || !maximum_bytes)
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  if (resources && !vibeqc::api::valid_descriptor(resources)) return VIBEQC_STATUS_ABI_MISMATCH;
  const auto n = vibeqc::molecule::ao_count(orbital->data),
             a = vibeqc::molecule::ao_count(auxiliary->data);
  const auto maximum = std::numeric_limits<std::size_t>::max() / sizeof(double);
  if (!n || !a || n > maximum / n || a > maximum / a || n * n > maximum / a ||
      count_a != n * n * a || count_m != a * a || gradient_count != 3 * orbital->data.atoms.size())
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  if (resources) *resources = {sizeof(*resources), VIBEQC_ABI_VERSION, 0, 0, 0, 0, 0, 0, 0, 0};
  context->last_detail.clear();
#if VIBEQC_HAS_CUDA
  if (context->state.executed_backend != VIBEQC_BACKEND_CUDA) {
    context->last_detail = "generic DF gradients require a CUDA context";
    return VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  try {
    std::vector<double> result;
    vibeqc::scf::DfGradientResources measured;
    auto span = [](const double* p, std::size_t n) {
      return p ? std::span<const double>(p, n) : std::span<const double>();
    };
    const auto status = vibeqc::scf::execute_cuda_df_gradient(
        context->state.device_id, orbital->data, auxiliary->data, span(bar_a, count_a),
        span(bar_m, count_m), schedule, maximum_bytes, maximum_tile_elements, result,
        context->last_detail, &measured);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    std::copy(result.begin(), result.end(), gradient);
    if (resources)
      *resources = {sizeof(*resources),
                    VIBEQC_ABI_VERSION,
                    measured.host_bytes,
                    measured.device_bytes,
                    measured.host_to_device_bytes,
                    measured.device_to_host_bytes,
                    measured.weight_tile_elements,
                    measured.tiles,
                    measured.uploads,
                    measured.stream_synchronizations};
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&context->last_detail);
  }
#else
  (void)bar_a;
  (void)bar_m;
  (void)maximum_tile_elements;
  context->last_detail = "CUDA DF gradients are unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}
