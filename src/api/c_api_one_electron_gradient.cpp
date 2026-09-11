#include <algorithm>
#include <limits>
#include <span>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "molecule/basis.hpp"
#include "scf/cuda_one_electron_gradient.hpp"
#include "vibeqc/vibeqc.h"

extern "C" vibeqc_status vibeqc_system_one_electron_gradient_cuda(
    vibeqc_context* context, const vibeqc_system* system, const double* overlap_weights,
    const double* kinetic_weights, const double* attraction_weights, size_t matrix_count,
    unsigned schedule, size_t maximum_bytes, double* gradient, size_t gradient_count,
    vibeqc_one_electron_gradient_resources* resources) {
  if (!context || !system || !gradient || schedule > 2 || !maximum_bytes)
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  if (resources && !vibeqc::api::valid_descriptor(resources)) return VIBEQC_STATUS_ABI_MISMATCH;
  const auto n = vibeqc::molecule::ao_count(system->data);
  if (!n || n > std::numeric_limits<std::size_t>::max() / n || matrix_count != n * n ||
      gradient_count != system->data.atoms.size() * 3)
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  if (resources) {
    *resources = {sizeof(*resources), VIBEQC_ABI_VERSION, 0, 0, 0, 0, 0, 0};
  }
  context->last_detail.clear();
#if VIBEQC_HAS_CUDA
  if (context->state.executed_backend != VIBEQC_BACKEND_CUDA) {
    context->last_detail = "generic generated gradients require a CUDA context";
    return VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  try {
    auto weights = [&](const double* p) {
      return p ? std::span<const double>(p, matrix_count) : std::span<const double>();
    };
    std::vector<double> result;
    vibeqc::scf::OneElectronGradientResources measured;
    auto status = vibeqc::scf::execute_cuda_one_electron_gradient(
        context->state.device_id, system->data, weights(overlap_weights), weights(kinetic_weights),
        weights(attraction_weights), schedule, maximum_bytes, result, context->last_detail,
        &measured);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    std::copy(result.begin(), result.end(), gradient);
    if (resources) {
      resources->device_bytes = measured.device_bytes;
      resources->host_numeric_bytes = measured.host_numeric_bytes;
      resources->host_to_device_bytes = measured.host_to_device_bytes;
      resources->device_to_host_bytes = measured.device_to_host_bytes;
      resources->synchronous_uploads = measured.synchronous_uploads;
      resources->stream_synchronizations = measured.stream_synchronizations;
    }
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&context->last_detail);
  }
#else
  (void)overlap_weights;
  (void)kinetic_weights;
  (void)attraction_weights;
  context->last_detail = "CUDA one-electron gradients are unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}
