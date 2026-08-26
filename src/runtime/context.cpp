#include "runtime/context.hpp"

namespace vibeqc::runtime {

vibeqc_status initialize_context(core::ContextState& state, std::string& detail) {
  if (state.requested_backend == VIBEQC_BACKEND_CPU_REFERENCE) {
    state.executed_backend = VIBEQC_BACKEND_CPU_REFERENCE;
    state.device_name = "CPU reference backend";
    return VIBEQC_STATUS_SUCCESS;
  }
  if (state.requested_backend != VIBEQC_BACKEND_CUDA) {
    detail = "unknown execution backend";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
#if VIBEQC_HAS_CUDA
  return initialize_cuda_context(state, detail);
#else
  detail = "the library was built without CUDA support";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

}  // namespace vibeqc::runtime
