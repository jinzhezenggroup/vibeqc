#include "runtime/context.hpp"

namespace qce::runtime {

qce_status initialize_context(core::ContextState& state, std::string& detail) {
  if (state.requested_backend == QCE_BACKEND_CPU_REFERENCE) {
    state.executed_backend = QCE_BACKEND_CPU_REFERENCE;
    state.device_name = "CPU reference backend";
    return QCE_STATUS_SUCCESS;
  }
  if (state.requested_backend != QCE_BACKEND_CUDA) {
    detail = "unknown execution backend";
    return QCE_STATUS_INVALID_ARGUMENT;
  }
#if QCE_HAS_CUDA
  return initialize_cuda_context(state, detail);
#else
  detail = "the library was built without CUDA support";
  return QCE_STATUS_NOT_IMPLEMENTED;
#endif
}

}  // namespace qce::runtime
