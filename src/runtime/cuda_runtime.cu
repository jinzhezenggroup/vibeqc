#include "runtime/context.hpp"

#include <cuda_runtime_api.h>

namespace vibeqc::runtime {

vibeqc_status initialize_cuda_context(core::ContextState& state, std::string& detail) {
  int count = 0;
  cudaError_t error = cudaGetDeviceCount(&count);
  if (error != cudaSuccess) {
    detail = cudaGetErrorString(error);
    return VIBEQC_STATUS_CUDA_ERROR;
  }
  if (state.device_id < 0 || state.device_id >= count) {
    detail = "CUDA device id is out of range";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  error = cudaSetDevice(state.device_id);
  if (error != cudaSuccess) {
    detail = cudaGetErrorString(error);
    return VIBEQC_STATUS_CUDA_ERROR;
  }
  cudaDeviceProp properties{};
  error = cudaGetDeviceProperties(&properties, state.device_id);
  if (error != cudaSuccess) {
    detail = cudaGetErrorString(error);
    return VIBEQC_STATUS_CUDA_ERROR;
  }
  // Force runtime initialization now so context creation, not the first
  // scientific call, owns and reports any driver/runtime incompatibility.
  error = cudaFree(nullptr);
  if (error != cudaSuccess) {
    detail = cudaGetErrorString(error);
    return VIBEQC_STATUS_CUDA_ERROR;
  }
  state.device_name = properties.name;

  // Context creation owns all CUDA-provider validation. RHF execution later
  // reports CUDA only after the device-resident scientific path is selected;
  // the CPU implementation remains a separately requested oracle backend.
  state.executed_backend = VIBEQC_BACKEND_CUDA;
  return VIBEQC_STATUS_SUCCESS;
}

}  // namespace vibeqc::runtime
