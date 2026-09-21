#include <cuda_runtime_api.h>

#include "runtime/context.hpp"
#include "runtime/cuda_target_info.hpp"
#include "scf/aot_shell_registry.hpp"

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
  const CudaTargetInfo target = cuda_target_info_from_properties(properties);
  state.device_name = properties.name;
  state.compute_capability_major = target.compute_capability_major;
  state.compute_capability_minor = target.compute_capability_minor;
  state.warp_size = static_cast<int>(target.warp_size);
  state.maximum_threads_per_sm = static_cast<int>(target.maximum_threads_per_sm);
  state.maximum_blocks_per_sm = static_cast<int>(target.maximum_blocks_per_sm);
  state.registers_per_sm = static_cast<int>(target.registers_per_sm);
  // CUDA exposes register-file limits per block/SM in cudaDeviceProp. The
  // architectural per-thread allocation ceiling is 255 for supported SMs.
  state.maximum_registers_per_thread = 255;
  state.shared_memory_per_block = target.shared_memory_per_block;
  state.shared_memory_per_sm = target.shared_memory_per_sm;
  state.multiprocessor_count = static_cast<int>(target.multiprocessor_count);

  // Resolve the generated kernel set once for this context/device. Unknown
  // devices retain the generic implementation instead of borrowing a tuned
  // schedule compiled and measured for another compute capability.
  scf::generated::select_profile_for_device(state.device_id, properties.major, properties.minor);
  const scf::generated::ProfileInfo& profile = scf::generated::selected_profile();
  state.aot_profile_name = profile.name;
  state.aot_profile_tuned = profile.tuned;
  state.aot_profile_portable = profile.portable;
  state.aot_profile_compatible = profile.compatible;

  // Context creation owns all CUDA-provider validation. RHF execution later
  // reports CUDA only after the device-resident scientific path is selected;
  // the CPU implementation remains a separately requested oracle backend.
  state.executed_backend = VIBEQC_BACKEND_CUDA;
  return VIBEQC_STATUS_SUCCESS;
}

}  // namespace vibeqc::runtime
