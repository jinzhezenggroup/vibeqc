#include "api/error.hpp"
#include "build_identity.hpp"
#include "vibeqc/vibeqc.h"

#if VIBEQC_HAS_CUDA
#include <cuda_runtime_api.h>

#include <cstdio>

#include "runtime/cuda_target_info.hpp"
#include "scf/aot_shell_registry.hpp"
#endif

extern "C" {

const char* vibeqc_get_source_identity(void) { return kVibeqcSourceIdentity; }

vibeqc_status vibeqc_cuda_tuning_device(int32_t device_id,
                                        vibeqc_cuda_tuning_device_descriptor* output) {
  if (!vibeqc::api::valid_descriptor(output)) {
    return output == nullptr ? VIBEQC_STATUS_INVALID_ARGUMENT : VIBEQC_STATUS_ABI_MISMATCH;
  }
#if VIBEQC_HAS_CUDA
  int previous = 0;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&previous) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device_id) != cudaSuccess ||
      cudaSetDevice(device_id) != cudaSuccess)
    return VIBEQC_STATUS_CUDA_ERROR;
  const vibeqc::runtime::CudaTargetInfo target =
      vibeqc::runtime::cuda_target_info_from_properties(properties);
  // Registry selection is device-scoped. Restore the caller's current device,
  // including after a failed probe, before exposing any usable identity.
  const cudaError_t runtime = cudaRuntimeGetVersion(&output->runtime_version);
  const cudaError_t driver = cudaDriverGetVersion(&output->driver_version);
  vibeqc::scf::generated::select_profile_for_device(device_id, properties.major, properties.minor);
  const auto& profile = vibeqc::scf::generated::selected_profile();
  std::snprintf(output->official_profile, sizeof(output->official_profile), "%s", profile.name);
  output->portable = profile.portable ? 1 : 0;
  const cudaError_t restored = cudaSetDevice(previous);
  if (runtime != cudaSuccess || driver != cudaSuccess || restored != cudaSuccess)
    return VIBEQC_STATUS_CUDA_ERROR;
  std::snprintf(output->name, sizeof(output->name), "%s", properties.name);
  output->major = target.compute_capability_major;
  output->minor = target.compute_capability_minor;
  output->warp_size = static_cast<int32_t>(target.warp_size);
  output->maximum_threads_per_block = static_cast<int32_t>(target.maximum_threads_per_block);
  output->maximum_threads_per_sm = static_cast<int32_t>(target.maximum_threads_per_sm);
  output->maximum_blocks_per_sm = static_cast<int32_t>(target.maximum_blocks_per_sm);
  output->registers_per_sm = static_cast<int32_t>(target.registers_per_sm);
  output->maximum_registers_per_thread = 255;
  output->sm_count = static_cast<int32_t>(target.multiprocessor_count);
  output->shared_memory_per_block = target.shared_memory_per_block;
  output->shared_memory_per_block_optin = target.shared_memory_per_block_optin;
  output->shared_memory_per_sm = target.shared_memory_per_sm;
#ifdef CUDART_VERSION
  output->toolkit_version = CUDART_VERSION;
#else
  // CUDA-compatible providers may not publish NVIDIA's compile-time version
  // macro. Preserve a usable provenance value from the runtime they expose.
  output->toolkit_version = output->runtime_version;
#endif
  output->release_build = VIBEQC_TUNING_RELEASE_BUILD;
  output->fast_compile = VIBEQC_CUDA_FAST_COMPILE;
  return VIBEQC_STATUS_SUCCESS;
#else
  (void)device_id;
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

}  // extern "C"
