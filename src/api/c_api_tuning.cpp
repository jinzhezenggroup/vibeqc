#include "api/error.hpp"
#include "build_identity.hpp"
#include "vibeqc/vibeqc.h"

#if VIBEQC_HAS_CUDA
#include <cuda_runtime_api.h>

#include <cstdio>

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
  output->major = properties.major;
  output->minor = properties.minor;
  output->warp_size = properties.warpSize;
  output->maximum_threads_per_block = properties.maxThreadsPerBlock;
  output->maximum_threads_per_sm = properties.maxThreadsPerMultiProcessor;
  output->maximum_blocks_per_sm = properties.maxBlocksPerMultiProcessor;
  output->registers_per_sm = properties.regsPerMultiprocessor;
  output->maximum_registers_per_thread = 255;
  output->sm_count = properties.multiProcessorCount;
  output->shared_memory_per_block = properties.sharedMemPerBlock;
  output->shared_memory_per_block_optin = properties.sharedMemPerBlockOptin;
  output->shared_memory_per_sm = properties.sharedMemPerMultiprocessor;
  output->toolkit_version = CUDART_VERSION;
  output->release_build = VIBEQC_TUNING_RELEASE_BUILD;
  output->fast_compile = VIBEQC_CUDA_FAST_COMPILE;
  return VIBEQC_STATUS_SUCCESS;
#else
  (void)device_id;
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

}  // extern "C"
