#pragma once

#include <cuda_runtime_api.h>

#include <limits>
#include <new>
#include <string>

#include "runtime/resource_cuda.cuh"
#include "vibeqc/vibeqc.h"

namespace vibeqc::scf::cuda_execution {

/** Preserve CUDA allocation versus execution failure classification. */
inline vibeqc_status source_cuda_status(cudaError_t status) {
  if (status == cudaSuccess) return VIBEQC_STATUS_SUCCESS;
  return status == cudaErrorMemoryAllocation ? VIBEQC_STATUS_OUT_OF_MEMORY
                                             : VIBEQC_STATUS_CUDA_ERROR;
}

/** Upload immutable metadata into a source-owned allocation registry.
 * Source supplies device_bytes and allocations. Register each allocation before
 * copying so failure cleanup remains owned even if host bookkeeping throws.
 * Direct and DF sources retain the established diagnostic strings and status
 * mapping; this helper contains no provider equations or launch policy.
 */
template <class Source>
vibeqc_status source_upload(Source& source, const void* host, std::size_t bytes, void** device,
                            std::string& detail) {
  if (bytes == 0U) {
    *device = nullptr;
    return VIBEQC_STATUS_SUCCESS;
  }
  if (source.device_bytes > std::numeric_limits<std::size_t>::max() - bytes) {
    detail = "bounded DF source metadata bytes overflow size_t";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  cudaError_t error = runtime::resource_cuda_malloc(device, bytes);
  if (error != cudaSuccess) {
    detail = "CUDA allocation failed for bounded DF source metadata";
    return source_cuda_status(error);
  }
  try {
    source.allocations.push_back(*device);
  } catch (const std::bad_alloc&) {
    (void)runtime::resource_cuda_free(*device);
    *device = nullptr;
    detail = "host allocation failed for bounded DF source metadata handles";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  source.device_bytes += bytes;
  error = cudaMemcpy(*device, host, bytes, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    detail = "CUDA upload failed for bounded DF source metadata";
    return source_cuda_status(error);
  }
  return VIBEQC_STATUS_SUCCESS;
}

}  // namespace vibeqc::scf::cuda_execution
