#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>

namespace vibeqc_tensor {

// Shared host error boundary for generated executors and reference export.
// Keep the typed allocation failure identical across these call sites.
// Preserve the allocator's typed failure across the generated C ABI. Driver,
// arithmetic and cuBLAS errors must never be guessed to mean device OOM.
struct DeviceAllocationError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline void cuda_check(cudaError_t status) {
  if (status == cudaErrorMemoryAllocation) throw DeviceAllocationError(cudaGetErrorString(status));
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}  // namespace vibeqc_tensor
