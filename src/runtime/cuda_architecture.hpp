#pragma once

#include <cuda_runtime.h>

namespace vibeqc::runtime {

/** Read only the two attributes needed by a schedule/profile lookup.
 * Full cudaGetDeviceProperties queries may inspect expensive unrelated
 * attributes. Do not cache a device ordinal across visibility/context owners.
 * Failure leaves the output unqualified and preserves the CUDA error.
 */
inline cudaError_t cuda_architecture(int device, unsigned& architecture) noexcept {
  architecture = 0;
  int major = 0, minor = 0;
  auto error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  if (error == cudaSuccess)
    error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  if (error == cudaSuccess) architecture = 10 * major + minor;
  return error;
}

}  // namespace vibeqc::runtime
