#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <cstddef>

#include "vibeqc/vibeqc.h"

namespace vibeqc::scf::cuda_execution {

/** Preserve the public CUDA/BLAS/solver status mapping at host launch boundaries. */
vibeqc_status cuda_status(cudaError_t status);

vibeqc_status solver_status(cusolverStatus_t status);

vibeqc_status blas_status(cublasStatus_t status);

/** Queue a nonempty host upload on the caller's stream; empty input is a no-op. */
vibeqc_status copy_to_device(void* destination, const void* source, std::size_t bytes,
                             cudaStream_t stream);

}  // namespace vibeqc::scf::cuda_execution
