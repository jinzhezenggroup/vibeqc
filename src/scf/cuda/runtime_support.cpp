#include "scf/cuda/runtime_support.hpp"

namespace vibeqc::scf::cuda_execution {

vibeqc_status cuda_status(cudaError_t status) {
  if (status == cudaSuccess) return VIBEQC_STATUS_SUCCESS;
  return status == cudaErrorMemoryAllocation ? VIBEQC_STATUS_OUT_OF_MEMORY
                                             : VIBEQC_STATUS_CUDA_ERROR;
}

vibeqc_status solver_status(cusolverStatus_t status) {
  if (status == CUSOLVER_STATUS_SUCCESS) return VIBEQC_STATUS_SUCCESS;
  return status == CUSOLVER_STATUS_ALLOC_FAILED ? VIBEQC_STATUS_OUT_OF_MEMORY
                                                : VIBEQC_STATUS_CUDA_ERROR;
}

vibeqc_status blas_status(cublasStatus_t status) {
  if (status == CUBLAS_STATUS_SUCCESS) return VIBEQC_STATUS_SUCCESS;
  return status == CUBLAS_STATUS_ALLOC_FAILED ? VIBEQC_STATUS_OUT_OF_MEMORY
                                              : VIBEQC_STATUS_CUDA_ERROR;
}

vibeqc_status copy_to_device(void* destination, const void* source, std::size_t bytes,
                             cudaStream_t stream) {
  if (bytes == 0) return VIBEQC_STATUS_SUCCESS;
  return cuda_status(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream));
}

}  // namespace vibeqc::scf::cuda_execution
