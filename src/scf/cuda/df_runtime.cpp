#include "scf/cuda/df_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "runtime/resource_cuda.cuh"

namespace vibeqc::scf::cuda_df {

// Allocation registration stays with the common resource ledger.
bool checked_multiply(std::size_t first, std::size_t second, std::size_t& product) {
  if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first) {
    return false;
  }
  product = first * second;
  return true;
}

bool checked_bytes(std::size_t elements, std::size_t& bytes) {
  return checked_multiply(elements, sizeof(double), bytes);
}

std::size_t saturating_bytes(long double bytes) {
  if (!(bytes > 0.0L)) return 0U;
  const long double limit = static_cast<long double>(std::numeric_limits<std::size_t>::max());
  return bytes >= limit ? std::numeric_limits<std::size_t>::max() : static_cast<std::size_t>(bytes);
}

bool finite_values(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

vibeqc_status cuda_failure(cudaError_t error, const char* operation, std::string& detail) {
  detail = std::string(operation) + ": " + cudaGetErrorString(error);
  return error == cudaErrorMemoryAllocation ? VIBEQC_STATUS_OUT_OF_MEMORY
                                            : VIBEQC_STATUS_CUDA_ERROR;
}

vibeqc_status blas_failure(cublasStatus_t status, const char* operation, std::string& detail) {
  detail = std::string(operation) + " failed with cuBLAS status " +
           std::to_string(static_cast<int>(status));
  return status == CUBLAS_STATUS_ALLOC_FAILED ? VIBEQC_STATUS_OUT_OF_MEMORY
                                              : VIBEQC_STATUS_CUDA_ERROR;
}

vibeqc_status solver_failure(cusolverStatus_t status, const char* operation, std::string& detail) {
  detail = std::string(operation) + " failed with cuSOLVER status " +
           std::to_string(static_cast<int>(status));
  return status == CUSOLVER_STATUS_ALLOC_FAILED ? VIBEQC_STATUS_OUT_OF_MEMORY
                                                : VIBEQC_STATUS_CUDA_ERROR;
}

vibeqc_status allocate_device(void** pointer, std::size_t bytes, const char* description,
                              std::string& detail) {
  const cudaError_t error = runtime::resource_cuda_malloc(pointer, bytes);
  return error == cudaSuccess ? VIBEQC_STATUS_SUCCESS : cuda_failure(error, description, detail);
}

}  // namespace vibeqc::scf::cuda_df
