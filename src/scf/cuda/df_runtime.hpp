#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cstddef>
#include <string>
#include <vector>

#include "vibeqc/vibeqc.h"

namespace vibeqc::scf::cuda_df {

/** Checked allocation and status helpers shared only by DF runtime owners. */
inline constexpr unsigned kThreads = 256;
inline unsigned blocks_for(std::size_t elements) {
  return static_cast<unsigned>((elements + kThreads - 1) / kThreads);
}

bool checked_multiply(std::size_t first, std::size_t second, std::size_t& product);
bool checked_bytes(std::size_t elements, std::size_t& bytes);
std::size_t saturating_bytes(long double bytes);
bool finite_values(const std::vector<double>& values);
vibeqc_status cuda_failure(cudaError_t error, const char* operation, std::string& detail);
vibeqc_status blas_failure(cublasStatus_t status, const char* operation, std::string& detail);
vibeqc_status solver_failure(cusolverStatus_t status, const char* operation, std::string& detail);
vibeqc_status allocate_device(void** pointer, std::size_t bytes, const char* description,
                              std::string& detail);

/** Count capacity, including allocator slack, without wrapping diagnostics. */
template <typename T>
std::size_t vector_capacity_bytes(const std::vector<T>& values) {
  return saturating_bytes(static_cast<long double>(values.capacity()) *
                          static_cast<long double>(sizeof(T)));
}

}  // namespace vibeqc::scf::cuda_df
