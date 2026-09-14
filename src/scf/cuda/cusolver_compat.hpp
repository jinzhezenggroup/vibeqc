#pragma once

#include <cusolverDn.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace vibeqc::scf::cuda_compat {
namespace detail {

template <class Fn>
inline cusolverStatus_t xsyev_batched_buffer_size_dispatch(
    Fn function, cusolverDnHandle_t handle, cusolverDnParams_t parameters, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, std::int64_t n, cudaDataType data_type_a, const void* a,
    std::int64_t lda, cudaDataType data_type_w, const void* w, cudaDataType compute_type,
    std::size_t* device_bytes, std::size_t* host_bytes, std::int64_t batch_size) {
  constexpr bool official =
      std::is_invocable_r_v<cusolverStatus_t, Fn, cusolverDnHandle_t, cusolverDnParams_t,
                            cusolverEigMode_t, cublasFillMode_t, std::int64_t, cudaDataType,
                            const void*, std::int64_t, cudaDataType, const void*, cudaDataType,
                            std::size_t*, std::size_t*, std::int64_t>;
  constexpr bool strided =
      std::is_invocable_r_v<cusolverStatus_t, Fn, cusolverDnHandle_t, cusolverDnParams_t,
                            cusolverEigMode_t, cublasFillMode_t, std::int64_t, cudaDataType,
                            const void*, std::int64_t, std::int64_t, cudaDataType, const void*,
                            std::int64_t, cudaDataType, std::int64_t, std::size_t*, std::size_t*>;
  static_assert(official || strided, "unsupported cusolverDnXsyevBatched_bufferSize signature");
  if constexpr (official) {
    return function(handle, parameters, jobz, uplo, n, data_type_a, a, lda, data_type_w, w,
                    compute_type, device_bytes, host_bytes, batch_size);
  } else {
    const std::int64_t stride_a = lda * n;
    const std::int64_t stride_w = n;
    return function(handle, parameters, jobz, uplo, n, data_type_a, a, lda, stride_a, data_type_w,
                    w, stride_w, compute_type, batch_size, device_bytes, host_bytes);
  }
}

template <class Fn>
inline cusolverStatus_t xsyev_batched_dispatch(
    Fn function, cusolverDnHandle_t handle, cusolverDnParams_t parameters, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, std::int64_t n, cudaDataType data_type_a, void* a, std::int64_t lda,
    cudaDataType data_type_w, void* w, cudaDataType compute_type, void* device_workspace,
    std::size_t device_bytes, void* host_workspace, std::size_t host_bytes, int* info,
    std::int64_t batch_size) {
  constexpr bool official =
      std::is_invocable_r_v<cusolverStatus_t, Fn, cusolverDnHandle_t, cusolverDnParams_t,
                            cusolverEigMode_t, cublasFillMode_t, std::int64_t, cudaDataType, void*,
                            std::int64_t, cudaDataType, void*, cudaDataType, void*, std::size_t,
                            void*, std::size_t, int*, std::int64_t>;
  constexpr bool strided =
      std::is_invocable_r_v<cusolverStatus_t, Fn, cusolverDnHandle_t, cusolverDnParams_t,
                            cusolverEigMode_t, cublasFillMode_t, std::int64_t, cudaDataType, void*,
                            std::int64_t, std::int64_t, cudaDataType, void*, std::int64_t,
                            cudaDataType, std::int64_t, void*, std::size_t, void*, std::size_t,
                            int*>;
  static_assert(official || strided, "unsupported cusolverDnXsyevBatched signature");
  if constexpr (official) {
    return function(handle, parameters, jobz, uplo, n, data_type_a, a, lda, data_type_w, w,
                    compute_type, device_workspace, device_bytes, host_workspace, host_bytes, info,
                    batch_size);
  } else {
    const std::int64_t stride_a = lda * n;
    const std::int64_t stride_w = n;
    return function(handle, parameters, jobz, uplo, n, data_type_a, a, lda, stride_a, data_type_w,
                    w, stride_w, compute_type, batch_size, device_workspace, device_bytes,
                    host_workspace, host_bytes, info);
  }
}

}  // namespace detail

// NVIDIA's public cusolverDnXsyevBatched API stores each batch contiguously and
// therefore has no explicit strides. Some CUDA-compatible providers expose the
// same operation with strideA/strideW arguments instead. Select between those
// signatures at compile time while preserving the same contiguous layout.
inline cusolverStatus_t xsyev_batched_buffer_size(
    cusolverDnHandle_t handle, cusolverDnParams_t parameters, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, std::int64_t n, cudaDataType data_type_a, const void* a,
    std::int64_t lda, cudaDataType data_type_w, const void* w, cudaDataType compute_type,
    std::size_t* device_bytes, std::size_t* host_bytes, std::int64_t batch_size) {
  return detail::xsyev_batched_buffer_size_dispatch(
      &cusolverDnXsyevBatched_bufferSize, handle, parameters, jobz, uplo, n, data_type_a, a, lda,
      data_type_w, w, compute_type, device_bytes, host_bytes, batch_size);
}

inline cusolverStatus_t xsyev_batched(cusolverDnHandle_t handle, cusolverDnParams_t parameters,
                                      cusolverEigMode_t jobz, cublasFillMode_t uplo, std::int64_t n,
                                      cudaDataType data_type_a, void* a, std::int64_t lda,
                                      cudaDataType data_type_w, void* w, cudaDataType compute_type,
                                      void* device_workspace, std::size_t device_bytes,
                                      void* host_workspace, std::size_t host_bytes, int* info,
                                      std::int64_t batch_size) {
  return detail::xsyev_batched_dispatch(&cusolverDnXsyevBatched, handle, parameters, jobz, uplo, n,
                                        data_type_a, a, lda, data_type_w, w, compute_type,
                                        device_workspace, device_bytes, host_workspace, host_bytes,
                                        info, batch_size);
}

}  // namespace vibeqc::scf::cuda_compat
