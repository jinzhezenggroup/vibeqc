#pragma once

#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <cstddef>
#include <cstdint>

#include "scf/cuda/cusolver_compat.hpp"
#include "scf/cuda/eigensolver_types.hpp"
#include "scf/cuda_batch.hpp"

namespace vibeqc::scf {

// Keep production call sites on NVIDIA's documented packed signature. Import
// the provider overloads and add a template fallback only on the CUDA runtime
// interface, not the provider-free policy header. NVIDIA's exact non-template
// overload wins when present; explicit-stride-only providers use the adapter.
using ::cusolverDnXsyevBatched;
using ::cusolverDnXsyevBatched_bufferSize;

template <class = void>
inline cusolverStatus_t cusolverDnXsyevBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t parameters, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, std::int64_t n, cudaDataType data_type_a, const void* a,
    std::int64_t lda, cudaDataType data_type_w, const void* w, cudaDataType compute_type,
    std::size_t* device_bytes, std::size_t* host_bytes, std::int64_t batch_size) {
  return cuda_compat::xsyev_batched_buffer_size(handle, parameters, jobz, uplo, n, data_type_a, a,
                                                lda, data_type_w, w, compute_type, device_bytes,
                                                host_bytes, batch_size);
}

template <class = void>
inline cusolverStatus_t cusolverDnXsyevBatched(
    cusolverDnHandle_t handle, cusolverDnParams_t parameters, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, std::int64_t n, cudaDataType data_type_a, void* a, std::int64_t lda,
    cudaDataType data_type_w, void* w, cudaDataType compute_type, void* device_workspace,
    std::size_t device_bytes, void* host_workspace, std::size_t host_bytes, int* info,
    std::int64_t batch_size) {
  return cuda_compat::xsyev_batched(handle, parameters, jobz, uplo, n, data_type_a, a, lda,
                                    data_type_w, w, compute_type, device_workspace, device_bytes,
                                    host_workspace, host_bytes, info, batch_size);
}

}  // namespace vibeqc::scf

namespace vibeqc::scf::cuda_execution {

/** Borrowed eigensolver resources; allocation, lifetime and exact-stack qualification remain with
 * the prepared owner. */
struct EigensolverResources {
  cudaStream_t stream_{};
  cusolverDnHandle_t solver_{};
  cusolverDnParams_t solver_parameters_{};
  syevjInfo_t jacobi_{};
  void* solver_workspace_{};
  std::size_t solver_workspace_bytes_{};
  void* solver_host_workspace_{};
  std::size_t solver_host_workspace_bytes_{};
};

struct EigensolverProfileLaunch {
  std::int32_t physical_batch_size{};
  const std::uint8_t* physical_active{};
  bool cublas_transformed_inactive{};
  std::uint32_t capacity{};
  std::uint32_t* count{};
  DeviceInactiveEigensolverProfileEntry* entries{};
};

/** Execute the resolved native/library family on the owning stream. Masks and diagnostics preserve
 * inactive terminal states. */
vibeqc_status launch_solver(const EigensolverResources& resources, CudaEigensolverFamily family,
                            int nbf, int batch_size, double* matrices,
                            double* eigenvector_workspace, double* eigenvalues, int lwork,
                            int* info, const std::uint8_t* active,
                            const EigensolverProfileLaunch* profile = nullptr);

/** Whether this family requires provider input sanitization and cuSOLVER workspace. */
bool provider_eigensolver(CudaEigensolverFamily family);

}  // namespace vibeqc::scf::cuda_execution
