#pragma once

#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <cstddef>

#include "scf/cuda/eigensolver_types.hpp"
#include "scf/cuda_batch.hpp"

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
