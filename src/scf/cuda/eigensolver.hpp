#pragma once

#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <cstddef>
#include <vector>

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

/** Prepared ordinary-stream eigensolver with explicit numeric workspace.
 * Borrows its owner's stream and matrix/eigenvalue buffers. Small matrices
 * retain the native path; larger matrices reuse the existing Xsyevd dispatch.
 * Graph capture is rejected explicitly instead of silently substituting an
 * unbounded maximum-pivot solve. Construction queries and charges workspace
 * once; repeated solves and serialized spin states allocate no numeric buffers.
 */
class OrdinaryStreamEigensolver {
 public:
  OrdinaryStreamEigensolver(cudaStream_t stream, int n, const double* matrix,
                            const double* eigenvalues);
  ~OrdinaryStreamEigensolver();
  OrdinaryStreamEigensolver(const OrdinaryStreamEigensolver&) = delete;
  OrdinaryStreamEigensolver& operator=(const OrdinaryStreamEigensolver&) = delete;
  vibeqc_status launch(int batch, double* matrices, double* native_workspace, double* eigenvalues,
                       int* info, const std::uint8_t* active) const;
  std::size_t device_bytes() const noexcept { return resources_.solver_workspace_bytes_; }
  std::size_t host_bytes() const noexcept { return host_workspace_.capacity(); }

 private:
  void cleanup() noexcept;
  int n_{}, device_{};
  EigensolverResources resources_{};
  std::vector<unsigned char> host_workspace_;
};

}  // namespace vibeqc::scf::cuda_execution
