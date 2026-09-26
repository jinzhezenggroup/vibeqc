#include "scf/cuda/eigensolver.hpp"

#include <stdexcept>

#include "runtime/resource_cuda.cuh"
#include "scf/cuda/eigensolver_kernels.hpp"
#include "scf/cuda/launch_geometry.hpp"
#include "scf/eigensolver_workspace.hpp"
#include "vibeqc/vibeqc.hpp"

namespace vibeqc::scf::cuda_execution {

/** Host-only provider dispatch and profiling order. Graph eligibility is resolved at setup before
 * this execution boundary. */
namespace {
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

}  // namespace

bool provider_eigensolver(CudaEigensolverFamily family) {
  return family == CudaEigensolverFamily::jacobi_batched ||
         family == CudaEigensolverFamily::xsyev_batched || family == CudaEigensolverFamily::xsyevd;
}

OrdinaryStreamEigensolver::OrdinaryStreamEigensolver(cudaStream_t stream, int n,
                                                     const double* matrix,
                                                     const double* eigenvalues)
    : n_(n) {
  if (n <= 0 || !stream || !matrix || !eigenvalues)
    throw std::invalid_argument("invalid ordinary eigensolver owner");
  const auto checked = [](vibeqc_status status) {
    if (status == VIBEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
    if (status != VIBEQC_STATUS_SUCCESS)
      throw vibeqc::Error(status, "ordinary CUDA eigensolver preparation failed");
  };
  checked(cuda_status(cudaGetDevice(&device_)));
  resources_.stream_ = stream;
  if (n <= kSmallEigensolverLimit) return;
  try {
    cudaStreamCaptureStatus capture{};
    checked(cuda_status(cudaStreamIsCapturing(stream, &capture)));
    if (capture != cudaStreamCaptureStatusNone)
      throw std::invalid_argument("ordinary eigensolver preparation cannot capture");
    checked(solver_status(cusolverDnCreate(&resources_.solver_)));
    checked(solver_status(cusolverDnSetStream(resources_.solver_, stream)));
    checked(solver_status(cusolverDnCreateParams(&resources_.solver_parameters_)));
    checked(solver_status(cusolverDnXsyevd_bufferSize(
        resources_.solver_, resources_.solver_parameters_, CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER, n, CUDA_R_64F, matrix, n, CUDA_R_64F, eigenvalues, CUDA_R_64F,
        &resources_.solver_workspace_bytes_, &resources_.solver_host_workspace_bytes_)));
    const auto allowance = ordinary_eigensolver_workspace_allowance(n);
    if (resources_.solver_workspace_bytes_ > allowance ||
        resources_.solver_host_workspace_bytes_ > allowance)
      throw std::bad_alloc();
    if (resources_.solver_workspace_bytes_)
      checked(cuda_status(runtime::resource_cuda_malloc(&resources_.solver_workspace_,
                                                        resources_.solver_workspace_bytes_)));
    host_workspace_.resize(resources_.solver_host_workspace_bytes_);
    resources_.solver_host_workspace_ = host_workspace_.data();
  } catch (...) {
    cleanup();
    throw;
  }
}

void OrdinaryStreamEigensolver::cleanup() noexcept {
  int previous = device_;
  (void)cudaGetDevice(&previous);
  (void)cudaSetDevice(device_);
  if (resources_.stream_) (void)cudaStreamSynchronize(resources_.stream_);
  if (resources_.solver_workspace_) (void)runtime::resource_cuda_free(resources_.solver_workspace_);
  if (resources_.solver_parameters_) (void)cusolverDnDestroyParams(resources_.solver_parameters_);
  if (resources_.solver_) (void)cusolverDnDestroy(resources_.solver_);
  resources_ = {};
  (void)cudaSetDevice(previous);
}

OrdinaryStreamEigensolver::~OrdinaryStreamEigensolver() { cleanup(); }

vibeqc_status OrdinaryStreamEigensolver::launch(int batch, double* matrices,
                                                double* native_workspace, double* eigenvalues,
                                                int* info, const std::uint8_t* active) const {
  if (batch <= 0 || !matrices || !native_workspace || !eigenvalues || !info || !active)
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  int device = -1;
  auto error = cudaGetDevice(&device);
  if (error != cudaSuccess) return cuda_status(error);
  if (device != device_) return VIBEQC_STATUS_INVALID_ARGUMENT;
  cudaStreamCaptureStatus capture{};
  error = cudaStreamIsCapturing(resources_.stream_, &capture);
  if (error != cudaSuccess) return cuda_status(error);
  if (capture != cudaStreamCaptureStatusNone) return VIBEQC_STATUS_INVALID_ARGUMENT;
  const auto family = n_ <= kSmallEigensolverLimit ? CudaEigensolverFamily::small_native
                                                   : CudaEigensolverFamily::xsyevd;
  return launch_solver(resources_, family, n_, batch, matrices, native_workspace, eigenvalues, 0,
                       info, active);
}

vibeqc_status launch_solver(const EigensolverResources& resources, CudaEigensolverFamily family,
                            int nbf, int batch_size, double* matrices,
                            double* eigenvector_workspace, double* eigenvalues, int lwork,
                            int* info, const std::uint8_t* active,
                            const EigensolverProfileLaunch* profile) {
  const bool provider_invoked = provider_eigensolver(family);
  if (profile != nullptr) {
    launch_begin_inactive_eigensolver_profile_kernel(
        1, 1, 0, resources.stream_, profile->physical_batch_size, batch_size,
        static_cast<std::uint32_t>(family), provider_invoked, profile->cublas_transformed_inactive,
        profile->physical_active, active, profile->capacity, profile->count, profile->entries);
    const cudaError_t profile_error = cudaPeekAtLastError();
    if (profile_error != cudaSuccess) return cuda_status(profile_error);
  }
  if (provider_invoked) {
    // One block per solver state returns immediately for active matrices. The
    // homogeneous fast path therefore pays one tiny mask kernel while a
    // divergent provider batch receives finite identity placeholders.
    launch_sanitize_inactive_solver_input_kernel(
        static_cast<unsigned>(batch_size), kCaptureSafeKernelThreads, 0, resources.stream_,
        batch_size, nbf, active, matrices, info, profile == nullptr ? 0U : profile->capacity,
        profile == nullptr ? nullptr : profile->count,
        profile == nullptr ? nullptr : profile->entries);
    const cudaError_t sanitize_error = cudaPeekAtLastError();
    if (sanitize_error != cudaSuccess) return cuda_status(sanitize_error);
  }
  if (profile != nullptr) {
    launch_start_inactive_eigensolver_timer_kernel(1, 1, 0, resources.stream_, profile->capacity,
                                                   profile->count, profile->entries);
    const cudaError_t profile_error = cudaPeekAtLastError();
    if (profile_error != cudaSuccess) return cuda_status(profile_error);
  }
  vibeqc_status status = VIBEQC_STATUS_SUCCESS;
  if (family == CudaEigensolverFamily::small_native) {
    launch_symmetric_eigen_small_kernel(static_cast<unsigned>(batch_size), 1, 0, resources.stream_,
                                        batch_size, nbf, matrices, eigenvalues, info, active);
    status = cuda_status(cudaPeekAtLastError());
  } else if (family == CudaEigensolverFamily::jacobi_batched) {
    const cusolverStatus_t status = cusolverDnDsyevjBatched(
        resources.solver_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, nbf, matrices, nbf,
        eigenvalues, static_cast<double*>(resources.solver_workspace_), lwork, info,
        resources.jacobi_, batch_size);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      return solver_status(status);
    }
  } else if (family == CudaEigensolverFamily::xsyev_batched) {
    // The setup-time exact-stack probe has already captured, instantiated,
    // host-replayed, and device-tail-replayed this signature.
    const cusolverStatus_t status = cusolverDnXsyevBatched(
        resources.solver_, resources.solver_parameters_, CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER, nbf, CUDA_R_64F, matrices, nbf, CUDA_R_64F, eigenvalues, CUDA_R_64F,
        resources.solver_workspace_, resources.solver_workspace_bytes_,
        resources.solver_host_workspace_, resources.solver_host_workspace_bytes_, info, batch_size);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      return solver_status(status);
    }
  } else if (family == CudaEigensolverFamily::xsyevd) {
    // GPU4PySCF uses the ordinary single-matrix Xsyevd/Sygvd family for
    // large AO spaces.  Unlike XsyevBatched, this provider is intentionally
    // kept outside CUDA Graph capture.  Calls are serialized on the owning
    // stream and reuse one workspace, which also makes a multi-system bucket
    // deterministic without requiring a pointer-array API.
    const std::size_t matrix_elements =
        static_cast<std::size_t>(nbf) * static_cast<std::size_t>(nbf);
    for (int system = 0; system < batch_size; ++system) {
      const cusolverStatus_t status = cusolverDnXsyevd(
          resources.solver_, resources.solver_parameters_, CUSOLVER_EIG_MODE_VECTOR,
          CUBLAS_FILL_MODE_LOWER, static_cast<std::int64_t>(nbf), CUDA_R_64F,
          matrices + static_cast<std::size_t>(system) * matrix_elements,
          static_cast<std::int64_t>(nbf), CUDA_R_64F,
          eigenvalues + static_cast<std::size_t>(system) * nbf, CUDA_R_64F,
          resources.solver_workspace_, resources.solver_workspace_bytes_,
          resources.solver_host_workspace_, resources.solver_host_workspace_bytes_, info + system);
      if (status != CUSOLVER_STATUS_SUCCESS) {
        return solver_status(status);
      }
    }
  } else {
    // API-ineligible or Graph-rejected signatures retain the unbounded native
    // implementation without treating a provider limitation as a calculation
    // failure.
    launch_symmetric_eigen_graph_maximum_pivot_kernel(
        static_cast<unsigned>(batch_size), kGraphEigensolverThreads, 0, resources.stream_,
        batch_size, nbf, matrices, eigenvector_workspace, eigenvalues, info, active);
    status = cuda_status(cudaPeekAtLastError());
  }
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  if (profile != nullptr) {
    launch_finish_inactive_eigensolver_profile_kernel(1, 1, 0, resources.stream_, batch_size,
                                                      active, info, profile->capacity,
                                                      profile->count, profile->entries);
    status = cuda_status(cudaPeekAtLastError());
  }
  return status;
}

}  // namespace vibeqc::scf::cuda_execution
