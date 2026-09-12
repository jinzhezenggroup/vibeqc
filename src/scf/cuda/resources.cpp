#include "scf/cuda/resources.hpp"

#include <cstdlib>

#include "runtime/allocation_measurement.hpp"
#include "runtime/resource_cuda.cuh"

namespace vibeqc::scf::cuda_execution {

CudaResources::~CudaResources() {
  std::lock_guard<std::mutex> allocation_lock(runtime::allocation_measurement_mutex);
  if (device_id_ >= 0) (void)cudaSetDevice(device_id_);
  if (post_eigensolver_graph_exec_ != nullptr) {
    (void)cudaGraphExecDestroy(post_eigensolver_graph_exec_);
  }
  if (post_eigensolver_graph_ != nullptr) {
    (void)cudaGraphDestroy(post_eigensolver_graph_);
  }
  if (iteration_graph_exec_ != nullptr) {
    (void)cudaGraphExecDestroy(iteration_graph_exec_);
  }
  if (iteration_graph_ != nullptr) (void)cudaGraphDestroy(iteration_graph_);
  if (jacobi_ != nullptr) (void)cusolverDnDestroySyevjInfo(jacobi_);
  if (solver_parameters_ != nullptr) {
    (void)cusolverDnDestroyParams(solver_parameters_);
  }
  if (solver_ != nullptr) (void)cusolverDnDestroy(solver_);
  if (blas_ != nullptr) (void)cublasDestroy(blas_);
  if (stream_ != nullptr) {
    // Both allocations come from CUDA's stream-ordered device pool. Queue
    // their release on the owning bucket stream so destroying one plan does
    // not impose a device-wide synchronization on unrelated workloads.
    if (solver_workspace_ != nullptr) {
      (void)runtime::resource_cuda_free_async(solver_workspace_, stream_);
    }
    if (direct_tile_validation_ != nullptr) {
      (void)runtime::resource_cuda_free_async(direct_tile_validation_, stream_);
    }
    if (arena_ != nullptr) (void)runtime::resource_cuda_free_async(arena_, stream_);
    (void)cudaStreamSynchronize(stream_);
    (void)cudaStreamDestroy(stream_);
  }
  std::free(solver_host_workspace_);
}

EigensolverResources CudaResources::eigensolver_view() const {
  return {stream_,
          solver_,
          solver_parameters_,
          jacobi_,
          solver_workspace_,
          solver_workspace_bytes_,
          solver_host_workspace_,
          solver_host_workspace_bytes_};
}

}  // namespace vibeqc::scf::cuda_execution
