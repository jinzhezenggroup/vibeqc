#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <cstddef>

#include "scf/cuda/eigensolver.hpp"
#include "scf/cuda/matrix_library.hpp"

namespace vibeqc::scf::cuda_execution {

struct DirectTileValidationRecord;

/** Sole owner of a bucket's streams, graphs, library workspaces and arena.
 * Teardown preserves stream-ordered allocation release and selects the owning
 * device. Borrowed eigensolver/matrix views never acquire lifetime ownership.
 */
class CudaResources {
 public:
  ~CudaResources();

  /** Borrow library state while retaining ownership in the prepared bucket. */
  EigensolverResources eigensolver_view() const;

  /** Borrow only the stream and BLAS handle; matrix execution owns no bucket state. */
  MatrixLibraryResources matrix_view() const { return {stream_, blas_}; }

  int device_id_{-1};
  cudaStream_t stream_{};
  cublasHandle_t blas_{};
  cusolverDnHandle_t solver_{};
  cusolverDnParams_t solver_parameters_{};
  syevjInfo_t jacobi_{};
  cudaGraph_t iteration_graph_{};
  cudaGraphExec_t iteration_graph_exec_{};
  // cuSOLVER XsyevBatched above 512 AOs executes efficiently on an ordinary
  // stream but rejects CUDA Graph capture on CUDA 12.9.  Large-matrix SCF
  // therefore replays a pre-solver Graph, launches the provider normally,
  // then replays this post-solver Graph under host convergence control.
  cudaGraph_t post_eigensolver_graph_{};
  cudaGraphExec_t post_eigensolver_graph_exec_{};
  void* arena_{};
  DirectTileValidationRecord* direct_tile_validation_{};
  void* solver_workspace_{};
  std::size_t solver_workspace_bytes_{};
  void* solver_host_workspace_{};
  std::size_t solver_host_workspace_bytes_{};
  /** Numeric reference peak includes the observed provider allocations. */
  std::size_t reference_peak_bytes_{};
  std::size_t provider_retained_bytes_{};
};

}  // namespace vibeqc::scf::cuda_execution
