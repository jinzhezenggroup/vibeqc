#pragma once

#include <cuda_runtime_api.h>

#include <functional>

#include "scf/types.hpp"

namespace vibeqc::scf::cuda_execution {

/** Result of constructing one reusable RHF/UHF host-controlled CUDA Graph. */
struct RhfGraphCaptureResult {
  vibeqc_status body_status{VIBEQC_STATUS_SUCCESS};
  cudaError_t cuda_error{cudaSuccess};

  bool ok() const noexcept {
    return body_status == VIBEQC_STATUS_SUCCESS && cuda_error == cudaSuccess;
  }
};

/**
 * Own the reusable CUDA Graph executables for one direct-HF bucket.
 *
 * Scientific launch order remains in the HF driver. This owner is limited to
 * capture lifecycle, graph instantiation/upload, replay, and teardown so graph
 * mechanics can change without rebuilding the direct numerical orchestration.
 * Callback exceptions end capture and release any abandoned graph before rethrow;
 * the borrowed stream remains reusable by teardown or a later retry.
 */
class RhfIterationGraphs {
 public:
  RhfIterationGraphs() = default;
  RhfIterationGraphs(const RhfIterationGraphs&) = delete;
  RhfIterationGraphs& operator=(const RhfIterationGraphs&) = delete;
  ~RhfIterationGraphs();

  RhfGraphCaptureResult capture_iteration(int device_id, cudaStream_t stream, bool device_launch,
                                          const std::function<vibeqc_status()>& body);
  RhfGraphCaptureResult capture_post_eigensolver(int device_id, cudaStream_t stream,
                                                 const std::function<vibeqc_status()>& body);

  cudaError_t launch_iteration(cudaStream_t stream) const noexcept;
  cudaError_t launch_post_eigensolver(cudaStream_t stream) const noexcept;

 private:
  RhfGraphCaptureResult capture(cudaStream_t stream, cudaGraph_t& graph,
                                cudaGraphExec_t& executable, unsigned long long instantiate_flags,
                                bool synchronize_before,
                                const std::function<vibeqc_status()>& body);
  static void destroy(cudaGraph_t& graph, cudaGraphExec_t& executable) noexcept;

  int device_id_{-1};
  cudaGraph_t iteration_graph_{};
  cudaGraphExec_t iteration_graph_exec_{};
  cudaGraph_t post_eigensolver_graph_{};
  cudaGraphExec_t post_eigensolver_graph_exec_{};
};

}  // namespace vibeqc::scf::cuda_execution
