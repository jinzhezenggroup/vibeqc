#include "scf/cuda/rhf_graph.hpp"

#include <mutex>

#include "runtime/allocation_measurement.hpp"

namespace vibeqc::scf::cuda_execution {

void RhfIterationGraphs::destroy(cudaGraph_t& graph, cudaGraphExec_t& executable) noexcept {
  if (executable != nullptr) {
    (void)cudaGraphExecDestroy(executable);
    executable = nullptr;
  }
  if (graph != nullptr) {
    (void)cudaGraphDestroy(graph);
    graph = nullptr;
  }
}

RhfIterationGraphs::~RhfIterationGraphs() {
  std::lock_guard<std::mutex> allocation_lock(runtime::allocation_measurement_mutex);
  if (device_id_ >= 0) (void)cudaSetDevice(device_id_);
  destroy(post_eigensolver_graph_, post_eigensolver_graph_exec_);
  destroy(iteration_graph_, iteration_graph_exec_);
}

RhfGraphCaptureResult RhfIterationGraphs::capture(cudaStream_t stream, cudaGraph_t& graph,
                                                  cudaGraphExec_t& executable,
                                                  unsigned long long instantiate_flags,
                                                  bool synchronize_before,
                                                  const std::function<vibeqc_status()>& body) {
  destroy(graph, executable);

  cudaError_t error = synchronize_before ? cudaStreamSynchronize(stream) : cudaSuccess;
  if (error == cudaSuccess) {
    error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  }
  if (error != cudaSuccess) return {VIBEQC_STATUS_SUCCESS, error};

  vibeqc_status body_status = VIBEQC_STATUS_SUCCESS;
  try {
    body_status = body();
  } catch (...) {
    // Preserve the original exception while releasing the stream's capture
    // state. Teardown and a later retry must not inherit an abandoned capture.
    cudaGraph_t abandoned_graph = nullptr;
    (void)cudaStreamEndCapture(stream, &abandoned_graph);
    if (abandoned_graph != nullptr) (void)cudaGraphDestroy(abandoned_graph);
    throw;
  }
  if (body_status != VIBEQC_STATUS_SUCCESS) {
    cudaGraph_t abandoned_graph = nullptr;
    (void)cudaStreamEndCapture(stream, &abandoned_graph);
    if (abandoned_graph != nullptr) (void)cudaGraphDestroy(abandoned_graph);
    return {body_status, cudaSuccess};
  }

  error = cudaStreamEndCapture(stream, &graph);
  if (error != cudaSuccess || graph == nullptr) {
    return {VIBEQC_STATUS_SUCCESS, error != cudaSuccess ? error : cudaErrorInvalidResourceHandle};
  }

  error = cudaGraphInstantiate(&executable, graph, instantiate_flags);
  if (error == cudaSuccess) error = cudaGraphUpload(executable, stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  return {VIBEQC_STATUS_SUCCESS, error};
}

RhfGraphCaptureResult RhfIterationGraphs::capture_iteration(
    int device_id, cudaStream_t stream, bool device_launch,
    const std::function<vibeqc_status()>& body) {
  device_id_ = device_id;
  const auto instantiate_flags =
      device_launch ? static_cast<unsigned long long>(cudaGraphInstantiateFlagDeviceLaunch) : 0ULL;
  return capture(stream, iteration_graph_, iteration_graph_exec_, instantiate_flags, true, body);
}

RhfGraphCaptureResult RhfIterationGraphs::capture_post_eigensolver(
    int device_id, cudaStream_t stream, const std::function<vibeqc_status()>& body) {
  device_id_ = device_id;
  return capture(stream, post_eigensolver_graph_, post_eigensolver_graph_exec_, 0ULL, false, body);
}

cudaError_t RhfIterationGraphs::launch_iteration(cudaStream_t stream) const noexcept {
  return cudaGraphLaunch(iteration_graph_exec_, stream);
}

cudaError_t RhfIterationGraphs::launch_post_eigensolver(cudaStream_t stream) const noexcept {
  return cudaGraphLaunch(post_eigensolver_graph_exec_, stream);
}

}  // namespace vibeqc::scf::cuda_execution
