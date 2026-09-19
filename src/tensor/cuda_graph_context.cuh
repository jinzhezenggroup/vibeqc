// TensorIR adapter for the shared graph owner. Scientific launches stay emitted.
#pragma once
#include "../runtime/cuda_graph_region.cuh"
#include "cuda_runtime.cuh"

namespace vibeqc_tensor {
struct GraphContext : Context {
  vibeqc::runtime::CudaGraphRegion graph;
  vibeqc::runtime::GraphBinding binding;
  bool graph_enabled = false;

  void configure_graph(bool enabled, const char* qualification) {
    check_device();
    if (!qualification || std::strlen(qualification) != 64)
      throw std::invalid_argument("invalid graph qualification identity");
    cuda_check(cudaStreamSynchronize(stream));
    std::lock_guard<std::mutex> allocation_lock(vibeqc::runtime::allocation_measurement_mutex);
    graph.invalidate();
    graph_enabled = enabled;
    binding = {qualification, device, stream, arena, handle};
  }
  template <class F>
  void submit_region(bool profile, F operation) {
    binding.device = device;
    binding.stream = stream;
    binding.arena = arena;
    binding.library = handle;
    graph.submit(binding, graph_enabled, profile, operation);
  }
  ~GraphContext() {
    std::lock_guard<std::mutex> allocation_lock(vibeqc::runtime::allocation_measurement_mutex);
    // Keep destruction nonthrowing and destroy graph references before base
    // Context destroys the stream, library handle, events and arena.
    int previous = 0;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    if (stream) cudaStreamSynchronize(stream);
    graph.release();
    cudaSetDevice(previous);
  }
};
}  // namespace vibeqc_tensor
