// Shared, method-neutral CUDA Graph lifecycle for compiler-qualified regions.
#pragma once
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "allocation_measurement.hpp"

namespace vibeqc::runtime {
struct GraphMetrics {
  uint64_t capture_attempts = 0, captures = 0, replays = 0, fallbacks = 0;
  uint64_t invalidations = 0, node_count = 0, retained_device_bytes = 0;
  double capture_ms = 0, instantiate_ms = 0, submission_ms = 0;
  int32_t mode = 0;  // ordinary, warmup, captured, replay, fallback, profiling
};

// A binding is owner-local, never an on-disk graph-cache key. The qualification
// digest covers compiler/artifact/schedule/specialization/runtime identities.
struct GraphBinding {
  std::string qualification;
  int device = 0;
  cudaStream_t stream = nullptr;
  const void* arena = nullptr;
  const void* library = nullptr;
  bool operator==(const GraphBinding& other) const {
    return qualification == other.qualification && device == other.device &&
           stream == other.stream && arena == other.arena && library == other.library;
  }
};

class CudaGraphRegion {
 public:
  GraphMetrics metrics;
  std::string reason = "graph mode not requested";
  CudaGraphRegion() = default;
  CudaGraphRegion(const CudaGraphRegion&) = delete;
  CudaGraphRegion& operator=(const CudaGraphRegion&) = delete;
  ~CudaGraphRegion() { release(); }

  // Caller owns the current device, stream, serialization and completion fence.
  // Destroy the executable BEFORE its buffers, stream, handles or code module.
  void release() noexcept {
    if (executable_) cudaGraphExecDestroy(executable_);
    executable_ = nullptr;
    metrics.node_count = metrics.retained_device_bytes = 0;
  }
  void invalidate() {
    release();
    bound_ = warmed_ = failed_ = false;
    ++metrics.invalidations;
    reason = "invalidated; ordinary warmup required";
  }

  template <class F>
  void submit(const GraphBinding& binding, bool enabled, bool profile, F operation) {
    const auto started = Clock::now();
    if (!bound_ || !(binding_ == binding)) {
      if (bound_) invalidate();
      binding_ = binding;
      bound_ = true;
    }
    if (!enabled || profile) {
      metrics.mode = profile ? 5 : 0;
      operation();
    } else if (failed_) {
      metrics.mode = 4;
      ++metrics.fallbacks;
      operation();
    } else if (!warmed_) {
      // Execute exactly once: do not secretly repeat a stateful region to warm
      // it up. Inputs may be refreshed before the next call/capture attempt.
      operation();
      warmed_ = true;
      metrics.mode = 1;
      reason = "ordinary warmup completed";
    } else {
      const bool cached = executable_ != nullptr;
      if (!cached) capture(binding.stream, operation);
      if (executable_) {
        // A launch/async execution error is NOT a capture-eligibility failure.
        // Never retry a possibly submitted scientific operation twice.
        check(cudaGraphLaunch(executable_, binding.stream));
        ++metrics.replays;
        metrics.mode = cached ? 3 : 2;
        reason = "captured device region";
      } else {
        ++metrics.fallbacks;
        metrics.mode = 4;
        operation();
      }
    }
    metrics.submission_ms = elapsed(started);
  }

 private:
  using Clock = std::chrono::steady_clock;
  cudaGraphExec_t executable_ = nullptr;
  GraphBinding binding_;
  bool bound_ = false, warmed_ = false, failed_ = false;
  static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  }
  static void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  }
  static bool capture_failure(cudaError_t status) {
    // CUDA's stable public error codes; compatibility headers such as CuMetal
    // may omit these enumerator names while retaining the runtime status ABI.
    const auto code = static_cast<int>(status);
    return code == 900 || code == 901 || status == cudaErrorNotSupported ||
           status == cudaErrorMemoryAllocation;
  }
  void fail(const char* text) {
    failed_ = true;
    reason = text;
  }
  template <class F>
  void capture(cudaStream_t stream, F operation) {
    std::lock_guard<std::mutex> allocation_lock(allocation_measurement_mutex);
    ++metrics.capture_attempts;
    size_t before = 0, after = 0, total = 0;
    check(cudaMemGetInfo(&before, &total));
    const auto started = Clock::now();
    auto status = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (status != cudaSuccess) {
      metrics.capture_ms += elapsed(started);
      if (!capture_failure(status)) check(status);
      fail("stream capture unavailable; ordinary fallback");
      cudaGetLastError();
      return;
    }
    cudaGraph_t graph = nullptr;
    try {
      operation();
    } catch (...) {
      // Every successful BeginCapture MUST have an EndCapture, even after a
      // library/launch exception. Only then may the stream execute normally.
      status = cudaStreamEndCapture(stream, &graph);
      if (graph) cudaGraphDestroy(graph);
      metrics.capture_ms += elapsed(started);
      if (status != cudaSuccess && !capture_failure(status)) check(status);
      cudaGetLastError();
      fail("capture submission failed; ordinary fallback");
      return;
    }
    status = cudaStreamEndCapture(stream, &graph);
    metrics.capture_ms += elapsed(started);
    if (status != cudaSuccess || !graph) {
      if (graph) cudaGraphDestroy(graph);
      if (status != cudaSuccess && !capture_failure(status)) check(status);
      cudaGetLastError();
      fail("capture ended without an executable graph; ordinary fallback");
      return;
    }
    size_t nodes = 0;
    status = cudaGraphGetNodes(graph, nullptr, &nodes);
    if (status != cudaSuccess) {
      cudaGraphDestroy(graph);
      check(status);
    }
    // Compiler eligibility already bounds launches. Retain a defensive native
    // ceiling for future consumers/providers that emit unexpected extra nodes.
    if (nodes > 4096) {
      cudaGraphDestroy(graph);
      fail("captured node ceiling exceeded; ordinary fallback");
      return;
    }
    const auto instantiate_started = Clock::now();
    status = cudaGraphInstantiate(&executable_, graph, nullptr, nullptr, 0);
    metrics.instantiate_ms += elapsed(instantiate_started);
    cudaGraphDestroy(graph);
    if (status != cudaSuccess) {
      release();
      // Instantiation has not submitted any graph work; ordinary execution is
      // safe for resource/capture/unsupported failures, not device-fatal errors.
      if (!capture_failure(status)) check(status);
      cudaGetLastError();
      fail("graph instantiation unavailable; ordinary fallback");
      return;
    }
    ++metrics.captures;
    metrics.node_count = nodes;
    check(cudaMemGetInfo(&after, &total));
    metrics.retained_device_bytes = before > after ? before - after : 0;
  }
};
}  // namespace vibeqc::runtime
