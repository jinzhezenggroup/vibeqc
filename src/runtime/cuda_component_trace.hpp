#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace vibeqc::runtime::cuda_trace {

/** Shape and route of one mathematical DF operation, independent of plan ownership. */
struct TraceShape {
  std::size_t systems{};
  std::size_t nbf{};
  std::size_t naux{};
  bool source_backed{};
  bool streamed{};
  // A force operation may borrow one member of a batched integral source.
  std::size_t system_offset{};
};

/** Opt-in component trace for one J, K, or analytic-response invocation.
 * Set VIBEQC_DF_TRACE to a JSONL path before execution. Ordinary execution then
 * records CUDA events and drains only this operation's final event before
 * writing a record. Timing runs with tracing are diagnostic runs; compare
 * endpoint performance separately with tracing disabled.
 *
 * Capture records contain host construction time and logical work only. They
 * are explicitly marked graph_capture, never reported as executed GPU work.
 * No event, synchronization, allocation, or file I/O is added when disabled.
 * Trace state is thread-local and cannot escape the synchronous host call.
 */
class TraceOperation {
 public:
  TraceOperation(const char* operation, cudaStream_t stream, TraceShape shape) noexcept;
  ~TraceOperation();
  TraceOperation(const TraceOperation&) = delete;
  TraceOperation& operator=(const TraceOperation&) = delete;

 private:
  struct State;
  std::unique_ptr<State> state_;
  friend class TraceRegion;
  friend void trace_counter(const char*, std::uint64_t) noexcept;
  friend void trace_tile(std::size_t, std::size_t, std::size_t, std::size_t, std::size_t,
                         std::int64_t, bool) noexcept;
  static thread_local State* active_;
};

/** Nested component on the active operation's stream, with inclusive timings.
 * The ledger records parent indices so aggregation can subtract immediate
 * children without double counting. finish() permits an allocation/transfer
 * phase to end while its buffers remain in the original lexical scope.
 */
class TraceRegion {
 public:
  TraceRegion(const char* name, cudaStream_t stream) noexcept;
  ~TraceRegion();
  void finish() noexcept;
  TraceRegion(const TraceRegion&) = delete;
  TraceRegion& operator=(const TraceRegion&) = delete;

 private:
  TraceOperation::State* state_{};
  std::size_t index_{};
};

/** Time a synchronous host submission while preserving its return/exception. */
template <class Function>
decltype(auto) trace_call(const char* name, cudaStream_t stream, Function&& function) {
  TraceRegion region(name, stream);
  return function();
}

/** Semantic work counts, collected only under an enabled operation. */
void trace_counter(const char* name, std::uint64_t count) noexcept;

/** Count production of one logical public-basis tile, before its launch.
 * Identical ranges share a key across Coulomb passes and exchange row/column
 * visits. Raw and metric-transformed values and derivative coordinates remain
 * distinct. Empty tiles should not call this function.
 */
void trace_tile(std::size_t system, std::size_t pair_begin, std::size_t pair_count,
                std::size_t auxiliary_begin, std::size_t auxiliary_count,
                std::int64_t derivative_coordinate, bool transformed) noexcept;

}  // namespace vibeqc::runtime::cuda_trace
