#ifndef VIBEQC_RUNTIME_RESOURCE_USAGE_HPP
#define VIBEQC_RUNTIME_RESOURCE_USAGE_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace vibeqc::runtime {

/** Opt-in samples of simultaneously owned CPU numeric buffers.
 *
 * A sample is an actual sum of vector capacities at an iteration boundary,
 * not an allocator high-water mark or a prediction. Recurrence/eigensolver
 * temporaries and other fleet items are outside this deliberately narrow
 * observation scope. Thread-local ownership isolates concurrent API callers.
 */
struct CpuResourceObservation {
  bool active{};
  std::uint64_t peak_bytes{};
  std::uint64_t samples{};
  /** Zero preserves ordinary fleet concurrency; one enforces the v1 bounded CPU schedule. */
  unsigned cpu_worker_limit{};
  std::uint64_t cuda_arena_peak_bytes{};
  std::uint64_t cuda_arena_samples{};
  /** Buffers retained by a synchronous outer owner, absent from item samples. */
  std::size_t outer_host_bytes{};
};

inline thread_local CpuResourceObservation cpu_resource_observation;

inline std::size_t add_capacity(std::size_t first, std::size_t second) noexcept {
  const auto maximum = std::numeric_limits<std::size_t>::max();
  return second > maximum - first ? maximum : first + second;
}

template <class T>
std::size_t vector_bytes(const std::vector<T>& values) noexcept {
  const auto maximum = std::numeric_limits<std::size_t>::max();
  return values.capacity() > maximum / sizeof(T) ? maximum : values.capacity() * sizeof(T);
}

template <class... Vectors>
std::size_t vector_capacities(const Vectors&... values) noexcept {
  std::size_t bytes = 0;
  ((bytes = add_capacity(bytes, vector_bytes(values))), ...);
  return bytes;
}

inline void sample_cpu_capacity(std::size_t bytes) noexcept {
  auto& record = cpu_resource_observation;
  if (!record.active) return;
  record.peak_bytes = std::max(
      record.peak_bytes, static_cast<std::uint64_t>(add_capacity(bytes, record.outer_host_bytes)));
  ++record.samples;
}

/** Compose retained fleet/seed buffers with an existing per-item observer.
 * Nested scopes restore their predecessor even when a numerical call fails.
 * These remain explicit capacity samples, not allocator high-water marks. */
class CpuRetainedCapacity {
 public:
  explicit CpuRetainedCapacity(std::size_t bytes) noexcept
      : previous_(cpu_resource_observation.outer_host_bytes) {
    cpu_resource_observation.outer_host_bytes = add_capacity(previous_, bytes);
  }
  ~CpuRetainedCapacity() { cpu_resource_observation.outer_host_bytes = previous_; }
  CpuRetainedCapacity(const CpuRetainedCapacity&) = delete;
  CpuRetainedCapacity& operator=(const CpuRetainedCapacity&) = delete;

 private:
  std::size_t previous_;
};

/** Samples owned direct-HF arenas; graph/driver allocations are excluded. */
inline void sample_cuda_arena_capacity(std::size_t bytes) noexcept {
  auto& record = cpu_resource_observation;
  if (!record.active || bytes == 0) return;
  record.cuda_arena_peak_bytes =
      std::max(record.cuda_arena_peak_bytes, static_cast<std::uint64_t>(bytes));
  ++record.cuda_arena_samples;
}

}  // namespace vibeqc::runtime

#endif
