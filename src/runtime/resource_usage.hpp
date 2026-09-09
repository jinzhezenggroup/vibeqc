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
  record.peak_bytes = std::max(record.peak_bytes, static_cast<std::uint64_t>(bytes));
  ++record.samples;
}

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
