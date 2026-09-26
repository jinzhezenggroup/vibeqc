#pragma once

#include <cstdint>

namespace vibeqc::scf::cuda_execution {

/** Generic eigensolver record layout and native launch limits, independent of HF provider storage.
 */
/** Compact device record retained only by the opt-in profiling Graph. */
struct DeviceInactiveEigensolverProfileEntry {
  std::uint64_t solver_start_nanoseconds;
  std::uint64_t solver_elapsed_nanoseconds;
  std::uint32_t iteration;
  std::uint32_t family;
  std::uint32_t physical_system_count;
  std::uint32_t solver_batch_count;
  std::uint32_t active_physical_count;
  std::uint32_t active_solver_count;
  std::uint32_t inactive_input_nonfinite_count;
  std::uint32_t inactive_submission_nonfinite_count;
  std::uint32_t inactive_info_nonzero_count;
  std::uint32_t inactive_touch_flags;
  std::uint32_t provider_invoked;
};

constexpr std::int32_t kSmallEigensolverLimit = 16;
constexpr std::int32_t kBatchedEigensolverLimit = 32;
constexpr unsigned kGraphEigensolverThreads = 64;

}  // namespace vibeqc::scf::cuda_execution
