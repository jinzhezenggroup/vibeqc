#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "core/types.hpp"

namespace vibeqc::runtime {

/** Host/device domains for method-neutral execution resource observations. */
enum class ExecutionMemorySpace : std::uint8_t { Host = 0, Device = 1 };

/** High-water observations supplied by execution owners.
 *
 * Numeric capacity and scratch workspace are deliberately separate: retained
 * method state must never be relabelled as transient workspace merely to make
 * accounting look uniform. These observations can overlap and must not be
 * summed into a total. Zero without an observation means unknown, not free.
 */
struct ExecutionResourceSnapshot {
  std::size_t host_numeric_peak_bytes{};
  std::size_t device_numeric_peak_bytes{};
  std::size_t host_workspace_peak_bytes{};
  std::size_t device_workspace_peak_bytes{};
  std::uint64_t numeric_observations{};
  std::uint64_t workspace_observations{};
};
class ExecutionResourceTracker {
 public:
  void observe_numeric_peak(ExecutionMemorySpace space, std::size_t bytes) noexcept {
    auto& peak = space == ExecutionMemorySpace::Host ? state_.host_numeric_peak_bytes
                                                     : state_.device_numeric_peak_bytes;
    peak = std::max(peak, bytes);
    increment(state_.numeric_observations);
  }

  void observe_workspace_peak(ExecutionMemorySpace space, std::size_t bytes) noexcept {
    auto& peak = space == ExecutionMemorySpace::Host ? state_.host_workspace_peak_bytes
                                                     : state_.device_workspace_peak_bytes;
    peak = std::max(peak, bytes);
    increment(state_.workspace_observations);
  }

  [[nodiscard]] ExecutionResourceSnapshot snapshot() const noexcept { return state_; }
  void reset() noexcept { state_ = {}; }

 private:
  static void increment(std::uint64_t& value) noexcept {
    if (value != std::numeric_limits<std::uint64_t>::max()) ++value;
  }

  ExecutionResourceSnapshot state_{};
};
/** Immutable backend/device contract owned by one prepared execution.
 *
 * This is intentionally smaller than core::ContextState: methods should not
 * depend on CUDA architecture discovery, AOT profile details, or API-owned
 * mutable error state merely to select an execution backend.
 */
class ExecutionContext {
 public:
  explicit ExecutionContext(const core::ContextState& state) noexcept
      : backend_(state.requested_backend), device_id_(state.device_id) {}

  [[nodiscard]] vibeqc_backend backend() const noexcept { return backend_; }
  [[nodiscard]] int device_id() const noexcept { return device_id_; }
  [[nodiscard]] bool cuda_requested() const noexcept { return backend_ == VIBEQC_BACKEND_CUDA; }

  void observe_numeric_peak(ExecutionMemorySpace space, std::size_t bytes) noexcept {
    resources_.observe_numeric_peak(space, bytes);
  }

  void observe_workspace_peak(ExecutionMemorySpace space, std::size_t bytes) noexcept {
    resources_.observe_workspace_peak(space, bytes);
  }

  [[nodiscard]] ExecutionResourceSnapshot resources() const noexcept {
    return resources_.snapshot();
  }

  void reset_resources() noexcept { resources_.reset(); }

 private:
  vibeqc_backend backend_{VIBEQC_BACKEND_CPU_REFERENCE};
  int device_id_{};
  ExecutionResourceTracker resources_{};
};

}  // namespace vibeqc::runtime
