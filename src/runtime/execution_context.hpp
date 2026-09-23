#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "core/types.hpp"

namespace vibeqc::runtime {

/** Host/device domains for method-neutral execution resource observations. */
enum class ExecutionMemorySpace : std::uint8_t { Host = 0, Device = 1 };

/** Method-neutral resource classes retained separately for diagnostics.
 *
 * These are observations, not admission budgets. The observation count is
 * authoritative for whether a value is known: a zero peak with zero
 * observations means unmeasured, not free. Numeric capacity and scratch are
 * kept compatible with the historical scalar fields while the additional
 * classes make pinned, provider-retained, staging, and capture-retained state
 * explicit without forcing methods to share an allocator.
 */
enum class ExecutionResourceKind : std::uint8_t {
  Numeric = 0,
  Scratch,
  Pinned,
  ProviderRetained,
  Staging,
  CaptureRetained,
  Count,
};

struct ExecutionResourceObservation {
  std::size_t host_peak_bytes{};
  std::size_t device_peak_bytes{};
  std::uint64_t host_observations{};
  std::uint64_t device_observations{};
};

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
  std::array<ExecutionResourceObservation, static_cast<std::size_t>(ExecutionResourceKind::Count)>
      by_kind{};

  [[nodiscard]] const ExecutionResourceObservation& observation(ExecutionResourceKind kind) const
      noexcept {
    return by_kind[static_cast<std::size_t>(kind)];
  }
};

class ExecutionResourceTracker {
 public:
  void observe_peak(ExecutionResourceKind kind, ExecutionMemorySpace space,
                    std::size_t bytes) noexcept {
    auto& observation = state_.by_kind[static_cast<std::size_t>(kind)];
    auto& peak = space == ExecutionMemorySpace::Host ? observation.host_peak_bytes
                                                     : observation.device_peak_bytes;
    auto& observations = space == ExecutionMemorySpace::Host ? observation.host_observations
                                                             : observation.device_observations;
    peak = std::max(peak, bytes);
    increment(observations);

    if (kind == ExecutionResourceKind::Numeric) {
      auto& legacy_peak = space == ExecutionMemorySpace::Host ? state_.host_numeric_peak_bytes
                                                              : state_.device_numeric_peak_bytes;
      legacy_peak = std::max(legacy_peak, bytes);
      increment(state_.numeric_observations);
    } else if (kind == ExecutionResourceKind::Scratch) {
      auto& legacy_peak = space == ExecutionMemorySpace::Host ? state_.host_workspace_peak_bytes
                                                              : state_.device_workspace_peak_bytes;
      legacy_peak = std::max(legacy_peak, bytes);
      increment(state_.workspace_observations);
    }
  }

  void observe_numeric_peak(ExecutionMemorySpace space, std::size_t bytes) noexcept {
    observe_peak(ExecutionResourceKind::Numeric, space, bytes);
  }

  void observe_workspace_peak(ExecutionMemorySpace space, std::size_t bytes) noexcept {
    observe_peak(ExecutionResourceKind::Scratch, space, bytes);
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

  void observe_resource_peak(ExecutionResourceKind kind, ExecutionMemorySpace space,
                             std::size_t bytes) noexcept {
    resources_.observe_peak(kind, space, bytes);
  }

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
