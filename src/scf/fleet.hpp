#ifndef VIBEQC_SCF_FLEET_HPP
#define VIBEQC_SCF_FLEET_HPP

#include "core/types.hpp"
#include "scf/cuda_batch.hpp"
#include "scf/types.hpp"

#include <cstddef>
#include <optional>
#include <vector>

namespace vibeqc::scf {

struct CudaRhfBucketPlan;

struct FleetItemResult {
  vibeqc_status status{VIBEQC_STATUS_INTERNAL_ERROR};
  ScfResult scf;
  std::size_t bucket_id{};
  bool warm_start_used{};
  bool warm_start_fallback{};
  vibeqc_backend executed_backend{VIBEQC_BACKEND_CPU_REFERENCE};
};

/**
 * Persistent ragged-system execution plan.
 *
 * Systems retain independent dimensions and SCF state. The execution order is
 * bucketed by compatible workload, while result indexing always matches the
 * caller's original system order. A plan is intentionally not concurrently
 * re-entrant because warm-start state is mutable.
 */
class FleetPlan {
 public:
  FleetPlan(std::vector<core::System> systems,
            vibeqc_method method,
            ScfOptions options,
            bool warm_starts_enabled,
            bool cuda_fock_enabled,
            bool shell_class_profiling_enabled,
            int device_id);
  ~FleetPlan();

  [[nodiscard]] std::size_t size() const noexcept { return systems_.size(); }

  std::vector<FleetItemResult> execute(
      const std::vector<std::optional<std::vector<double>>>& coordinates);

  void clear_warm_starts();

  /**
   * Control whether successful executions replace the retained warm guesses.
   *
   * Disabling updates freezes the current per-system density snapshots. This
   * is useful for reproducible A/B measurements where every replay must start
   * from the same dm0; it does not enable warm starts or manufacture missing
   * snapshots.
   */
  void set_warm_start_updates(bool enabled) noexcept;

  /** Return the final-density profile from the most recent CUDA execution. */
  [[nodiscard]] const std::optional<CudaRhfShellClassProfile>&
  last_shell_class_profile() const noexcept {
    return last_shell_class_profile_;
  }

  /** Return PPPS queue statistics from the most recent profiled execution. */
  [[nodiscard]] const std::optional<CudaPppsQueueProfile>&
  last_ppps_queue_profile() const noexcept {
    return last_ppps_queue_profile_;
  }

  /** Return one setup-time eigensolver decision for every executed bucket. */
  [[nodiscard]] const std::vector<CudaEigensolverDiagnostic>&
  last_eigensolver_diagnostics() const noexcept {
    return last_eigensolver_diagnostics_;
  }

 private:
  std::vector<core::System> systems_;
  vibeqc_method method_{VIBEQC_METHOD_RHF};
  ScfOptions options_;
  bool warm_starts_enabled_{};
  bool warm_start_updates_enabled_{true};
  bool cuda_fock_enabled_{};
  bool shell_class_profiling_enabled_{};
  int device_id_{};
  std::vector<std::size_t> execution_order_;
  std::vector<std::size_t> bucket_ids_;
  std::vector<std::optional<std::vector<double>>> warm_densities_;
  std::optional<CudaRhfShellClassProfile> last_shell_class_profile_;
  std::optional<CudaPppsQueueProfile> last_ppps_queue_profile_;
  std::vector<CudaEigensolverDiagnostic> last_eigensolver_diagnostics_;
  // One allocation/Graph owner per workload bucket. Raw opaque pointers keep
  // CUDA headers out of this public C++ translation unit; the destructor owns
  // them through the backend-specific destroy function.
  std::vector<CudaRhfBucketPlan*> cuda_bucket_plans_;
};

}  // namespace vibeqc::scf

#endif
