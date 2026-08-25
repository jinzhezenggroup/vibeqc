#ifndef QCE_SCF_FLEET_HPP
#define QCE_SCF_FLEET_HPP

#include "core/types.hpp"
#include "scf/rhf.hpp"

#include <cstddef>
#include <optional>
#include <vector>

namespace qce::scf {

struct CudaRhfBucketPlan;

struct FleetItemResult {
  qce_status status{QCE_STATUS_INTERNAL_ERROR};
  core::ScfResult scf;
  std::size_t bucket_id{};
  bool warm_start_used{};
  bool warm_start_fallback{};
  qce_backend executed_backend{QCE_BACKEND_CPU_REFERENCE};
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
            qce_method method,
            core::ScfOptions options,
            bool warm_starts_enabled,
            bool cuda_fock_enabled,
            bool shell_class_profiling_enabled,
            int device_id);
  ~FleetPlan();

  [[nodiscard]] std::size_t size() const noexcept { return systems_.size(); }

  std::vector<FleetItemResult> execute(
      const std::vector<std::optional<std::vector<double>>>& coordinates);

  void clear_warm_starts();

  /** Return the final-density profile from the most recent CUDA execution. */
  [[nodiscard]] const std::optional<CudaRhfShellClassProfile>&
  last_shell_class_profile() const noexcept {
    return last_shell_class_profile_;
  }

 private:
  std::vector<core::System> systems_;
  qce_method method_{QCE_METHOD_RHF};
  core::ScfOptions options_;
  bool warm_starts_enabled_{};
  bool cuda_fock_enabled_{};
  bool shell_class_profiling_enabled_{};
  int device_id_{};
  std::vector<std::size_t> execution_order_;
  std::vector<std::size_t> bucket_ids_;
  std::vector<std::optional<std::vector<double>>> warm_densities_;
  std::optional<CudaRhfShellClassProfile> last_shell_class_profile_;
  // One allocation/Graph owner per workload bucket. Raw opaque pointers keep
  // CUDA headers out of this public C++ translation unit; the destructor owns
  // them through the backend-specific destroy function.
  std::vector<CudaRhfBucketPlan*> cuda_bucket_plans_;
};

}  // namespace qce::scf

#endif
