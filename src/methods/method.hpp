#ifndef VIBEQC_METHODS_METHOD_HPP
#define VIBEQC_METHODS_METHOD_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "scf/cuda_density_fitting.hpp"
#include "vibeqc/vibeqc.h"

namespace vibeqc::methods {

/** Registry metadata used by the public capability query and method factory. */
struct Capabilities {
  vibeqc_method method{};
  vibeqc_method_family family{};
  vibeqc_property_flags supported_properties{};
  bool available{};
  bool supports_batch{};
};

/** Method-neutral convergence diagnostics published through the legacy ABI. */
struct Convergence {
  unsigned iterations{};
  double energy_change{};
  double residual_rms{};
  bool converged{};
};

/** Common calculation result. Method-specific retained state stays in the plan. */
struct Result {
  double energy{};
  std::vector<double> forces;
  Convergence convergence;
  vibeqc_backend executed_backend{VIBEQC_BACKEND_CPU_REFERENCE};
};

struct BatchItemResult {
  vibeqc_status status{VIBEQC_STATUS_INTERNAL_ERROR};
  Result calculation;
  std::size_t bucket_id{};
  bool warm_start_used{};
  bool warm_start_fallback{};
};

struct DirectShellClassProfileEntry {
  std::uint64_t shell_quartets{};
  std::uint64_t tiles{};
  std::uint64_t ao_quartets{};
  std::uint64_t primitive_quartets{};
};

/** Final-density statistics for the resident scalar PPPS force queue. */
struct DirectPppsQueueProfile {
  static constexpr std::size_t kBlockSizeCount = 4;
  static constexpr std::size_t kOrientationCount = 2;
  static constexpr std::size_t kPrimitivePairBucketCount = 65;

  std::uint64_t descriptor_slots{};
  std::uint64_t non_empty_descriptors{};
  std::uint64_t empty_descriptors{};
  std::uint64_t tasks{};
  std::uint64_t primitive_work{};
  std::uint32_t ket_count_min{};
  std::uint32_t ket_count_median{};
  std::uint32_t ket_count_p90{};
  std::uint32_t ket_count_p99{};
  std::uint32_t ket_count_max{};
  std::array<double, kBlockSizeCount> lane_efficiency{};
  double primitive_warp_efficiency{};
  std::array<double, kBlockSizeCount> task_tail_imbalance{};
  std::array<double, kBlockSizeCount> primitive_tail_imbalance{};
  std::array<std::uint64_t, kOrientationCount> orientation_tasks{};
  std::array<std::uint64_t, kOrientationCount> orientation_primitive_work{};
  std::array<std::uint64_t, kPrimitivePairBucketCount> bra_primitive_tasks{};
  std::array<std::uint64_t, kPrimitivePairBucketCount> bra_primitive_work{};
  std::array<std::uint64_t, kPrimitivePairBucketCount> ket_primitive_tasks{};
  std::array<std::uint64_t, kPrimitivePairBucketCount> ket_primitive_work{};
};

/** Method-neutral record of one CUDA bucket's setup-time eigensolver choice. */
struct EigensolverDiagnostic {
  std::uint32_t bucket_id{};
  std::uint32_t ordinary_family{};
  std::uint32_t graph_family{};
  std::uint32_t selection_source{};
  std::uint64_t matrix_dimension{};
  std::uint64_t physical_system_count{};
  std::uint64_t solver_batch_count{};
  bool api_eligible{};
  std::uint32_t api_reason{};
  std::uint64_t matrix_batch_product{};
  std::uint32_t probe_failure_stage{};
  std::uint64_t device_workspace_bytes{};
  std::uint64_t host_workspace_bytes{};
  std::uint64_t available_device_bytes{};
  std::int32_t device_id{-1};
  std::array<std::uint8_t, 16> device_uuid{};
  std::array<char, 256> device_name{};
  std::int32_t compute_capability_major{};
  std::int32_t compute_capability_minor{};
  std::int32_t cuda_runtime_version{};
  std::int32_t cuda_driver_version{};
  std::int32_t cusolver_version{};
  std::int32_t cuda_error{};
  std::int32_t cusolver_error{};
  bool ordinary_execution_passed{};
  bool graph_capture_passed{};
  bool host_graph_replay_passed{};
  bool device_tail_replay_passed{};
  bool graph_eligible{};
  double maximum_eigenvalue_error{};
  double maximum_residual{};
  double maximum_orthogonality_error{};
};

/** Method-neutral per-iteration inactive-eigensolver measurement. */
struct InactiveEigensolverProfileEntry {
  std::uint32_t bucket_id{};
  std::uint32_t iteration{};
  std::uint32_t family{};
  std::uint32_t physical_system_count{};
  std::uint32_t solver_batch_count{};
  std::uint32_t active_physical_count{};
  std::uint32_t active_solver_count{};
  std::uint64_t solver_elapsed_nanoseconds{};
  std::uint32_t inactive_input_nonfinite_count{};
  std::uint32_t inactive_submission_nonfinite_count{};
  std::uint32_t inactive_info_nonzero_count{};
  std::uint32_t inactive_touch_flags{};
  bool provider_invoked{};
};

using Coordinates = std::vector<std::optional<std::vector<double>>>;

/** Prepared single-system method execution, independent of the public C ABI. */
class PreparedCalculation {
 public:
  virtual ~PreparedCalculation() = default;
  [[nodiscard]] virtual std::size_t atom_count() const noexcept = 0;
  [[nodiscard]] virtual const Capabilities& capabilities() const noexcept = 0;
  virtual Result execute() = 0;
};

/** Prepared ragged execution. Method families choose their own batching policy. */
class PreparedBatch {
 public:
  virtual ~PreparedBatch() = default;
  [[nodiscard]] virtual std::size_t size() const noexcept = 0;
  virtual std::vector<BatchItemResult> execute(const Coordinates& coordinates) = 0;
  virtual void clear_warm_starts() = 0;
  virtual void set_warm_start_updates(bool enabled) = 0;
  [[nodiscard]] virtual std::optional<std::vector<DirectShellClassProfileEntry>>
  last_direct_shell_class_profile() const = 0;
  [[nodiscard]] virtual std::optional<DirectPppsQueueProfile> last_direct_ppps_queue_profile()
      const = 0;
  [[nodiscard]] virtual std::vector<EigensolverDiagnostic> last_eigensolver_diagnostics() const = 0;
  [[nodiscard]] virtual std::vector<scf::CudaDensityFittingMetricDiagnostic>
  last_density_fitting_metric_diagnostics() const = 0;
  [[nodiscard]] virtual std::vector<InactiveEigensolverProfileEntry>
  last_inactive_eigensolver_profile() const = 0;
};

/** Exception carrying an exact public status across the C++ method boundary. */
class MethodError final : public std::runtime_error {
 public:
  MethodError(vibeqc_status status, const std::string& message)
      : std::runtime_error(message), status_(status) {}

  [[nodiscard]] vibeqc_status status() const noexcept { return status_; }

 private:
  vibeqc_status status_;
};

[[nodiscard]] const Capabilities* find_capabilities(vibeqc_method method) noexcept;

std::unique_ptr<PreparedCalculation> prepare_calculation(
    core::ContextState& context, const core::System& system,
    const vibeqc_method_descriptor& descriptor);

std::unique_ptr<PreparedBatch> prepare_batch(core::ContextState& context,
                                             std::vector<core::System> systems,
                                             const vibeqc_method_descriptor& descriptor,
                                             vibeqc_batch_flags flags);

}  // namespace vibeqc::methods

#endif
