#ifndef VIBEQC_SCF_CUDA_EIGENSOLVER_POLICY_HPP
#define VIBEQC_SCF_CUDA_EIGENSOLVER_POLICY_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace vibeqc::scf {

/** CUDA 12.9 documented dimension limit for generic XsyevBatched. */
inline constexpr std::uint64_t kXsyevBatchedDocumentedDimensionLimit = 32768;

/** Why an FP64 XsyevBatched signature is or is not API-eligible. */
enum class XsyevBatchedEligibilityReason : std::uint32_t {
  eligible = 0,
  zero_dimension,
  invalid_leading_dimension,
  documented_dimension_limit,
  solver_batch_limit,
  documented_product_limit,
};

struct XsyevBatchedEligibility {
  bool eligible{};
  XsyevBatchedEligibilityReason reason{XsyevBatchedEligibilityReason::zero_dimension};
  std::uint64_t matrix_batch_product{};
};

/**
 * Check the documented CUDA 12.9 API limits without allocating matrices.
 *
 * `solver_batch` is the batch submitted to cuSOLVER, not merely the physical
 * system count: RHF uses one matrix per system while UHF uses two spin states.
 * Division checks precede every multiplication so boundary tests can exercise
 * signatures near INT32_MAX without overflowing an intermediate.
 */
constexpr XsyevBatchedEligibility xsyev_batched_api_eligibility(
    std::uint64_t n, std::uint64_t lda, std::uint64_t solver_batch) noexcept {
  if (n == 0 || solver_batch == 0) {
    return {false, XsyevBatchedEligibilityReason::zero_dimension, 0};
  }
  if (lda < n) {
    return {false, XsyevBatchedEligibilityReason::invalid_leading_dimension, 0};
  }
  if (n > kXsyevBatchedDocumentedDimensionLimit) {
    return {false, XsyevBatchedEligibilityReason::documented_dimension_limit, 0};
  }
  constexpr std::uint64_t product_limit =
      static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
  if (solver_batch > product_limit) {
    return {false, XsyevBatchedEligibilityReason::solver_batch_limit, 0};
  }
  if (n > product_limit / lda) {
    return {false, XsyevBatchedEligibilityReason::documented_product_limit, 0};
  }
  const std::uint64_t matrix_elements = n * lda;
  if (solver_batch > product_limit / matrix_elements) {
    return {false, XsyevBatchedEligibilityReason::documented_product_limit, 0};
  }
  return {true, XsyevBatchedEligibilityReason::eligible, matrix_elements * solver_batch};
}

/** Stage that prevented exact device-launch Graph qualification. */
enum class XsyevBatchedGraphProbeStage : std::uint32_t {
  none = 0,
  api_eligibility,
  select_device,
  device_identity,
  create_stream,
  create_solver,
  create_parameters,
  allocate_probe_data,
  query_workspace,
  insufficient_device_memory,
  allocate_workspace,
  ordinary_execution,
  ordinary_validation,
  begin_capture,
  capture_provider,
  end_capture,
  instantiate_device_launch_graph,
  upload_graph,
  host_graph_replay,
  host_graph_validation,
  device_tail_replay,
  device_tail_validation,
};

/** Exact-stack qualification evidence for one FP64 solver signature. */
struct XsyevBatchedGraphProbeResult {
  XsyevBatchedEligibility api;
  XsyevBatchedGraphProbeStage failure_stage{XsyevBatchedGraphProbeStage::none};
  std::uint64_t n{};
  std::uint64_t solver_batch{};
  std::size_t device_workspace_bytes{};
  std::size_t host_workspace_bytes{};
  std::size_t available_device_bytes{};
  int device_id{-1};
  std::array<std::uint8_t, 16> device_uuid{};
  std::array<char, 256> device_name{};
  int compute_capability_major{};
  int compute_capability_minor{};
  int cuda_runtime_version{};
  int cuda_driver_version{};
  int cusolver_version{};
  int cuda_error{};
  int cusolver_error{};
  bool ordinary_execution_passed{};
  bool graph_capture_passed{};
  bool host_graph_replay_passed{};
  bool device_tail_replay_passed{};
  bool graph_eligible{};
  double maximum_eigenvalue_error{};
  double maximum_residual{};
  double maximum_orthogonality_error{};
};

struct XsyevBatchedDispatch {
  bool ordinary_stream_provider{};
  bool device_launch_graph_provider{};
};

/** Keep ordinary cuSOLVER use independent from its stronger Graph contract. */
constexpr XsyevBatchedDispatch select_xsyev_batched_dispatch(
    const XsyevBatchedGraphProbeResult& probe) noexcept {
  return {probe.ordinary_execution_passed, probe.ordinary_execution_passed && probe.graph_eligible};
}

/** Probe one exact device/toolkit/dimension/batch signature during setup. */
XsyevBatchedGraphProbeResult probe_xsyev_batched_device_launch_graph(
    int device_id, std::uint64_t n, std::uint64_t solver_batch) noexcept;

}  // namespace vibeqc::scf

#endif
