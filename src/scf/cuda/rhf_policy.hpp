#ifndef VIBEQC_SCF_CUDA_RHF_POLICY_HPP
#define VIBEQC_SCF_CUDA_RHF_POLICY_HPP

#include <cstddef>
#include <cstdint>
#include <optional>

#include "runtime/cuda_provider.hpp"
#include "runtime/cuda_target_info.hpp"
#include "vibeqc/vibeqc.h"

namespace vibeqc::scf::cuda_policy {

/**
 * Explicit Direct-J/K tuning evidence.
 *
 * These defaults preserve the previously qualified production choices, but
 * they are profile inputs rather than CUDA semantics. Autotuning/profile
 * selection can provide another compatible record without changing kernels.
 */
/**
 * Analytical small-HF work model.
 *
 * Device calibration supplies only primitive launch/rate constants. Work
 * volume itself is derived exactly from the workload, so changing the GPU does
 * not require re-benchmarking every AO size or molecule.
 */
struct SmallHfMatrixCalibration {
  double native_launch_nanoseconds{};
  double native_fp64_flops_per_nanosecond{};
  double native_bytes_per_nanosecond{};
  double cublas_launch_nanoseconds{};
  double cublas_fp64_flops_per_nanosecond{};
  double cublas_bytes_per_nanosecond{};
};

struct SmallHfProfitabilityProfile {
  // Compatibility evidence used only when no complete device calibration is
  // available. These values are not CUDA semantics.
  std::size_t fallback_persistent_eri_ao_limit{16};
  std::size_t fallback_cublas_matrix_product_ao_threshold{17};
  SmallHfMatrixCalibration matrix{};
};

struct SmallHfWorkload {
  std::size_t nbf{};
  /** Number of square matrices submitted to the most demanding product call. */
  std::size_t matrix_count{};
  /** Physical systems whose complete AO ERI tensors would be cached. */
  std::size_t system_count{};
  /** RHF/UHF Fock states contracting the cached AO ERI tensor. */
  std::size_t fock_state_count{};
};

struct SmallHfAnalyticEstimate {
  std::uint64_t matrix_flops{};
  std::uint64_t native_matrix_semantic_bytes{};
  std::uint64_t cublas_matrix_semantic_bytes{};
  std::uint64_t native_matrix_blocks{};
  std::uint64_t native_matrix_waves{};
  std::uint64_t eri_elements{};
  std::uint64_t eri_bytes{};
  std::uint64_t cached_fock_ao_quartets{};
};

struct SmallHfProfitabilityPolicy {
  SmallHfAnalyticEstimate estimate{};
  bool use_cublas{};
  bool persistent_eri{};
  bool cublas_from_calibration{};
  double native_matrix_nanoseconds{};
  double cublas_matrix_nanoseconds{};
};

SmallHfAnalyticEstimate estimate_small_hf_workload(const runtime::CudaTargetInfo& target,
                                                   const SmallHfWorkload& workload) noexcept;

SmallHfProfitabilityPolicy resolve_small_hf_profitability(
    const runtime::CudaTargetInfo& target, const SmallHfWorkload& workload,
    SmallHfProfitabilityProfile profile = SmallHfProfitabilityProfile{}) noexcept;

struct DirectJkFixedTopologyTaskProfile {
  std::size_t maximum_arena_bytes{std::size_t{1} << 30};
};

struct DirectJkBoundedStreamingTaskProfile {
  std::size_t maximum_task_capacity{8U * 1024U * 1024U};
  // Keep the qualified 1.5-GiB page cap; task capacity follows the current ABI.
  std::size_t maximum_arena_bytes{std::size_t{3} << 29};
};

struct DirectJkTuningProfile {
  DirectJkFixedTopologyTaskProfile fixed_topology{};
  DirectJkBoundedStreamingTaskProfile bounded_streaming{};
  std::size_t cuda_stack_limit_bytes{std::size_t{64} << 10};
  unsigned maximum_persistent_quartet_warps_per_sm{8};
};

struct DirectJkFixedTopologyTaskPolicy {
  std::size_t arena_maximum_bytes{};
};

struct DirectJkBoundedStreamingTaskPolicy {
  std::size_t task_capacity_ceiling{};
  std::size_t arena_maximum_bytes{};
};

/** Resource-legal Direct-J/K schedule resolved for one runtime target. */
struct DirectJkSchedulePolicy {
  DirectJkFixedTopologyTaskPolicy fixed_topology{};
  DirectJkBoundedStreamingTaskPolicy bounded_streaming{};
  std::size_t cuda_stack_limit_bytes{};
  unsigned persistent_quartet_warps_per_sm{};
};

DirectJkSchedulePolicy resolve_direct_jk_schedule_policy(
    const runtime::CudaTargetInfo& target,
    DirectJkTuningProfile profile = DirectJkTuningProfile{}) noexcept;

std::size_t direct_jk_bounded_streaming_task_capacity_limit(
    const DirectJkSchedulePolicy& policy, std::size_t generated_task_bytes) noexcept;

/**
 * IEEE-754 binary32 unit roundoff (2^-24). Published from the shared header so
 * the host admission and the CUDA tile gate resolve the identical cutoff.
 */
inline constexpr double kMixedPrecisionFloat32UnitRoundoff = 5.9604644775390625e-08;

/** Runtime policy switches kept in a host-only translation unit. */
bool reuse_converged_fock_requested() noexcept;
std::optional<double> configured_mixed_precision_fock_threshold(
    double screening_tolerance) noexcept;
/**
 * Outcome of the budget-aware admission for the public \p auto policy.
 *
 * \p eligible_tiles is the mixed-capable tile census of the exact prepared
 * topology: the active Fock tiles in the high-angular-order shell classes that
 * may run in FP32. A zero census is never admitted because the accumulated
 * bound cannot be evaluated without it.
 */
struct AutoMixedPrecisionAdmission {
  /** The mixed iterative Fock may run for this reference. */
  bool admitted{false};
  /** Resolved contribution cutoff for FP32 tiles; zero when refused. */
  double threshold{0.0};
  /** Error the policy reserved for the iterative operator out of the target. */
  double reserved_error{0.0};
  /** Mixed-capable tile census the accumulated bound was evaluated against. */
  double eligible_tiles{0.0};
};
/**
 * Budget-aware admission for the public \p auto policy.
 *
 * One global contribution cutoff is not an error budget: any number of
 * individually eligible FP32 tiles can accumulate. This evaluates the
 * worst-case accumulated bound `eps32 * cutoff * eligible_tiles` against the
 * error the policy reserves for the iterative operator out of the requested
 * energy tolerance and resolves the largest cutoff that bound certifies. A
 * cutoff at or below the screening floor admits no mixed tile, so the operator
 * stays FP64. Because the census is the exact per-shell-class active tile count
 * of this reference, a larger or lower-symmetry tile census tightens the cutoff
 * instead of silently accumulating error.
 */
AutoMixedPrecisionAdmission admit_auto_mixed_precision_fock(double energy_tolerance,
                                                            double screening_tolerance,
                                                            double eligible_tiles) noexcept;
/** Complete resolution of the requested precision policy, including its audit. */
struct MixedPrecisionFockPolicy {
  /** Resolved FP32 tile cutoff for the whole batch; empty keeps the FP64 path. */
  std::optional<double> threshold;
  /** The cutoff came from the certified accumulated-error budget. */
  bool budget_certified{false};
  /** Error reserved for the iterative operator; zero when uncertified. */
  double reserved_error{0.0};
  /** Mixed-capable tile census ceiling the budget was evaluated against. */
  double eligible_tiles{0.0};
  /**
   * Cutoff ceiling for one item. For \p auto this is the tolerance anchor that
   * the item's own budget may tighten; for an explicit diagnostic cutoff it is
   * the diagnostic value itself.
   */
  double item_cutoff_ceiling{0.0};
  /**
   * Error each item's own census divides. Zero means the cutoff does not depend
   * on a per-item census (explicit diagnostic override), so every item shares
   * the ceiling.
   */
  double item_budget_error{0.0};
};
/** Per-item admission for one system of a prepared batch. */
struct MixedPrecisionItemPolicy {
  /** The item may run the mixed iterative operator. */
  bool admitted{false};
  /** Item contribution cutoff the device applies; zero keeps the item FP64. */
  double threshold{0.0};
  /** Mixed-capable tile census the cutoff was resolved from (1 if census-free). */
  std::uint32_t census{0};
};
/**
 * Resolve one item's precision policy from the resolved batch policy.
 *
 * A budget-resolved \p auto policy is per item: the accumulated bound uses the
 * item's own mixed-capable tile census, and the item is only admitted when its
 * own starting state is a validated warm density, because the reserved budget
 * bounds the perturbation of a known state rather than of a cold guess. An item
 * that cannot be certified keeps the exact FP64 operator without affecting the
 * other items of the batch. A census-free diagnostic cutoff stays item
 * agnostic, so the legacy switch behaves exactly as before.
 */
MixedPrecisionItemPolicy resolve_mixed_precision_item(const MixedPrecisionFockPolicy& policy,
                                                      bool validated_warm_state,
                                                      std::size_t item_tile_census,
                                                      double screening_tolerance) noexcept;
/**
 * Resolve the mixed-precision Fock policy from the requested public policy and
 * the tolerances. \p nullopt preserves the legacy
 * VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD diagnostic switch verbatim. An explicit
 * \p VIBEQC_PRECISION_FP64 keeps the pure double path (the environment cannot
 * relax it). An explicit \p VIBEQC_PRECISION_AUTO derives the threshold from the
 * budget-aware admission above, with an explicit numeric environment value
 * acting as a hard diagnostic override that bypasses the budget.
 */
MixedPrecisionFockPolicy resolve_mixed_precision_fock_policy(
    std::optional<vibeqc_precision_mode> precision_mode, double energy_tolerance,
    double screening_tolerance, double eligible_tiles) noexcept;
bool graph_native_eigensolver_override_requested() noexcept;
bool xsyev_probe_skip_diagnostic_requested() noexcept;
bool bounded_direct_streaming_override_requested() noexcept;
bool bounded_direct_count_diagnostic_requested() noexcept;
bool bounded_direct_aot_only_diagnostic_requested() noexcept;
bool bounded_direct_fock_only_diagnostic_requested() noexcept;
bool bounded_fock_class_timing_requested() noexcept;
/** True when force AOT classes are explicitly narrowed for a diagnostic replay. */
bool aot_shell_class_selection_override_requested() noexcept;
/**
 * Direct-tile validation is a structural diagnostic, never a numerical endpoint.
 *
 * When requested, CUDA execution may build and validate the compacted descriptor
 * queue, but it must stop before reporting SCF energy/force results.  The explicit
 * non-success endpoint status prevents diagnostic output from being mistaken for
 * scientific correctness evidence.
 */
struct DirectTileValidationPolicy {
  bool requested{};
  bool produces_numerical_endpoint{true};
  vibeqc_status endpoint_status{VIBEQC_STATUS_SUCCESS};
};
DirectTileValidationPolicy resolve_direct_tile_validation_policy() noexcept;
bool direct_tile_validation_requested() noexcept;
double converged_fock_reuse_density_rms(double density_tolerance) noexcept;
bool force_density_product_screening_requested() noexcept;
bool resident_ppps_bra_requested() noexcept;
bool ppps_signature_bucketing_requested() noexcept;
bool psps_signature_bucketing_requested() noexcept;
bool ppss_signature_bucketing_requested() noexcept;
unsigned ppps_resident_block_threads_requested() noexcept;
struct OneElectronValuePolicy {
  /** 0: one AO pair per thread; 1: one shell pair per warp. */
  unsigned mapping{};
  /** An explicit diagnostic selector was supplied. */
  bool diagnostic_override{};
  /** The selector requested a schedule the active provider does not advertise. */
  bool capability_fallback{};
};
/** Resolve one-electron scheduling from an explicit provider capability contract. */
OneElectronValuePolicy resolve_one_electron_value_policy(
    const runtime::CudaProviderCapabilities& provider) noexcept;
/** Mapping selected for the active configured CUDA provider. */
unsigned one_electron_value_mapping_requested() noexcept;
/** Generated derivatives default; reference/none/0 selects the retained native exception. */
bool generated_one_electron_derivatives_requested() noexcept;
/** 0: AO threads; 1: shell-pair/component warp lanes;
 * 2: deterministic serial diagnostics; 3: AO-pair warp with nucleus lanes (default). */
unsigned one_electron_derivative_mapping_requested() noexcept;
bool resident_psss_bra_requested() noexcept;

/** 0: cooperative dense elements; 1: deterministic serial traversal. */
unsigned df_derivative_mapping_requested() noexcept;
/** 0: contiguous auxiliary outputs; 1: AO components; 2: primitive lanes. */
unsigned df_value_mapping_requested() noexcept;

/** Parse diagnostic value math before constructing an immutable source owner. */
bool df_value_math_requested(unsigned& math) noexcept;
/** Sentinel for the generated class-filtered schedule, rather than a lane count. */
inline constexpr unsigned kDfCandidateRawSchedule = 0;
/** Raw export uses 1/4/32 lanes or kDfCandidateRawSchedule; auto stays scalar. */
bool df_value_raw_lanes_requested(unsigned& lanes) noexcept;

}  // namespace vibeqc::scf::cuda_policy

#endif
