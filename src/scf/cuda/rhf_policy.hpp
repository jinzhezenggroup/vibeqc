#ifndef VIBEQC_SCF_CUDA_RHF_POLICY_HPP
#define VIBEQC_SCF_CUDA_RHF_POLICY_HPP

#include <cstddef>
#include <cstdint>
#include <optional>

#include "vibeqc/vibeqc.h"

namespace vibeqc::scf::cuda_policy {

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
bool direct_tile_validation_requested() noexcept;
double converged_fock_reuse_density_rms(double density_tolerance) noexcept;
bool force_density_product_screening_requested() noexcept;
bool resident_ppps_bra_requested() noexcept;
bool ppps_signature_bucketing_requested() noexcept;
bool psps_signature_bucketing_requested() noexcept;
bool ppss_signature_bucketing_requested() noexcept;
unsigned ppps_resident_block_threads_requested() noexcept;
bool one_electron_force_scalar_requested() noexcept;
/** 0: one AO pair per thread; 1: one shell pair per warp. */
unsigned one_electron_value_mapping_requested() noexcept;
/** Generated derivative candidates are opt-in and read at each force execution. */
bool generated_one_electron_derivatives_requested() noexcept;
/** 0: AO threads; 1: shell-pair warp lanes; 2: deterministic serial diagnostics. */
unsigned one_electron_derivative_mapping_requested() noexcept;
bool resident_psss_bra_requested() noexcept;
/** Generated weighted primitive candidate; frozen into a prepared bucket. */
bool generated_psss_weighted_requested() noexcept;

/** 0: cooperative dense elements; 1: deterministic serial traversal. */
unsigned df_derivative_mapping_requested() noexcept;
/** 0: contiguous auxiliary outputs; 1: AO components; 2: primitive lanes. */
unsigned df_value_mapping_requested() noexcept;

}  // namespace vibeqc::scf::cuda_policy

#endif
