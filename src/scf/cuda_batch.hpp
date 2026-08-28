#ifndef VIBEQC_SCF_CUDA_BATCH_HPP
#define VIBEQC_SCF_CUDA_BATCH_HPP

#include "core/types.hpp"
#include "scf/direct_task_layout.hpp"
#include "scf/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace vibeqc::scf {

struct RhfBucketItem {
  vibeqc_status status{VIBEQC_STATUS_INTERNAL_ERROR};
  ScfResult scf;
};

struct CudaRhfBucketPlan;

/** Final-density work retained by CUDA direct screening for one shell class. */
struct CudaRhfShellClassProfileEntry {
  std::uint64_t shell_quartets{};
  std::uint64_t tiles{};
  std::uint64_t ao_quartets{};
  std::uint64_t primitive_quartets{};
};

using CudaRhfShellClassProfile =
    std::array<CudaRhfShellClassProfileEntry,
               detail::kDirectQuartetShellClassCount>;

/** Scalar CTA sizes compared by the PPPS production-queue diagnostic. */
inline constexpr std::array<unsigned, 4> kPppsProfileBlockThreads{
    32U, 64U, 128U, 256U};

/**
 * Exact final-density PPPS queue statistics retained by the CUDA backend.
 *
 * Primitive-pair histograms use buckets 0..63 verbatim and bucket 64 for all
 * larger values.  The queue builder currently admits at most 64 bra pairs,
 * while the overflow bucket keeps this diagnostic safe for arbitrary custom
 * ket contractions.  Denominators are retained instead of rounded ratios so
 * heterogeneous fleet buckets can be aggregated without losing precision.
 */
struct CudaPppsQueueProfile {
  static constexpr std::size_t kOrientationCount = 2;
  static constexpr std::size_t kPrimitivePairBucketCount = 65;

  std::uint64_t descriptor_slots{};
  std::uint64_t non_empty_descriptors{};
  std::uint64_t tasks{};
  std::uint64_t primitive_work{};
  std::vector<std::uint64_t> ket_count_histogram;
  std::array<std::uint64_t, kPppsProfileBlockThreads.size()> lane_slots{};
  std::uint64_t primitive_warp_slots{};
  std::array<std::uint64_t, kOrientationCount> orientation_tasks{};
  std::array<std::uint64_t, kOrientationCount> orientation_primitive_work{};
  std::array<std::uint64_t, kPrimitivePairBucketCount> bra_primitive_tasks{};
  std::array<std::uint64_t, kPrimitivePairBucketCount>
      bra_primitive_work{};
  std::array<std::uint64_t, kPrimitivePairBucketCount> ket_primitive_tasks{};
  std::array<std::uint64_t, kPrimitivePairBucketCount>
      ket_primitive_work{};
  // Each bucket launch is sequential. Summing its ideal load and simulated
  // makespan therefore produces an aggregate tail estimate for a ragged fleet.
  std::array<double, kPppsProfileBlockThreads.size()> task_schedule_ideal{};
  std::array<double, kPppsProfileBlockThreads.size()> task_schedule_makespan{};
  std::array<double, kPppsProfileBlockThreads.size()>
      primitive_schedule_ideal{};
  std::array<double, kPppsProfileBlockThreads.size()>
      primitive_schedule_makespan{};
};

/** Internal diagnostics for the immutable CUDA basis topology layout. */
struct CudaRhfBasisLayoutStats {
  std::size_t system_count{};
  std::size_t shell_count{};
  std::size_t shell_pair_count{};
  std::size_t shell_quartet_count{};
  std::size_t ao_count{};
  std::size_t unique_primitive_count{};
  std::size_t expanded_primitive_references{};
  std::size_t device_basis_bytes{};
};

CudaRhfBasisLayoutStats inspect_rhf_cuda_basis_layout(
    const std::vector<core::System>& systems);

std::vector<RhfBucketItem> run_rhf_cuda_bucket(
    const std::vector<core::System>& systems,
    const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities,
    int device_id,
    bool shell_class_profiling = false);

std::vector<RhfBucketItem> run_rhf_cuda_bucket_cached(
    CudaRhfBucketPlan** plan,
    const std::vector<core::System>& systems,
    const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities,
    int device_id,
    bool shell_class_profiling = false);

std::vector<RhfBucketItem> run_uhf_cuda_bucket(
    const std::vector<core::System>& systems,
    const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities,
    int device_id,
    bool shell_class_profiling = false);

std::vector<RhfBucketItem> run_uhf_cuda_bucket_cached(
    CudaRhfBucketPlan** plan,
    const std::vector<core::System>& systems,
    const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities,
    int device_id,
    bool shell_class_profiling = false);

bool get_rhf_cuda_shell_class_profile(
    const CudaRhfBucketPlan* plan,
    CudaRhfShellClassProfile& profile) noexcept;

bool get_rhf_cuda_ppps_queue_profile(
    const CudaRhfBucketPlan* plan,
    CudaPppsQueueProfile& profile) noexcept;

void destroy_rhf_cuda_bucket_plan(CudaRhfBucketPlan* plan) noexcept;

/**
 * Control whether a cached CUDA bucket may advance its fixed warm-start seed.
 *
 * A true-to-false transition snapshots the currently resident converged
 * density together with its previous-energy seed. Replays of that exact dm0
 * can then restore the same one-iteration convergence baseline even after the
 * device-resident density has advanced. Re-enabling updates discards the
 * frozen baseline without invalidating the current resident-density cache.
 */
void set_rhf_cuda_bucket_warm_start_updates(
    CudaRhfBucketPlan* plan, bool enabled) noexcept;

/** Discard both resident and frozen warm-start state for a CUDA bucket. */
void clear_rhf_cuda_bucket_warm_starts(CudaRhfBucketPlan* plan) noexcept;

}  // namespace vibeqc::scf

#endif
