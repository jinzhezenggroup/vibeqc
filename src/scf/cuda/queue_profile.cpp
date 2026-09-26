#include <algorithm>
#include <array>
#include <numeric>

#include "scf/cuda/topology.hpp"

namespace vibeqc::scf::cuda_execution {

/** Host-only queue diagnostics. Simulated CTA assignment describes descriptor tails, not measured
 * occupancy or execution time. */
/** Simulate hardware CTA assignment with one descriptor at a time per SM. */
double ppps_profile_schedule_makespan(const std::vector<double>& weights,
                                      unsigned multiprocessor_count) {
  if (weights.empty() || multiprocessor_count == 0U) return 0.0;
  std::vector<double> loads(multiprocessor_count, 0.0);
  for (const double weight : weights) {
    auto next = std::min_element(loads.begin(), loads.end());
    *next += weight;
  }
  return *std::max_element(loads.begin(), loads.end());
}

/**
 * Summarize the exact compacted PPPS queue copied from the device.
 *
 * Signatures retain device materialization order, so the warp-divergence
 * denominator measures the queue that the production kernel actually saw.
 * The scheduling model intentionally stays descriptor-only: it estimates the
 * fixed-bra tail across physical SMs without claiming to reproduce occupancy
 * or instruction-level latency.
 */
CudaPppsQueueProfile build_ppps_queue_profile(const HostBatch& host,
                                              const std::vector<std::uint32_t>& descriptor_counts,
                                              const std::vector<std::uint32_t>& ordered_signatures,
                                              unsigned multiprocessor_count) {
  CudaPppsQueueProfile profile;
  profile.descriptor_slots = descriptor_counts.size();
  std::array<std::vector<double>, kPppsProfileBlockThreads.size()> task_schedule_weights;
  std::array<std::vector<double>, kPppsProfileBlockThreads.size()> primitive_schedule_weights;
  std::size_t ket_begin = 0;
  constexpr std::uint32_t kCountMask = 0x7fffffffU;
  constexpr std::uint32_t kOrientationMask = 0x80000000U;

  for (std::size_t bra_pair = 0; bra_pair < descriptor_counts.size(); ++bra_pair) {
    const std::size_t ket_count = descriptor_counts[bra_pair];
    if (ket_count == 0U) continue;
    if (ket_begin > ordered_signatures.size() ||
        ket_count > ordered_signatures.size() - ket_begin) {
      // A truncated diagnostic must never be mistaken for valid queue data.
      return {};
    }
    ++profile.non_empty_descriptors;
    profile.tasks += ket_count;
    if (profile.ket_count_histogram.size() <= ket_count) {
      profile.ket_count_histogram.resize(ket_count + 1U, 0U);
    }
    ++profile.ket_count_histogram[ket_count];

    const std::int64_t bra_begin = host.shell_pair_primitive_offsets[bra_pair];
    const std::int64_t bra_end = host.shell_pair_primitive_offsets[bra_pair + 1U];
    const std::uint64_t bra_primitives =
        bra_end > bra_begin ? static_cast<std::uint64_t>(bra_end - bra_begin) : 0U;
    const std::size_t bra_bucket = std::min<std::uint64_t>(
        bra_primitives, CudaPppsQueueProfile::kPrimitivePairBucketCount - 1U);
    std::vector<std::uint64_t> primitive_counts(ket_count, 0U);

    for (std::size_t local_ket = 0; local_ket < ket_count; ++local_ket) {
      const std::uint32_t signature = ordered_signatures[ket_begin + local_ket];
      const std::size_t orientation = (signature & kOrientationMask) == 0U ? 0U : 1U;
      const std::uint64_t ket_primitives = signature & kCountMask;
      const std::uint64_t primitive_work = bra_primitives * ket_primitives;
      primitive_counts[local_ket] = primitive_work;
      profile.primitive_work += primitive_work;
      ++profile.orientation_tasks[orientation];
      profile.orientation_primitive_work[orientation] += primitive_work;
      ++profile.bra_primitive_tasks[bra_bucket];
      profile.bra_primitive_work[bra_bucket] += primitive_work;
      const std::size_t ket_bucket = std::min<std::uint64_t>(
          ket_primitives, CudaPppsQueueProfile::kPrimitivePairBucketCount - 1U);
      ++profile.ket_primitive_tasks[ket_bucket];
      profile.ket_primitive_work[ket_bucket] += primitive_work;
    }

    for (std::size_t warp_begin = 0; warp_begin < ket_count; warp_begin += 32U) {
      const std::size_t warp_end = std::min(ket_count, warp_begin + 32U);
      const std::uint64_t maximum =
          *std::max_element(primitive_counts.begin() + static_cast<std::ptrdiff_t>(warp_begin),
                            primitive_counts.begin() + static_cast<std::ptrdiff_t>(warp_end));
      profile.primitive_warp_slots += 32U * maximum;
    }

    for (std::size_t candidate = 0; candidate < kPppsProfileBlockThreads.size(); ++candidate) {
      const std::size_t block_threads = kPppsProfileBlockThreads[candidate];
      const std::size_t rounds = (ket_count + block_threads - 1U) / block_threads;
      profile.lane_slots[candidate] += block_threads * rounds;
      task_schedule_weights[candidate].push_back(static_cast<double>(rounds));
      std::uint64_t descriptor_primitive_time = 0U;
      for (std::size_t round_begin = 0; round_begin < ket_count; round_begin += block_threads) {
        const std::size_t round_end = std::min(ket_count, round_begin + block_threads);
        descriptor_primitive_time +=
            *std::max_element(primitive_counts.begin() + static_cast<std::ptrdiff_t>(round_begin),
                              primitive_counts.begin() + static_cast<std::ptrdiff_t>(round_end));
      }
      primitive_schedule_weights[candidate].push_back(
          static_cast<double>(descriptor_primitive_time));
    }
    ket_begin += ket_count;
  }

  for (std::size_t candidate = 0; candidate < kPppsProfileBlockThreads.size(); ++candidate) {
    const double task_total = std::accumulate(task_schedule_weights[candidate].begin(),
                                              task_schedule_weights[candidate].end(), 0.0);
    const double primitive_total =
        std::accumulate(primitive_schedule_weights[candidate].begin(),
                        primitive_schedule_weights[candidate].end(), 0.0);
    profile.task_schedule_ideal[candidate] = task_total / static_cast<double>(multiprocessor_count);
    profile.task_schedule_makespan[candidate] =
        ppps_profile_schedule_makespan(task_schedule_weights[candidate], multiprocessor_count);
    profile.primitive_schedule_ideal[candidate] =
        primitive_total / static_cast<double>(multiprocessor_count);
    profile.primitive_schedule_makespan[candidate] =
        ppps_profile_schedule_makespan(primitive_schedule_weights[candidate], multiprocessor_count);
  }
  return profile;
}

}  // namespace vibeqc::scf::cuda_execution
