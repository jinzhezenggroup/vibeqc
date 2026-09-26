#include <cmath>

#include "scf/cuda/direct_bounded_pages.hpp"
#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_page_screening.cuh"
#include "scf/cuda/direct_screening.cuh"
#include "scf/cuda/direct_task_encoding.cuh"

namespace vibeqc::scf::cuda_execution {

/**
 * Materialize one page of an overflowed exact class for its generated kernel.
 *
 * Page membership is determined by the unscreened candidate ordinal so every
 * launch covers a disjoint, deterministic slice without a device-to-host
 * synchronization. Surviving tasks are compacted within that page and then
 * consumed by the exact generated shell-class kernel; this is scheduling,
 * not a generic integral-evaluation fallback.
 */
template <bool Unrestricted, DirectScreeningPurpose Purpose>
__global__ void compact_bounded_exact_class_force_wave_kernel(
    DeviceBatch batch, const GeneratedShellPairStream* topology_pointer, unsigned shell_class,
    unsigned high_pair_class, unsigned low_pair_class, double screening_tolerance,
    std::uint64_t page_begin, std::uint32_t page_capacity, std::uint32_t bra_ordinal_begin,
    std::uint32_t bra_ordinal_end, bool same_pair_class, GeneratedShellTask* tasks,
    std::uint32_t* task_count, std::uint32_t* bra_head, const std::uint32_t* overflow,
    bool force_execution, std::uint32_t* signature_counts, const std::uint32_t* signature_offsets) {
  __shared__ std::uint32_t bra_ordinal;
  if (shell_class >= detail::kDirectQuartetShellClassCount ||
      (!force_execution && overflow[shell_class] == 0U)) {
    return;
  }
  const GeneratedShellPairStream& topology = *topology_pointer;
  const std::size_t stride = static_cast<std::size_t>(topology.batch_size) + 1U;
  const std::uint32_t bra_begin = topology.pair_class_offsets[high_pair_class * stride];
  const auto* density_bounds =
      reinterpret_cast<const ShellPairDensityBounds*>(topology.shell_pair_density_bounds);

  while (true) {
    if (threadIdx.x == 0U) {
      // ``bra_head`` is reset for every page.  Starting the scheduler at the
      // first bra row that intersects this page avoids replaying all earlier
      // rows when a class spans many pages.  The page range is conservative:
      // the first row may begin before ``page_begin`` and the last row may
      // extend beyond ``page_end``; the ket loop below clips both edges.
      bra_ordinal = bra_ordinal_begin + atomicAdd(bra_head, 1U);
    }
    __syncthreads();
    if (bra_ordinal >= bra_ordinal_end) return;
    const std::uint32_t bra_pair = topology.pair_order[bra_begin + bra_ordinal];
    const std::int32_t system = topology.shell_pair_systems[bra_pair];
    if (topology.active != nullptr && topology.active[system] == 0U) {
      continue;
    }
    // This bound is fixed for the bra row and pair class. Pair-order
    // segments are not Schwarz-sorted, so it may reject only the current
    // ket; the exact predicate below still decides every survivor.
    const BoundedPageDensityTails page_density_tails = bounded_page_density_tails(
        batch, topology, system, bra_pair, high_pair_class, low_pair_class);
    const std::uint32_t ket_begin =
        topology.pair_class_offsets[low_pair_class * stride + static_cast<std::size_t>(system)];
    const std::uint32_t ket_end =
        topology
            .pair_class_offsets[low_pair_class * stride + static_cast<std::size_t>(system) + 1U];
    const std::uint32_t system_bra_begin =
        topology.pair_class_offsets[high_pair_class * stride + static_cast<std::size_t>(system)];
    const std::uint64_t bra_local = bra_begin + bra_ordinal - system_bra_begin;
    const std::uint64_t ket_count = ket_end - ket_begin;
    std::uint64_t system_candidate_begin = 0U;
    for (std::int32_t previous = 0; previous < system; ++previous) {
      const std::uint32_t previous_bra_begin =
          topology
              .pair_class_offsets[high_pair_class * stride + static_cast<std::size_t>(previous)];
      const std::uint32_t previous_bra_end =
          topology.pair_class_offsets[high_pair_class * stride +
                                      static_cast<std::size_t>(previous) + 1U];
      const std::uint32_t previous_ket_begin =
          topology.pair_class_offsets[low_pair_class * stride + static_cast<std::size_t>(previous)];
      const std::uint32_t previous_ket_end =
          topology.pair_class_offsets[low_pair_class * stride + static_cast<std::size_t>(previous) +
                                      1U];
      const std::uint64_t previous_bra_count = previous_bra_end - previous_bra_begin;
      const std::uint64_t previous_ket_count = previous_ket_end - previous_ket_begin;
      system_candidate_begin += same_pair_class
                                    ? previous_bra_count * (previous_bra_count + 1U) / 2U
                                    : previous_bra_count * previous_ket_count;
    }
    // Candidate ordinals are contiguous first by system, then by bra pair;
    // same-class streams pack each bra row as a lower-triangle row. Restrict
    // each page to the bra/ket rows that intersect its ordinal interval;
    // otherwise every page would rescan all bra rows and turn a bounded queue
    // into an O(number_of_pages * topology) traversal.
    const std::uint64_t bra_candidate_begin =
        system_candidate_begin +
        (same_pair_class ? bra_local * (bra_local + 1U) / 2U : bra_local * ket_count);
    const std::uint64_t row_candidate_count = same_pair_class ? bra_local + 1U : ket_count;
    const std::uint64_t bra_candidate_end = bra_candidate_begin + row_candidate_count;
    const std::uint64_t page_end = page_begin + page_capacity;
    if (bra_candidate_end <= page_begin) continue;
    if (bra_candidate_begin >= page_end) return;
    const std::uint64_t first_page_offset =
        page_begin > bra_candidate_begin ? page_begin - bra_candidate_begin : 0U;
    const std::uint64_t last_page_offset =
        page_end < bra_candidate_end ? page_end - bra_candidate_begin : row_candidate_count;
    const std::uint32_t ket_first = ket_begin + static_cast<std::uint32_t>(first_page_offset);
    const std::uint32_t ket_last = ket_begin + static_cast<std::uint32_t>(last_page_offset);
    const bool has_density_bound =
        topology.system_pair_density_bounds != nullptr || topology.system_density_bounds != nullptr;
    for (std::uint32_t ket_ordinal = ket_first + threadIdx.x; ket_ordinal < ket_last;
         ket_ordinal += blockDim.x) {
      const std::uint32_t ket_pair = topology.pair_order[ket_ordinal];
      const double quartet_bound =
          topology.shell_pair_bounds[bra_pair] * topology.shell_pair_bounds[ket_pair];
      if constexpr (Purpose == DirectScreeningPurpose::Force) {
        const double force_tolerance =
            fmin(screening_tolerance, kForceDensityProductScreeningTolerance);
        if (has_density_bound && quartet_bound * page_density_tails.force < force_tolerance) {
          continue;
        }
      } else if (has_density_bound &&
                 quartet_bound * page_density_tails.fock < screening_tolerance) {
        continue;
      }
      if (!direct_shell_quartet_survives_screening<Unrestricted, Purpose>(
              batch, bra_pair, ket_pair, screening_tolerance, topology.shell_pair_bounds,
              density_bounds)) {
        continue;
      }
      std::uint32_t ordinal = 0U;
      if (signature_counts != nullptr) {
        const unsigned signature = bounded_force_signature_bucket(batch, bra_pair, ket_pair);
        const std::uint32_t signature_ordinal = atomicAdd(signature_counts + signature, 1U);
        if (signature_offsets == nullptr) continue;
        ordinal = signature_offsets[signature] + signature_ordinal;
      } else {
        ordinal = atomicAdd(task_count, 1U);
      }
      populate_generated_shell_task(batch, {bra_pair, ket_pair, 0U}, tasks[ordinal]);
    }
    __syncthreads();
  }
}

void launch_compact_bounded_exact_class_force_wave_kernel(
    bool unrestricted, DirectScreeningPurpose purpose, dim3 grid, dim3 block,
    std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    const GeneratedShellPairStream* topology_pointer, unsigned shell_class,
    unsigned high_pair_class, unsigned low_pair_class, double screening_tolerance,
    std::uint64_t page_begin, std::uint32_t page_capacity, std::uint32_t bra_ordinal_begin,
    std::uint32_t bra_ordinal_end, bool same_pair_class, GeneratedShellTask* tasks,
    std::uint32_t* task_count, std::uint32_t* bra_head, const std::uint32_t* overflow,
    bool force_execution, std::uint32_t* signature_counts, const std::uint32_t* signature_offsets) {
  if (unrestricted == true) {
    if (purpose == DirectScreeningPurpose::Fock) {
      compact_bounded_exact_class_force_wave_kernel<true, DirectScreeningPurpose::Fock>
          <<<grid, block, shared_bytes, stream>>>(
              batch, topology_pointer, shell_class, high_pair_class, low_pair_class,
              screening_tolerance, page_begin, page_capacity, bra_ordinal_begin, bra_ordinal_end,
              same_pair_class, tasks, task_count, bra_head, overflow, force_execution,
              signature_counts, signature_offsets);
    } else {
      compact_bounded_exact_class_force_wave_kernel<true, DirectScreeningPurpose::Force>
          <<<grid, block, shared_bytes, stream>>>(
              batch, topology_pointer, shell_class, high_pair_class, low_pair_class,
              screening_tolerance, page_begin, page_capacity, bra_ordinal_begin, bra_ordinal_end,
              same_pair_class, tasks, task_count, bra_head, overflow, force_execution,
              signature_counts, signature_offsets);
    }
  } else {
    if (purpose == DirectScreeningPurpose::Fock) {
      compact_bounded_exact_class_force_wave_kernel<false, DirectScreeningPurpose::Fock>
          <<<grid, block, shared_bytes, stream>>>(
              batch, topology_pointer, shell_class, high_pair_class, low_pair_class,
              screening_tolerance, page_begin, page_capacity, bra_ordinal_begin, bra_ordinal_end,
              same_pair_class, tasks, task_count, bra_head, overflow, force_execution,
              signature_counts, signature_offsets);
    } else {
      compact_bounded_exact_class_force_wave_kernel<false, DirectScreeningPurpose::Force>
          <<<grid, block, shared_bytes, stream>>>(
              batch, topology_pointer, shell_class, high_pair_class, low_pair_class,
              screening_tolerance, page_begin, page_capacity, bra_ordinal_begin, bra_ordinal_end,
              same_pair_class, tasks, task_count, bra_head, overflow, force_execution,
              signature_counts, signature_offsets);
    }
  }
}

}  // namespace vibeqc::scf::cuda_execution
