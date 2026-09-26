#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "scf/cuda/topology.hpp"
#include "scf/direct_task_layout.hpp"

namespace vibeqc::scf::cuda_execution {

/** Bounded direct queue planning depends only on topology, capacity and compiled capabilities; it
 * submits no GPU work. */
/**
 * Resolve one bounded generated page to the bra rows it can intersect.
 *
 * Generated force pages are ordered by system, then by bra shell pair, then
 * by ket shell pair.  The old page kernel restarted its bra scheduler at zero
 * for every page and discarded the prefix with a bounds check.  On a large
 * class this turned a bounded page stream into repeated O(N_pair) work.  The
 * host already owns the class-major pair offsets, so derive the first and
 * last relevant bra rows once per page and pass that narrow range to CUDA.
 * Same-class products use the packed lower triangle directly, while mixed
 * pair classes retain their rectangular stream. This keeps page boundaries
 * disjoint without materializing a second class-specific index array.
 */
struct BoundedGeneratedPageRange {
  std::uint64_t candidate_count{};
  std::uint32_t bra_begin{};
  std::uint32_t bra_end{};
};

/** Assign bounded class slices using topology histograms and the compiled registry. */
std::array<std::uint32_t, detail::kDirectQuartetShellClassCount + 1>
make_bounded_generated_task_offsets(
    const HostBatch& host, std::size_t task_capacity,
    std::array<std::uint64_t, detail::kDirectQuartetShellClassCount>* upper_bounds);

/** Return exactly the shell classes present in the packed topology. */
std::uint64_t present_direct_shell_class_mask(const HostBatch& host);

/** Sort shell pairs by class with deterministic per-system offsets. */
/** Build class-major/system-major shell-pair segments for AOT streaming. */
bool make_bounded_stream_shell_pair_order(const HostBatch& host,
                                          std::vector<std::uint32_t>& pair_order,
                                          std::vector<std::uint32_t>& pair_class_offsets);

/** Clip a bounded candidate page to its contributing bra rows, including partial rows. */
BoundedGeneratedPageRange bounded_generated_page_range(
    const std::vector<std::uint32_t>& pair_class_offsets, std::size_t batch_size,
    unsigned high_pair_class, unsigned low_pair_class, std::uint64_t page_begin,
    std::uint32_t page_capacity);

}  // namespace vibeqc::scf::cuda_execution
