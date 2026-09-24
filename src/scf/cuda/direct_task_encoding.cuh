#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_queue_index.cuh"

namespace vibeqc::scf::cuda_execution {

/**
 * Return the canonical ``pp`` pair for an active ppps tile.
 *
 * Direct compaction stores an unordered pair-of-pairs, while generated
 * kernels consume the pair with the larger triangular class in slot zero.
 * Keeping this test in one device helper makes resident grouping use exactly
 * the same symmetry convention as generated task materialization.  The
 * primitive-pair limit is part of the predicate: a bra that cannot fit in
 * shared memory must remain visible to the ordinary generated ppps queue.
 */
__device__ __forceinline__ bool resident_ppps_bra_pair(const DeviceBatch& batch,
                                                       const ActiveShellQuartetTile& tile,
                                                       std::uint32_t& bra_pair) {
  if (tile.tile != 0U) return false;
  const std::int32_t first_shell = batch.shell_pair_first[tile.first_pair];
  const std::int32_t second_shell = batch.shell_pair_second[tile.first_pair];
  const std::int32_t third_shell = batch.shell_pair_first[tile.second_pair];
  const std::int32_t fourth_shell = batch.shell_pair_second[tile.second_pair];
  const unsigned first_pair_class = direct_shell_pair_class_cuda(batch.shell_angular[first_shell],
                                                                 batch.shell_angular[second_shell]);
  const unsigned second_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[third_shell], batch.shell_angular[fourth_shell]);
  if (first_pair_class == 2U && second_pair_class == 1U) {
    bra_pair = tile.first_pair;
  } else if (first_pair_class == 1U && second_pair_class == 2U) {
    bra_pair = tile.second_pair;
  } else {
    return false;
  }
  const std::int64_t begin = batch.shell_pair_primitive_offsets[bra_pair];
  const std::int64_t end =
      batch.shell_pair_primitive_offsets[static_cast<std::size_t>(bra_pair) + 1U];
  const std::int64_t count = end - begin;
  return count > 0 &&
         count <= static_cast<std::int64_t>(kGeneratedPppsResidentMaximumBraPrimitivePairs);
}

/** Return the exact orientation/ket-primitive bucket for one resident tile. */
__device__ __forceinline__ unsigned resident_ppps_signature_bucket(
    const DeviceBatch& batch, const ActiveShellQuartetTile& tile, std::uint32_t bra_pair) {
  const bool pair_exchanged = bra_pair == tile.second_pair;
  const std::uint32_t ket_pair = pair_exchanged ? tile.first_pair : tile.second_pair;
  const std::int64_t ket_begin = batch.shell_pair_primitive_offsets[ket_pair];
  const std::int64_t ket_end = batch.shell_pair_primitive_offsets[ket_pair + 1U];
  const std::uint64_t ket_count =
      ket_end > ket_begin ? static_cast<std::uint64_t>(ket_end - ket_begin) : 0U;
  const unsigned primitive_bucket = static_cast<unsigned>(
      min(ket_count, static_cast<std::uint64_t>(kPppsSignaturePrimitivePairBuckets - 1U)));
  return (pair_exchanged ? kPppsSignaturePrimitivePairBuckets : 0U) + primitive_bucket;
}

/** Return the page-local loop/orientation signature for one bounded task. */
__device__ __forceinline__ unsigned bounded_force_signature_bucket(const DeviceBatch& batch,
                                                                   std::uint32_t first_pair,
                                                                   std::uint32_t second_pair) {
  const std::int64_t first_begin = batch.shell_pair_primitive_offsets[first_pair];
  const std::int64_t first_end = batch.shell_pair_primitive_offsets[first_pair + 1U];
  const std::int64_t second_begin = batch.shell_pair_primitive_offsets[second_pair];
  const std::int64_t second_end = batch.shell_pair_primitive_offsets[second_pair + 1U];
  const std::uint64_t first_count =
      first_end > first_begin ? static_cast<std::uint64_t>(first_end - first_begin) : 0U;
  const std::uint64_t second_count =
      second_end > second_begin ? static_cast<std::uint64_t>(second_end - second_begin) : 0U;
  const unsigned first_bucket = static_cast<unsigned>(
      min(first_count, static_cast<std::uint64_t>(kPppsSignaturePrimitivePairBuckets - 1U)));
  const unsigned second_bucket = static_cast<unsigned>(
      min(second_count, static_cast<std::uint64_t>(kPppsSignaturePrimitivePairBuckets - 1U)));
  const std::int32_t first_shell = batch.shell_pair_first[first_pair];
  const std::int32_t second_shell = batch.shell_pair_second[first_pair];
  const std::int32_t third_shell = batch.shell_pair_first[second_pair];
  const std::int32_t fourth_shell = batch.shell_pair_second[second_pair];
  const unsigned orientation =
      (batch.shell_angular[first_shell] < batch.shell_angular[second_shell] ? 2U : 0U) |
      (batch.shell_angular[third_shell] < batch.shell_angular[fourth_shell] ? 1U : 0U);
  return (orientation * kPppsSignaturePrimitivePairBuckets + first_bucket) *
             kPppsSignaturePrimitivePairBuckets +
         second_bucket;
}

/** Return the ordered primitive-pair loop signature for one low-order tile. */
__device__ __forceinline__ unsigned generated_low_order_signature_bucket(
    const DeviceBatch& batch, const ActiveShellQuartetTile& tile) {
  const std::int64_t first_begin = batch.shell_pair_primitive_offsets[tile.first_pair];
  const std::int64_t first_end = batch.shell_pair_primitive_offsets[tile.first_pair + 1U];
  const std::int64_t second_begin = batch.shell_pair_primitive_offsets[tile.second_pair];
  const std::int64_t second_end = batch.shell_pair_primitive_offsets[tile.second_pair + 1U];
  const std::uint64_t first_count =
      first_end > first_begin ? static_cast<std::uint64_t>(first_end - first_begin) : 0U;
  const std::uint64_t second_count =
      second_end > second_begin ? static_cast<std::uint64_t>(second_end - second_begin) : 0U;
  const unsigned first_bucket = static_cast<unsigned>(
      min(first_count, static_cast<std::uint64_t>(kLowOrderSignaturePrimitivePairBuckets - 1U)));
  const unsigned second_bucket = static_cast<unsigned>(
      min(second_count, static_cast<std::uint64_t>(kLowOrderSignaturePrimitivePairBuckets - 1U)));
  return first_bucket * kLowOrderSignaturePrimitivePairBuckets + second_bucket;
}

/** Map each supported scalar class to its private signature histogram. */
__device__ __forceinline__ unsigned generated_low_order_signature_index(unsigned shell_class,
                                                                        unsigned signature) {
  const unsigned class_slot = shell_class == kPspsShellClass ? 0U : 1U;
  return class_slot * kLowOrderSignatureBucketsPerClass + signature;
}

/**
 * Fill the stable generated task ABI from one canonicalized shell quartet.
 *
 * Both the ordinary class queue and the resident ppps queue use this helper.
 * In particular, the two one-bit pair-orientation mask records the swaps
 * applied before pair-exchange canonicalization; generated force code uses
 * it to map primitive-pair product scales back to physical centers.
 */
__device__ __forceinline__ void populate_generated_shell_task(const DeviceBatch& batch,
                                                              const ActiveShellQuartetTile& tile,
                                                              GeneratedShellTask& task) {
  std::int32_t shells[4] = {
      batch.shell_pair_first[tile.first_pair],
      batch.shell_pair_second[tile.first_pair],
      batch.shell_pair_first[tile.second_pair],
      batch.shell_pair_second[tile.second_pair],
  };
  std::uint32_t shell_pairs[2] = {tile.first_pair, tile.second_pair};
  std::uint32_t reversed_shell_pair_mask = 0U;
  if (batch.shell_angular[shells[0]] < batch.shell_angular[shells[1]]) {
    const std::int32_t swap = shells[0];
    shells[0] = shells[1];
    shells[1] = swap;
    reversed_shell_pair_mask |= 1U;
  }
  if (batch.shell_angular[shells[2]] < batch.shell_angular[shells[3]]) {
    const std::int32_t swap = shells[2];
    shells[2] = shells[3];
    shells[3] = swap;
    reversed_shell_pair_mask |= 2U;
  }
  const unsigned first_pair_class =
      direct_shell_pair_class_cuda(batch.shell_angular[shells[0]], batch.shell_angular[shells[1]]);
  const unsigned second_pair_class =
      direct_shell_pair_class_cuda(batch.shell_angular[shells[2]], batch.shell_angular[shells[3]]);
  if (first_pair_class < second_pair_class) {
    const std::int32_t first = shells[0];
    const std::int32_t second = shells[1];
    shells[0] = shells[2];
    shells[1] = shells[3];
    shells[2] = first;
    shells[3] = second;
    const std::uint32_t pair_swap = shell_pairs[0];
    shell_pairs[0] = shell_pairs[1];
    shell_pairs[1] = pair_swap;
    reversed_shell_pair_mask =
        ((reversed_shell_pair_mask & 1U) << 1U) | ((reversed_shell_pair_mask & 2U) >> 1U);
  }

  const std::int32_t system = batch.shell_pair_systems[tile.first_pair];
  const std::size_t matrix_order = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = matrix_order * matrix_order;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * matrix_order;
#pragma unroll
  for (unsigned center = 0; center < 4U; ++center) {
    const std::int32_t shell = shells[center];
    task.primitive_begin[center] = static_cast<std::uint64_t>(batch.shell_primitive_offsets[shell]);
    task.primitive_end[center] =
        static_cast<std::uint64_t>(batch.shell_primitive_offsets[shell + 1]);
    const std::size_t ao_begin = static_cast<std::size_t>(batch.shell_direct_ao_offsets[shell]);
    task.ao_begin[center] = static_cast<std::uint64_t>(ao_begin - system_ao_begin);
    task.ao_coefficient_begin[center] = static_cast<std::uint64_t>(ao_begin);
    task.shell[center] = static_cast<std::uint32_t>(shell);
    task.atom[center] = static_cast<std::uint32_t>(batch.shell_atoms[shell]);
  }
  task.density_offset = static_cast<std::uint64_t>(static_cast<std::size_t>(system) * matrix_size);
  task.spin_offset =
      static_cast<std::uint64_t>(static_cast<std::size_t>(system) * 2U * matrix_size);
  task.matrix_order = static_cast<std::uint32_t>(matrix_order);
  task.shell_pair[0] = shell_pairs[0];
  task.shell_pair[1] = shell_pairs[1];
  task.reversed_shell_pair_mask = reversed_shell_pair_mask;
  task.fock_consumer = detail::GeneratedFockConsumer::HartreeFock;
}

/** Read the runtime exact-class mask used by the bounded generated routes. */
__device__ __forceinline__ bool bounded_generated_class_enabled(
    unsigned shell_class, const std::uint64_t* enabled_mask_pointer, std::uint64_t enabled_mask) {
  if (enabled_mask_pointer != nullptr) enabled_mask = *enabled_mask_pointer;
  return shell_class < detail::kDirectQuartetShellClassCount &&
         (enabled_mask & (std::uint64_t{1} << shell_class)) != 0U;
}

}  // namespace vibeqc::scf::cuda_execution
