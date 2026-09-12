#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/direct_fock_accumulation.cuh"
#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/direct_native_order2_shell.cuh"
#include "scf/cuda/direct_queue_index.cuh"
#include "scf/cuda/matrix_index.cuh"
#include "scf/cuda/packed_basis.hpp"

// Retained direct fock order2 contraction helpers.
// Borrow immutable metadata and density/output views; host plans own lifetime.

namespace vibeqc::scf::cuda_execution {

/** One canonical shell slot and its position in the original quartet. */
struct Order2SourceSlot {
  std::int32_t shell;
  unsigned original;
};

/** Map raw AO slots to the fused vector's canonical Cartesian product. */
__device__ __forceinline__ unsigned order2_component_index(const DeviceBatch& batch,
                                                           const Order2SourceSlot (&slots)[4],
                                                           const std::size_t (&raw_ao)[4],
                                                           std::size_t system_ao_begin) {
  unsigned output = 0;
#pragma unroll
  for (unsigned slot = 0; slot < 4; ++slot) {
    const unsigned angular = batch.shell_angular[slots[slot].shell];
    const unsigned component_count = (angular + 1) * (angular + 2) / 2;
    const std::size_t local_begin =
        static_cast<std::size_t>(batch.shell_direct_ao_offsets[slots[slot].shell]) -
        system_ao_begin;
    const unsigned component = static_cast<unsigned>(raw_ao[slots[slot].original] - local_begin);
    output = output * component_count + component;
  }
  return output;
}

/** Evaluate and scatter one complete psps, ppss, or dsss shell task. */
template <bool Unrestricted>
__device__ inline __noinline__ void contract_fock_direct_order2_task(
    const DeviceBatch& batch, ActiveShellQuartetTile task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* fock,
    const std::uint64_t* generated_fock_shell_class_mask) {
  if (task.tile != 0U) return;
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active != nullptr && active[system] == 0) return;

  Order2SourceSlot slots[4] = {
      {batch.shell_pair_first[first_pair], 0},
      {batch.shell_pair_second[first_pair], 1},
      {batch.shell_pair_first[second_pair], 2},
      {batch.shell_pair_second[second_pair], 3},
  };
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[slots[0].shell], batch.shell_angular[slots[1].shell],
      batch.shell_angular[slots[2].shell], batch.shell_angular[slots[3].shell]);
  if (shell_class != 2U && shell_class != 3U && shell_class != 6U) return;
  // Generated order-two workers execute before this handwritten fallback.
  // Honor the exact-class mask here as the generic subtile path does, or the
  // same shell quartet is scattered into the Fock matrix twice.
  if (generated_fock_shell_class_mask != nullptr &&
      ((*generated_fock_shell_class_mask & (std::uint64_t{1} << shell_class)) != 0U)) {
    return;
  }

  if (batch.shell_angular[slots[0].shell] < batch.shell_angular[slots[1].shell]) {
    const Order2SourceSlot swap = slots[0];
    slots[0] = slots[1];
    slots[1] = swap;
  }
  if (batch.shell_angular[slots[2].shell] < batch.shell_angular[slots[3].shell]) {
    const Order2SourceSlot swap = slots[2];
    slots[2] = slots[3];
    slots[3] = swap;
  }
  const unsigned first_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[slots[0].shell], batch.shell_angular[slots[1].shell]);
  const unsigned second_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[slots[2].shell], batch.shell_angular[slots[3].shell]);
  if (first_pair_class < second_pair_class) {
    const Order2SourceSlot first_swap = slots[0];
    slots[0] = slots[2];
    slots[2] = first_swap;
    const Order2SourceSlot second_swap = slots[1];
    slots[1] = slots[3];
    slots[3] = second_swap;
  }

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  const std::size_t first_ao_pair_count = shell_ao_pair_count(batch, first_pair);
  const std::size_t second_ao_pair_count = shell_ao_pair_count(batch, second_pair);
  const bool same_shell_pair = first_pair == second_pair;
  const std::size_t ao_quartet_count = same_shell_pair
                                           ? first_ao_pair_count * (first_ao_pair_count + 1) / 2
                                           : first_ao_pair_count * second_ao_pair_count;

  unsigned active_component_mask = 0;
  for (std::size_t ordinal = 0; ordinal < ao_quartet_count; ++ordinal) {
    std::size_t first_ao_pair = 0;
    std::size_t second_ao_pair = 0;
    if (same_shell_pair) {
      decode_lower_triangle(ordinal, first_ao_pair, second_ao_pair);
    } else {
      first_ao_pair = ordinal / second_ao_pair_count;
      second_ao_pair = ordinal % second_ao_pair_count;
    }
    std::size_t raw_ao[4];
    decode_shell_ao_pair(batch, first_pair, first_ao_pair, system_ao_begin, raw_ao[0], raw_ao[1]);
    decode_shell_ao_pair(batch, second_pair, second_ao_pair, system_ao_begin, raw_ao[2], raw_ao[3]);
    if (schwarz_bounds[physical_offset + matrix_index(raw_ao[0], raw_ao[1], n)] *
            schwarz_bounds[physical_offset + matrix_index(raw_ao[2], raw_ao[3], n)] <
        screening_tolerance) {
      continue;
    }
    active_component_mask |= 1U << order2_component_index(batch, slots, raw_ao, system_ao_begin);
  }
  if (active_component_mask == 0) return;

  Order2IntegralVector integral{};
  switch (shell_class) {
    case 2:
      integral = contracted_eri_cartesian_source_order2_shell<1, 0, 1, 0>(
          batch, slots[0].shell, slots[1].shell, slots[2].shell, slots[3].shell,
          active_component_mask);
      break;
    case 3:
      integral = contracted_eri_cartesian_source_order2_shell<1, 1, 0, 0>(
          batch, slots[0].shell, slots[1].shell, slots[2].shell, slots[3].shell,
          active_component_mask);
      break;
    default:
      integral = contracted_eri_cartesian_source_order2_shell<2, 0, 0, 0>(
          batch, slots[0].shell, slots[1].shell, slots[2].shell, slots[3].shell,
          active_component_mask);
      break;
  }

  for (std::size_t ordinal = 0; ordinal < ao_quartet_count; ++ordinal) {
    std::size_t first_ao_pair = 0;
    std::size_t second_ao_pair = 0;
    if (same_shell_pair) {
      decode_lower_triangle(ordinal, first_ao_pair, second_ao_pair);
    } else {
      first_ao_pair = ordinal / second_ao_pair_count;
      second_ao_pair = ordinal % second_ao_pair_count;
    }
    std::size_t raw_ao[4];
    decode_shell_ao_pair(batch, first_pair, first_ao_pair, system_ao_begin, raw_ao[0], raw_ao[1]);
    decode_shell_ao_pair(batch, second_pair, second_ao_pair, system_ao_begin, raw_ao[2], raw_ao[3]);
    const unsigned component = order2_component_index(batch, slots, raw_ao, system_ao_begin);
    if ((active_component_mask & (1U << component)) == 0 || integral.component[component] == 0.0) {
      continue;
    }
    accumulate_direct_fock_integral<Unrestricted>(n, physical_offset, spin_offset, density, fock,
                                                  raw_ao[0], raw_ao[1], raw_ao[2], raw_ao[3],
                                                  integral.component[component]);
  }
}

}  // namespace vibeqc::scf::cuda_execution
