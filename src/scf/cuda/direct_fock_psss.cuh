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
#include "scf/cuda/direct_native_psss.cuh"
#include "scf/cuda/matrix_index.cuh"
#include "scf/cuda/packed_basis.hpp"

// Retained direct fock psss contraction helpers.
// Borrow immutable metadata and density/output views; host plans own lifetime.

namespace vibeqc::scf::cuda_execution {

/**
 * Evaluate and scatter one complete order-one shell task.
 *
 * Shell-pair topology is index-canonical rather than angular-canonical, so
 * the p shell can occupy any input slot. Integral evaluation is reordered to
 * canonical (p s|s s), while screening and Fock scatter keep the original AO
 * slots to preserve the existing eightfold symmetry semantics.
 */
template <bool Unrestricted>
__device__ inline __noinline__ void contract_fock_direct_psss_task(
    const DeviceBatch& batch, ActiveShellQuartetTile task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* fock) {
  // A psss shell quartet has three AO outputs and therefore exactly one tile.
  if (task.tile != 0U) return;
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active != nullptr && active[system] == 0) return;

  const std::int32_t raw_shell[4] = {
      batch.shell_pair_first[first_pair],
      batch.shell_pair_second[first_pair],
      batch.shell_pair_first[second_pair],
      batch.shell_pair_second[second_pair],
  };
  unsigned p_slot = 4;
  unsigned p_count = 0;
  for (unsigned slot = 0; slot < 4; ++slot) {
    const unsigned angular = batch.shell_angular[raw_shell[slot]];
    if (angular == 1) {
      p_slot = slot;
      ++p_count;
    } else if (angular != 0) {
      return;
    }
  }
  if (p_count != 1) return;

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  std::size_t raw_ao[4] = {
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[raw_shell[0]]) - system_ao_begin,
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[raw_shell[1]]) - system_ao_begin,
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[raw_shell[2]]) - system_ao_begin,
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[raw_shell[3]]) - system_ao_begin,
  };
  const std::size_t p_ao_begin = raw_ao[p_slot];
  unsigned active_axis_mask = 0;
  for (unsigned axis = 0; axis < 3; ++axis) {
    raw_ao[p_slot] = p_ao_begin + axis;
    const double first_bound =
        schwarz_bounds[physical_offset + matrix_index(raw_ao[0], raw_ao[1], n)];
    const double second_bound =
        schwarz_bounds[physical_offset + matrix_index(raw_ao[2], raw_ao[3], n)];
    if (first_bound * second_bound >= screening_tolerance) {
      active_axis_mask |= 1U << axis;
    }
  }
  if (active_axis_mask == 0) return;

  std::int32_t canonical_shell[4] = {raw_shell[0], raw_shell[1], raw_shell[2], raw_shell[3]};
  std::size_t canonical_pair[2] = {first_pair, second_pair};
  if (p_slot == 1) {
    const std::int32_t swap = canonical_shell[0];
    canonical_shell[0] = canonical_shell[1];
    canonical_shell[1] = swap;
  } else if (p_slot >= 2) {
    if (p_slot == 3) {
      const std::int32_t swap = canonical_shell[2];
      canonical_shell[2] = canonical_shell[3];
      canonical_shell[3] = swap;
    }
    const std::int32_t first_swap = canonical_shell[0];
    canonical_shell[0] = canonical_shell[2];
    canonical_shell[2] = first_swap;
    const std::int32_t second_swap = canonical_shell[1];
    canonical_shell[1] = canonical_shell[3];
    canonical_shell[3] = second_swap;
    const std::size_t pair_swap = canonical_pair[0];
    canonical_pair[0] = canonical_pair[1];
    canonical_pair[1] = pair_swap;
  }

  const PsssIntegralVector integral = contracted_eri_cartesian_source_psss(
      batch, canonical_pair[0], canonical_pair[1], canonical_shell[0], canonical_shell[1],
      canonical_shell[2], canonical_shell[3]);
  for (unsigned axis = 0; axis < 3; ++axis) {
    if ((active_axis_mask & (1U << axis)) == 0 || integral.axis[axis] == 0.0) {
      continue;
    }
    raw_ao[p_slot] = p_ao_begin + axis;
    accumulate_direct_fock_integral<Unrestricted>(n, physical_offset, spin_offset, density, fock,
                                                  raw_ao[0], raw_ao[1], raw_ao[2], raw_ao[3],
                                                  integral.axis[axis]);
  }
}

}  // namespace vibeqc::scf::cuda_execution
