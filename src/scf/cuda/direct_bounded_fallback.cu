#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/direct_bounded_contraction.cuh"
#include "scf/cuda/direct_bounded_fallback.hpp"
#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_fock_order2.cuh"
#include "scf/cuda/direct_fock_psss.cuh"
#include "scf/cuda/direct_fock_quartet.cuh"
#include "scf/cuda/direct_force_low_order.cuh"
#include "scf/cuda/direct_force_order2.cuh"
#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/direct_queue_index.cuh"
#include "scf/cuda/direct_queue_profile.cuh"
#include "scf/cuda/direct_screening.cuh"
#include "scf/cuda/direct_task_encoding.cuh"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

/**
 * Enumerate, screen, queue, and drain shell pair-of-pairs hierarchically.
 *
 * A persistent CTA first claims one pair-block product. Conservative Schwarz
 * and density maxima reject the complete block without visiting its members;
 * surviving blocks are expanded in fixed 256-candidate chunks and retain the
 * exact shell-quartet predicate. This keeps storage bounded while replacing
 * the former unconditional O(N_shell^4) scan with a small O(N_shell^4/B^2)
 * outer domain plus exact work only in surviving blocks.
 */
template <bool Unrestricted, DirectScreeningPurpose Purpose, bool Force>
__global__ __launch_bounds__(kBoundedDirectThreads, 1) void bounded_direct_shell_quartet_kernel(
    DeviceBatch batch, double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* shell_pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint64_t* enabled_mask_pointer, std::uint64_t enabled_mask,
    const std::uint32_t* bounded_generated_overflow, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* output,
    unsigned long long* global_cursor, DeviceShellClassProfileEntry* profile) {
  __shared__ ActiveShellQuartetTile queue[detail::kBoundedDirectQueueCapacity];
  __shared__ std::uint32_t queue_count;
  __shared__ unsigned long long block_quartet;
  const unsigned lane = threadIdx.x % detail::kDirectQuartetThreads;
  const unsigned warp = threadIdx.x / detail::kDirectQuartetThreads;
  const std::size_t total = static_cast<std::size_t>(batch.total_shell_pair_block_quartets);

  while (true) {
    if (threadIdx.x == 0) {
      block_quartet = atomicAdd(global_cursor, 1ULL);
    }
    __syncthreads();
    if (block_quartet >= total) return;

    const std::size_t packed_block_quartet = static_cast<std::size_t>(block_quartet);
    const std::int32_t system = shell_pair_block_quartet_system(batch, packed_block_quartet);
    if (active != nullptr && active[system] == 0) continue;
    const std::size_t local_block_quartet =
        packed_block_quartet -
        static_cast<std::size_t>(batch.system_shell_pair_block_quartet_offsets[system]);
    std::size_t first_block_local = 0;
    std::size_t second_block_local = 0;
    decode_lower_triangle(local_block_quartet, first_block_local, second_block_local);
    const std::size_t system_block_begin =
        static_cast<std::size_t>(batch.system_shell_pair_block_offsets[system]);
    const std::size_t first_block = system_block_begin + first_block_local;
    const std::size_t second_block = system_block_begin + second_block_local;
    if (!bounded_direct_block_pair_survives_screening<Purpose>(
            first_block, second_block, system, screening_tolerance, shell_pair_block_bounds,
            system_density_bounds)) {
      continue;
    }

    const std::size_t system_pair_begin =
        static_cast<std::size_t>(batch.system_shell_pair_offsets[system]);
    const std::size_t system_pair_end =
        static_cast<std::size_t>(batch.system_shell_pair_offsets[system + 1]);
    const std::size_t first_ordered_begin =
        system_pair_begin + first_block_local * detail::kBoundedDirectShellPairBlockSize;
    const std::size_t second_ordered_begin =
        system_pair_begin + second_block_local * detail::kBoundedDirectShellPairBlockSize;
    const std::size_t first_count =
        min(detail::kBoundedDirectShellPairBlockSize, system_pair_end - first_ordered_begin);
    const std::size_t second_count =
        min(detail::kBoundedDirectShellPairBlockSize, system_pair_end - second_ordered_begin);
    const bool same_block = first_block == second_block;
    const std::size_t candidate_count =
        same_block ? first_count * (first_count + 1) / 2 : first_count * second_count;

    for (std::size_t candidate_begin = 0; candidate_begin < candidate_count;
         candidate_begin += detail::kBoundedDirectQueueCapacity) {
      if (threadIdx.x == 0) queue_count = 0;
      __syncthreads();
      const std::size_t candidate = candidate_begin + threadIdx.x;
      if (candidate < candidate_count) {
        std::size_t first_local = 0;
        std::size_t second_local = 0;
        if (same_block) {
          decode_lower_triangle(candidate, first_local, second_local);
        } else {
          first_local = candidate / second_count;
          second_local = candidate % second_count;
        }
        const std::size_t first_pair = shell_pair_order[first_ordered_begin + first_local];
        const std::size_t second_pair = shell_pair_order[second_ordered_begin + second_local];
        if (direct_shell_quartet_survives_screening<Unrestricted, Purpose>(
                batch, first_pair, second_pair, screening_tolerance, shell_pair_bounds,
                shell_pair_density_bounds)) {
          const std::int32_t first_shell = batch.shell_pair_first[first_pair];
          const std::int32_t second_shell = batch.shell_pair_second[first_pair];
          const std::int32_t third_shell = batch.shell_pair_first[second_pair];
          const std::int32_t fourth_shell = batch.shell_pair_second[second_pair];
          const unsigned shell_class = direct_quartet_shell_class_device(
              batch.shell_angular[first_shell], batch.shell_angular[second_shell],
              batch.shell_angular[third_shell], batch.shell_angular[fourth_shell]);
          const bool generated_class =
              bounded_generated_overflow[shell_class] == 0U &&
              bounded_generated_class_enabled(shell_class, enabled_mask_pointer, enabled_mask);
          if (!generated_class) {
            const std::uint32_t slot = atomicAdd(&queue_count, 1U);
            queue[slot] = {static_cast<std::uint32_t>(first_pair),
                           static_cast<std::uint32_t>(second_pair), 0U};
            if constexpr (Force) {
              profile_bounded_direct_shell_quartet(batch, queue[slot], profile);
            }
          }
        }
      }
      __syncthreads();

      // Low-order shell tasks fit in one scalar lane. Drain up to 256 of
      // them concurrently before assigning the larger classes one warp each;
      // the former generic path spent 31 idle lanes on every ssss/psss/order2
      // task and dominates molecular systems built from s/p/d basis shells.
      for (std::uint32_t slot = threadIdx.x; slot < queue_count; slot += blockDim.x) {
        const ActiveShellQuartetTile task = queue[slot];
        const std::int32_t first_shell = batch.shell_pair_first[task.first_pair];
        const std::int32_t second_shell = batch.shell_pair_second[task.first_pair];
        const std::int32_t third_shell = batch.shell_pair_first[task.second_pair];
        const std::int32_t fourth_shell = batch.shell_pair_second[task.second_pair];
        const unsigned angular_order =
            batch.shell_angular[first_shell] + batch.shell_angular[second_shell] +
            batch.shell_angular[third_shell] + batch.shell_angular[fourth_shell];
        if constexpr (Force) {
          if (angular_order == 0U) {
            contract_two_electron_force_ssss_task<Unrestricted>(
                batch, task, screening_tolerance, schwarz_bounds, density, active, output, 0U);
          } else if (angular_order == 1U) {
            contract_two_electron_force_psss_task<Unrestricted>(
                batch, task, screening_tolerance, schwarz_bounds, density, active, output, 0U);
          } else if (angular_order == 2U) {
            contract_two_electron_force_psps_task<Unrestricted>(
                batch, task, screening_tolerance, schwarz_bounds, density, active, output, 0U);
            contract_two_electron_force_pair_order2_task<Unrestricted, kPpssShellClass>(
                batch, task, screening_tolerance, schwarz_bounds, density, active, output, 0U);
            contract_two_electron_force_pair_order2_task<Unrestricted, kDsssShellClass>(
                batch, task, screening_tolerance, schwarz_bounds, density, active, output, 0U);
          }
        } else {
          if (angular_order == 0U) {
            contract_fock_direct_quartet_subtile<Unrestricted, 0U>(
                batch, &queue_count, queue + slot, screening_tolerance, schwarz_bounds, density,
                active, output, nullptr, 0U, 0U);
          } else if (angular_order == 1U) {
            contract_fock_direct_psss_task<Unrestricted>(batch, task, screening_tolerance,
                                                         schwarz_bounds, density, active, output);
          } else if (angular_order == 2U) {
            contract_fock_direct_order2_task<Unrestricted>(
                batch, task, screening_tolerance, schwarz_bounds, density, active, output, nullptr);
          }
        }
      }
      __syncthreads();

      for (std::uint32_t slot = warp; slot < queue_count;
           slot += kBoundedDirectThreads / detail::kDirectQuartetThreads) {
        const ActiveShellQuartetTile base = queue[slot];
        const std::int32_t first_shell = batch.shell_pair_first[base.first_pair];
        const std::int32_t second_shell = batch.shell_pair_second[base.first_pair];
        const std::int32_t third_shell = batch.shell_pair_first[base.second_pair];
        const std::int32_t fourth_shell = batch.shell_pair_second[base.second_pair];
        const unsigned angular_order =
            batch.shell_angular[first_shell] + batch.shell_angular[second_shell] +
            batch.shell_angular[third_shell] + batch.shell_angular[fourth_shell];
        if (angular_order <= 2U) continue;
        const std::size_t first_ao_count = shell_ao_pair_count(batch, base.first_pair);
        const std::size_t second_ao_count = shell_ao_pair_count(batch, base.second_pair);
        const std::size_t ao_quartets = base.first_pair == base.second_pair
                                            ? first_ao_count * (first_ao_count + 1) / 2
                                            : first_ao_count * second_ao_count;
        const std::uint32_t tile_count = static_cast<std::uint32_t>(
            (ao_quartets + detail::kDirectQuartetTileSize - 1) / detail::kDirectQuartetTileSize);
        const std::size_t subtile_count = detail::direct_quartet_subtiles_per_tile(angular_order);
        for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
          if (lane == 0) queue[slot].tile = tile;
          __syncwarp();
          for (std::size_t subtile = 0; subtile < subtile_count; ++subtile) {
            if constexpr (Force) {
              contract_bounded_direct_force_subtile<Unrestricted>(
                  batch, angular_order, &queue_count, queue + slot, screening_tolerance,
                  schwarz_bounds, density, active, output, subtile, lane);
            } else {
              contract_bounded_direct_fock_subtile<Unrestricted>(
                  batch, angular_order, &queue_count, queue + slot, screening_tolerance,
                  schwarz_bounds, density, active, output, subtile, lane);
            }
          }
          __syncwarp();
        }
      }
      __syncthreads();
    }
  }
}

void launch_bounded_direct_shell_quartet_kernel(
    bool unrestricted, DirectScreeningPurpose purpose, dim3 grid, dim3 block,
    std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch, double screening_tolerance,
    const double* shell_pair_bounds, const ShellPairDensityBounds* shell_pair_density_bounds,
    const std::uint32_t* shell_pair_order, const double* shell_pair_block_bounds,
    const double* system_density_bounds, const std::uint64_t* enabled_mask_pointer,
    std::uint64_t enabled_mask, const std::uint32_t* bounded_generated_overflow,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* output,
    unsigned long long* global_cursor, DeviceShellClassProfileEntry* profile) {
  if (unrestricted == true) {
    if (purpose == DirectScreeningPurpose::Fock) {
      bounded_direct_shell_quartet_kernel<true, DirectScreeningPurpose::Fock, true>
          <<<grid, block, shared_bytes, stream>>>(
              batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
              shell_pair_order, shell_pair_block_bounds, system_density_bounds,
              enabled_mask_pointer, enabled_mask, bounded_generated_overflow, schwarz_bounds,
              density, active, output, global_cursor, profile);
    } else {
      bounded_direct_shell_quartet_kernel<true, DirectScreeningPurpose::Force, true>
          <<<grid, block, shared_bytes, stream>>>(
              batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
              shell_pair_order, shell_pair_block_bounds, system_density_bounds,
              enabled_mask_pointer, enabled_mask, bounded_generated_overflow, schwarz_bounds,
              density, active, output, global_cursor, profile);
    }
  } else {
    if (purpose == DirectScreeningPurpose::Fock) {
      bounded_direct_shell_quartet_kernel<false, DirectScreeningPurpose::Fock, true>
          <<<grid, block, shared_bytes, stream>>>(
              batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
              shell_pair_order, shell_pair_block_bounds, system_density_bounds,
              enabled_mask_pointer, enabled_mask, bounded_generated_overflow, schwarz_bounds,
              density, active, output, global_cursor, profile);
    } else {
      bounded_direct_shell_quartet_kernel<false, DirectScreeningPurpose::Force, true>
          <<<grid, block, shared_bytes, stream>>>(
              batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
              shell_pair_order, shell_pair_block_bounds, system_density_bounds,
              enabled_mask_pointer, enabled_mask, bounded_generated_overflow, schwarz_bounds,
              density, active, output, global_cursor, profile);
    }
  }
}

}  // namespace vibeqc::scf::cuda_execution
