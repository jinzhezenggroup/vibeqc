#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/direct_angular_force.hpp"
#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_force_low_order.cuh"
#include "scf/cuda/direct_force_order2.cuh"
#include "scf/cuda/direct_force_quartet.cuh"
#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/direct_screening.cuh"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

/** Fixed-capacity wrapper for the small generic high-order force grids. */
template <bool Unrestricted, unsigned AngularOrder>
__global__ void two_electron_force_quartet_kernel(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    std::uint64_t generated_shell_class_mask) {
  contract_two_electron_force_quartet_subtile<Unrestricted, AngularOrder>(
      batch, active_shell_quartet_tile_count, active_shell_quartet_tiles, screening_tolerance,
      schwarz_bounds, density, active, forces, generated_shell_class_mask,
      static_cast<std::size_t>(blockIdx.x), threadIdx.x);
}

/** Pack independent ssss derivative shell tasks across one worker warp. */
template <bool Unrestricted, unsigned AngularOrder>
__global__ void two_electron_force_quartet_packed_persistent_kernel(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, std::uint32_t* task_head,
    double screening_tolerance, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* forces, std::uint64_t generated_shell_class_mask) {
  static_assert(AngularOrder < kPackedSsssAngularOrderCount);
  const unsigned lane = threadIdx.x;
  const std::uint32_t work_count = *active_shell_quartet_tile_count;
  while (true) {
    std::uint32_t packed_begin = 0;
    if (lane == 0) {
      packed_begin = atomicAdd(task_head, static_cast<std::uint32_t>(warpSize));
    }
    packed_begin = __shfl_sync(0xffffffffU, packed_begin, 0);
    if (packed_begin >= work_count) return;
    const std::uint32_t packed_item = packed_begin + lane;
    if (packed_item < work_count) {
      contract_two_electron_force_ssss_task<Unrestricted>(
          batch, active_shell_quartet_tiles[packed_item], screening_tolerance, schwarz_bounds,
          density, active, forces, generated_shell_class_mask);
    }
  }
}

/**
 * Keep one p-s bra resident while block threads traverse all s-s ket pairs.
 *
 * Total angular order one contains only psss quartets, so enumerating each
 * p-s shell pair once and each s-s shell pair in its system once preserves
 * unique quartet ownership without consulting the unordered compact queue.
 * The shared primitive-pair records remove the remaining repeated bra loads
 * across the hundreds of ket tasks normally associated with one 192-AO bra.
 */
template <bool Unrestricted>
// Four resident 128-thread blocks cap this register-heavy contraction at
// 128 registers/thread on sm_120.  The extra occupancy hides the long
// primitive-pair dependency chain without changing the resident-bra schedule.
__global__
__launch_bounds__(kResidentPsssThreads, 4) void two_electron_force_psss_resident_bra_kernel(
    DeviceBatch batch, const PsssResidentTask* resident_tasks,
    const std::uint32_t* resident_ket_pairs, std::size_t resident_task_count,
    double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, bool force_density_product_screening,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    std::uint64_t generated_shell_class_mask) {
  extern __shared__ PrimitivePairData resident_first_pairs[];
  const std::size_t task_index = static_cast<std::size_t>(blockIdx.x);
  if (task_index >= resident_task_count) return;
  const PsssResidentTask task = resident_tasks[task_index];
  const std::size_t bra_pair = task.bra_pair;

  const std::int32_t system = batch.shell_pair_systems[bra_pair];
  if (active[system] == 0) return;
  const std::int32_t bra_first_shell = batch.shell_pair_first[bra_pair];
  const std::int32_t bra_second_shell = batch.shell_pair_second[bra_pair];
  const unsigned bra_first_angular = batch.shell_angular[bra_first_shell];
  const unsigned bra_second_angular = batch.shell_angular[bra_second_shell];
  if (bra_first_angular + bra_second_angular != 1U) return;

  const std::int64_t bra_primitive_begin = batch.shell_pair_primitive_offsets[bra_pair];
  const std::int64_t bra_primitive_count =
      batch.shell_pair_primitive_offsets[bra_pair + 1U] - bra_primitive_begin;
  if (bra_primitive_count <= 0 ||
      bra_primitive_count > static_cast<std::int64_t>(kResidentPsssMaximumBraPrimitivePairs))
    return;
  for (std::int64_t primitive = threadIdx.x; primitive < bra_primitive_count;
       primitive += blockDim.x) {
    resident_first_pairs[primitive] = batch.shell_primitive_pairs[bra_primitive_begin + primitive];
  }
  __syncthreads();

  for (std::size_t local_ket = threadIdx.x; local_ket < task.ket_count; local_ket += blockDim.x) {
    const std::size_t ket_pair = resident_ket_pairs[task.ket_begin + local_ket];
    const std::size_t first_pair = bra_pair > ket_pair ? bra_pair : ket_pair;
    const std::size_t second_pair = bra_pair > ket_pair ? ket_pair : bra_pair;
    const bool survives_screening =
        force_density_product_screening
            ? direct_shell_quartet_survives_screening<Unrestricted, DirectScreeningPurpose::Force>(
                  batch, first_pair, second_pair, screening_tolerance, shell_pair_bounds,
                  shell_pair_density_bounds)
            : direct_shell_quartet_survives_screening<Unrestricted, DirectScreeningPurpose::Fock>(
                  batch, first_pair, second_pair, screening_tolerance, shell_pair_bounds,
                  shell_pair_density_bounds);
    if (!survives_screening) {
      continue;
    }
    contract_two_electron_force_psss_task<Unrestricted, true>(
        batch,
        {static_cast<std::uint32_t>(first_pair), static_cast<std::uint32_t>(second_pair), 0U},
        screening_tolerance, schwarz_bounds, density, active, forces, generated_shell_class_mask,
        resident_first_pairs, bra_primitive_count);
  }
}

/** Consume complete density-weighted psss force tasks, one task per lane. */
template <bool Unrestricted>
__global__ void two_electron_force_psss_persistent_kernel(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, std::uint32_t* task_head,
    double screening_tolerance, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* forces, std::uint64_t generated_shell_class_mask) {
  const unsigned lane = threadIdx.x;
  const std::uint32_t work_count = *active_shell_quartet_tile_count;
  while (true) {
    std::uint32_t packed_begin = 0;
    if (lane == 0) {
      packed_begin = atomicAdd(task_head, static_cast<std::uint32_t>(warpSize));
    }
    packed_begin = __shfl_sync(0xffffffffU, packed_begin, 0);
    if (packed_begin >= work_count) return;
    const std::uint32_t packed_item = packed_begin + lane;
    if (packed_item < work_count) {
      contract_two_electron_force_psss_task<Unrestricted>(
          batch, active_shell_quartet_tiles[packed_item], screening_tolerance, schwarz_bounds,
          density, active, forces, generated_shell_class_mask);
    }
    // Keep tail lanes live through the next full-mask queue broadcast.
  }
}

/** Scan compact order-two tiles and consume complete psps shell tasks. */
template <bool Unrestricted>
__global__ void two_electron_force_psps_grid_stride_kernel(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    std::uint64_t generated_shell_class_mask) {
  const std::uint32_t work_count = *active_shell_quartet_tile_count;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  for (std::uint32_t task_index = blockIdx.x * blockDim.x + threadIdx.x; task_index < work_count;
       task_index += stride) {
    contract_two_electron_force_psps_task<Unrestricted>(
        batch, active_shell_quartet_tiles[task_index], screening_tolerance, schwarz_bounds, density,
        active, forces, generated_shell_class_mask);
  }
}

/** Scan compact order-two tiles for one exact ppss or dsss class. */
template <bool Unrestricted, unsigned TargetShellClass>
__global__ void two_electron_force_pair_order2_grid_stride_kernel(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    std::uint64_t generated_shell_class_mask) {
  static_assert(TargetShellClass == kPpssShellClass || TargetShellClass == kDsssShellClass);
  const std::uint32_t work_count = *active_shell_quartet_tile_count;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  for (std::uint32_t task_index = blockIdx.x * blockDim.x + threadIdx.x; task_index < work_count;
       task_index += stride) {
    contract_two_electron_force_pair_order2_task<Unrestricted, TargetShellClass>(
        batch, active_shell_quartet_tiles[task_index], screening_tolerance, schwarz_bounds, density,
        active, forces, generated_shell_class_mask);
  }
}

/**
 * Persistent one-warp workers dynamically consume only compacted force work.
 *
 * The topology-capacity launch remains useful for CUDA Graph Fock replay, but
 * the final force executes outside that iterative Graph. A device task head
 * therefore removes empty capacity blocks and balances irregular AO-quartet
 * derivative cost without introducing a host readback of compacted counts.
 */
template <bool Unrestricted, unsigned AngularOrder>
__global__ void two_electron_force_quartet_persistent_kernel(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, std::uint32_t* task_head,
    double screening_tolerance, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* forces, std::uint64_t generated_shell_class_mask) {
  static_assert(AngularOrder < detail::kDirectQuartetAngularOrderCount);
  const unsigned lane = threadIdx.x % warpSize;
  constexpr std::uint32_t subtiles_per_tile =
      static_cast<std::uint32_t>(detail::direct_quartet_subtiles_per_tile(AngularOrder));
  const std::uint32_t work_count = *active_shell_quartet_tile_count * subtiles_per_tile;
  while (true) {
    std::uint32_t active_subtile = 0;
    if (lane == 0) active_subtile = atomicAdd(task_head, 1U);
    active_subtile = __shfl_sync(0xffffffffU, active_subtile, 0);
    if (active_subtile >= work_count) return;
    contract_two_electron_force_quartet_subtile<Unrestricted, AngularOrder>(
        batch, active_shell_quartet_tile_count, active_shell_quartet_tiles, screening_tolerance,
        schwarz_bounds, density, active, forces, generated_shell_class_mask, active_subtile,
        threadIdx.x);
  }
}

template <bool Unrestricted, unsigned AngularOrder = 0>
void launch_angular_force_quartets(
    cudaStream_t stream,
    const std::array<std::size_t, detail::kDirectQuartetAngularOrderCount>& capacities,
    const std::array<std::uint32_t, detail::kDirectQuartetAngularOrderCount + 1>& offsets,
    DeviceBatch batch, const std::uint32_t* active_tile_counts,
    const ActiveShellQuartetTile* active_tiles, const std::uint32_t* generic_order5_tile_count,
    const ActiveShellQuartetTile* generic_order5_tiles, std::uint32_t* persistent_task_heads,
    unsigned persistent_worker_blocks, const PsssResidentTask* psss_resident_tasks,
    const std::uint32_t* psss_resident_ket_pairs, std::size_t psss_resident_task_count,
    std::size_t resident_psss_bra_primitive_pairs, double screening_tolerance,
    const double* shell_pair_bounds, const ShellPairDensityBounds* shell_pair_density_bounds,
    bool force_density_product_screening, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* forces, std::uint64_t generated_shell_class_mask) {
  if constexpr (AngularOrder < detail::kDirectQuartetAngularOrderCount) {
    if (capacities[AngularOrder] != 0) {
      const std::uint32_t* order_tile_count = active_tile_counts + AngularOrder;
      const ActiveShellQuartetTile* order_tiles = active_tiles + offsets[AngularOrder];
      if constexpr (AngularOrder == kGenericOrderFiveAngularOrder) {
        // Generated classes have exact queues. The generic order-five worker
        // consumes only classes not selected by the current runtime mask.
        order_tile_count = generic_order5_tile_count;
        order_tiles = generic_order5_tiles;
      }
      if constexpr (AngularOrder < kPackedSsssAngularOrderCount) {
        const unsigned capacity_workers =
            static_cast<unsigned>((capacities[AngularOrder] + detail::kDirectQuartetThreads - 1) /
                                  detail::kDirectQuartetThreads);
        two_electron_force_quartet_packed_persistent_kernel<Unrestricted, AngularOrder>
            <<<std::min(capacity_workers, persistent_worker_blocks), detail::kDirectQuartetThreads,
               0, stream>>>(batch, order_tile_count, order_tiles,
                            persistent_task_heads + AngularOrder, screening_tolerance,
                            schwarz_bounds, density, active, forces, generated_shell_class_mask);
      } else if constexpr (AngularOrder == kFusedPsssAngularOrder) {
        if (psss_resident_task_count != 0 && resident_psss_bra_primitive_pairs != 0 &&
            resident_psss_bra_primitive_pairs <= kResidentPsssMaximumBraPrimitivePairs) {
          two_electron_force_psss_resident_bra_kernel<Unrestricted>
              <<<static_cast<unsigned>(psss_resident_task_count), kResidentPsssThreads,
                 resident_psss_bra_primitive_pairs * sizeof(PrimitivePairData), stream>>>(
                  batch, psss_resident_tasks, psss_resident_ket_pairs, psss_resident_task_count,
                  screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
                  force_density_product_screening, schwarz_bounds, density, active, forces,
                  generated_shell_class_mask);
        } else {
          const unsigned capacity_workers =
              static_cast<unsigned>((capacities[AngularOrder] + detail::kDirectQuartetThreads - 1) /
                                    detail::kDirectQuartetThreads);
          two_electron_force_psss_persistent_kernel<Unrestricted>
              <<<std::min(capacity_workers, persistent_worker_blocks),
                 detail::kDirectQuartetThreads, 0, stream>>>(
                  batch, order_tile_count, order_tiles, persistent_task_heads + AngularOrder,
                  screening_tolerance, schwarz_bounds, density, active, forces,
                  generated_shell_class_mask);
        }
      } else if constexpr (AngularOrder == kFusedOrderTwoAngularOrder) {
        const unsigned capacity_workers =
            static_cast<unsigned>((capacities[AngularOrder] + detail::kDirectQuartetThreads - 1) /
                                  detail::kDirectQuartetThreads);
        two_electron_force_psps_grid_stride_kernel<Unrestricted>
            <<<std::min(capacity_workers, persistent_worker_blocks), detail::kDirectQuartetThreads,
               0, stream>>>(batch, order_tile_count, order_tiles, screening_tolerance,
                            schwarz_bounds, density, active, forces, generated_shell_class_mask);
        two_electron_force_pair_order2_grid_stride_kernel<Unrestricted, kPpssShellClass>
            <<<std::min(capacity_workers, persistent_worker_blocks), detail::kDirectQuartetThreads,
               0, stream>>>(batch, order_tile_count, order_tiles, screening_tolerance,
                            schwarz_bounds, density, active, forces, generated_shell_class_mask);
        two_electron_force_pair_order2_grid_stride_kernel<Unrestricted, kDsssShellClass>
            <<<std::min(capacity_workers, persistent_worker_blocks), detail::kDirectQuartetThreads,
               0, stream>>>(batch, order_tile_count, order_tiles, screening_tolerance,
                            schwarz_bounds, density, active, forces, generated_shell_class_mask);

        // Exact shell workers own all three order-two classes unless an
        // enabled generated kernel already consumed one. The generic launch
        // is retained as a guarded safety net for future class additions.
        const std::uint64_t generic_shell_class_mask =
            generated_shell_class_mask | (std::uint64_t{1} << kPspsShellClass) |
            (std::uint64_t{1} << kPpssShellClass) | (std::uint64_t{1} << kDsssShellClass);
        const unsigned capacity_blocks = static_cast<unsigned>(
            capacities[AngularOrder] * detail::direct_quartet_subtiles_per_tile(AngularOrder));
        two_electron_force_quartet_persistent_kernel<Unrestricted, AngularOrder>
            <<<std::min(capacity_blocks, persistent_worker_blocks), detail::kDirectQuartetThreads,
               0, stream>>>(batch, order_tile_count, order_tiles,
                            persistent_task_heads + AngularOrder, screening_tolerance,
                            schwarz_bounds, density, active, forces, generic_shell_class_mask);
      } else if constexpr (AngularOrder < kPersistentForceAngularOrderCount) {
        const unsigned capacity_blocks = static_cast<unsigned>(
            capacities[AngularOrder] * detail::direct_quartet_subtiles_per_tile(AngularOrder));
        two_electron_force_quartet_persistent_kernel<Unrestricted, AngularOrder>
            <<<std::min(capacity_blocks, persistent_worker_blocks), detail::kDirectQuartetThreads,
               0, stream>>>(batch, order_tile_count, order_tiles,
                            persistent_task_heads + AngularOrder, screening_tolerance,
                            schwarz_bounds, density, active, forces, generated_shell_class_mask);
      } else {
        two_electron_force_quartet_kernel<Unrestricted, AngularOrder>
            <<<static_cast<unsigned>(capacities[AngularOrder] *
                                     detail::direct_quartet_subtiles_per_tile(AngularOrder)),
               detail::kDirectQuartetThreads, 0, stream>>>(
                batch, order_tile_count, order_tiles, screening_tolerance, schwarz_bounds, density,
                active, forces, generated_shell_class_mask);
      }
    }
    launch_angular_force_quartets<Unrestricted, AngularOrder + 1>(
        stream, capacities, offsets, batch, active_tile_counts, active_tiles,
        generic_order5_tile_count, generic_order5_tiles, persistent_task_heads,
        persistent_worker_blocks, psss_resident_tasks, psss_resident_ket_pairs,
        psss_resident_task_count, resident_psss_bra_primitive_pairs, screening_tolerance,
        shell_pair_bounds, shell_pair_density_bounds, force_density_product_screening,
        schwarz_bounds, density, active, forces, generated_shell_class_mask);
  }
}

void launch_two_electron_force_psss_resident_bra_kernel(
    bool unrestricted, dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    DeviceBatch batch, const PsssResidentTask* resident_tasks,
    const std::uint32_t* resident_ket_pairs, std::size_t resident_task_count,
    double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, bool force_density_product_screening,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    std::uint64_t generated_shell_class_mask) {
  if (unrestricted == true) {
    two_electron_force_psss_resident_bra_kernel<true><<<grid, block, shared_bytes, stream>>>(
        batch, resident_tasks, resident_ket_pairs, resident_task_count, screening_tolerance,
        shell_pair_bounds, shell_pair_density_bounds, force_density_product_screening,
        schwarz_bounds, density, active, forces, generated_shell_class_mask);
  } else {
    two_electron_force_psss_resident_bra_kernel<false><<<grid, block, shared_bytes, stream>>>(
        batch, resident_tasks, resident_ket_pairs, resident_task_count, screening_tolerance,
        shell_pair_bounds, shell_pair_density_bounds, force_density_product_screening,
        schwarz_bounds, density, active, forces, generated_shell_class_mask);
  }
}

void dispatch_angular_force_quartets(
    bool unrestricted, cudaStream_t stream,
    const std::array<std::size_t, detail::kDirectQuartetAngularOrderCount>& capacities,
    const std::array<std::uint32_t, detail::kDirectQuartetAngularOrderCount + 1>& offsets,
    DeviceBatch batch, const std::uint32_t* active_tile_counts,
    const ActiveShellQuartetTile* active_tiles, const std::uint32_t* generic_order5_tile_count,
    const ActiveShellQuartetTile* generic_order5_tiles, std::uint32_t* persistent_task_heads,
    unsigned persistent_worker_blocks, const PsssResidentTask* psss_resident_tasks,
    const std::uint32_t* psss_resident_ket_pairs, std::size_t psss_resident_task_count,
    std::size_t resident_psss_bra_primitive_pairs, double screening_tolerance,
    const double* shell_pair_bounds, const ShellPairDensityBounds* shell_pair_density_bounds,
    bool force_density_product_screening, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* forces, std::uint64_t generated_shell_class_mask) {
  if (unrestricted) {
    launch_angular_force_quartets<true>(
        stream, capacities, offsets, batch, active_tile_counts, active_tiles,
        generic_order5_tile_count, generic_order5_tiles, persistent_task_heads,
        persistent_worker_blocks, psss_resident_tasks, psss_resident_ket_pairs,
        psss_resident_task_count, resident_psss_bra_primitive_pairs, screening_tolerance,
        shell_pair_bounds, shell_pair_density_bounds, force_density_product_screening,
        schwarz_bounds, density, active, forces, generated_shell_class_mask);
  } else {
    launch_angular_force_quartets<false>(
        stream, capacities, offsets, batch, active_tile_counts, active_tiles,
        generic_order5_tile_count, generic_order5_tiles, persistent_task_heads,
        persistent_worker_blocks, psss_resident_tasks, psss_resident_ket_pairs,
        psss_resident_task_count, resident_psss_bra_primitive_pairs, screening_tolerance,
        shell_pair_bounds, shell_pair_density_bounds, force_density_product_screening,
        schwarz_bounds, density, active, forces, generated_shell_class_mask);
  }
}

}  // namespace vibeqc::scf::cuda_execution
