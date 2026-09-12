#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/direct_angular_fock.hpp"
#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_fock_order2.cuh"
#include "scf/cuda/direct_fock_psss.cuh"
#include "scf/cuda/direct_fock_quartet.cuh"
#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

/** Fixed-capacity wrapper retained for high-register angular orders. */
template <bool Unrestricted, unsigned AngularOrder, typename EvalScalar = double>
__global__ void build_fock_direct_quartet_kernel(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* fock,
    const std::uint64_t* generated_fock_shell_class_mask) {
  contract_fock_direct_quartet_subtile<Unrestricted, AngularOrder, EvalScalar>(
      batch, active_shell_quartet_tile_count, active_shell_quartet_tiles, screening_tolerance,
      schwarz_bounds, density, active, fock, generated_fock_shell_class_mask,
      static_cast<std::size_t>(blockIdx.x), threadIdx.x);
}

/** Pack exact ssss shell tasks across all lanes of one worker warp. */
template <bool Unrestricted, unsigned AngularOrder>
__global__ void build_fock_direct_quartet_packed_persistent_kernel(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, std::uint32_t* task_head,
    double screening_tolerance, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* fock) {
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
      // Order zero has one subtile and one exact logical tile per shell
      // quartet, so packed_item is also its compact subtile index.
      contract_fock_direct_quartet_subtile<Unrestricted, AngularOrder>(
          batch, active_shell_quartet_tile_count, active_shell_quartet_tiles, screening_tolerance,
          schwarz_bounds, density, active, fock, nullptr, packed_item, 0U);
    }
  }
}

/** Consume complete psss shell tasks, one independent task per lane. */
template <bool Unrestricted>
__global__ void build_fock_direct_psss_persistent_kernel(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, std::uint32_t* task_head,
    double screening_tolerance, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* fock) {
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
      contract_fock_direct_psss_task<Unrestricted>(batch, active_shell_quartet_tiles[packed_item],
                                                   screening_tolerance, schwarz_bounds, density,
                                                   active, fock);
    }
    // Tail lanes must remain live until the next warp-uniform queue exit so
    // the full-mask shuffle above is valid on every persistent iteration.
  }
}

/** Consume complete order-two shell tasks, one independent task per lane. */
template <bool Unrestricted>
__global__ void build_fock_direct_order2_persistent_kernel(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, std::uint32_t* task_head,
    double screening_tolerance, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* fock,
    const std::uint64_t* generated_fock_shell_class_mask) {
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
      contract_fock_direct_order2_task<Unrestricted>(batch, active_shell_quartet_tiles[packed_item],
                                                     screening_tolerance, schwarz_bounds, density,
                                                     active, fock, generated_fock_shell_class_mask);
    }
  }
}

/** Consume only the active compacted Fock domain from a device queue. */
template <bool Unrestricted, unsigned AngularOrder, typename EvalScalar = double>
__global__ void build_fock_direct_quartet_persistent_kernel(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, std::uint32_t* task_head,
    double screening_tolerance, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* fock,
    const std::uint64_t* generated_fock_shell_class_mask) {
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
    contract_fock_direct_quartet_subtile<Unrestricted, AngularOrder, EvalScalar>(
        batch, active_shell_quartet_tile_count, active_shell_quartet_tiles, screening_tolerance,
        schwarz_bounds, density, active, fock, generated_fock_shell_class_mask, active_subtile,
        threadIdx.x);
  }
}

template <bool Unrestricted, typename EvalScalar = double, unsigned AngularOrder = 0>
void launch_angular_fock_quartets(
    cudaStream_t stream,
    const std::array<std::size_t, detail::kDirectQuartetAngularOrderCount>& capacities,
    const std::array<std::uint32_t, detail::kDirectQuartetAngularOrderCount + 1>& offsets,
    DeviceBatch batch, const std::uint32_t* active_tile_counts,
    const ActiveShellQuartetTile* active_tiles, const std::uint32_t* generic_order5_tile_count,
    const ActiveShellQuartetTile* generic_order5_tiles, std::uint32_t* persistent_task_heads,
    unsigned persistent_worker_blocks, double screening_tolerance, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* fock,
    const std::uint64_t* generated_fock_shell_class_mask) {
  if constexpr (AngularOrder < detail::kDirectQuartetAngularOrderCount) {
    if (capacities[AngularOrder] != 0) {
      const std::uint32_t* order_tile_count = active_tile_counts + AngularOrder;
      const ActiveShellQuartetTile* order_tiles = active_tiles + offsets[AngularOrder];
      if constexpr (AngularOrder == kGenericOrderFiveAngularOrder &&
                    std::is_same_v<EvalScalar, double>) {
        // Every order-five class currently enabled for generated Fock owns an
        // exact queue. Avoid making the generic persistent worker claim all
        // six subtiles only to decode the class and return.
        order_tile_count = generic_order5_tile_count;
        order_tiles = generic_order5_tiles;
      }
      if constexpr (std::is_same_v<EvalScalar, MixedPrecisionFloat>) {
        // Mixed work begins at order three.  It keeps an independent queue and
        // persistent head so the ERI recurrence contains no per-warp precision
        // branch and low-order shell-fused workers remain unchanged.
        if constexpr (AngularOrder >= kMixedFockMinimumAngularOrder &&
                      AngularOrder < kPersistentFockAngularOrderCount) {
          const unsigned capacity_blocks = static_cast<unsigned>(
              capacities[AngularOrder] * detail::direct_quartet_subtiles_per_tile(AngularOrder));
          build_fock_direct_quartet_persistent_kernel<Unrestricted, AngularOrder, EvalScalar>
              <<<std::min(capacity_blocks, persistent_worker_blocks), detail::kDirectQuartetThreads,
                 0, stream>>>(batch, order_tile_count, order_tiles,
                              persistent_task_heads + AngularOrder, screening_tolerance,
                              schwarz_bounds, density, active, fock,
                              generated_fock_shell_class_mask);
        } else if constexpr (AngularOrder >= kPersistentFockAngularOrderCount) {
          build_fock_direct_quartet_kernel<Unrestricted, AngularOrder, EvalScalar>
              <<<static_cast<unsigned>(capacities[AngularOrder] *
                                       detail::direct_quartet_subtiles_per_tile(AngularOrder)),
                 detail::kDirectQuartetThreads, 0, stream>>>(
                  batch, order_tile_count, order_tiles, screening_tolerance, schwarz_bounds,
                  density, active, fock, generated_fock_shell_class_mask);
        }
      } else if constexpr (AngularOrder < kPackedSsssAngularOrderCount) {
        const unsigned capacity_workers =
            static_cast<unsigned>((capacities[AngularOrder] + detail::kDirectQuartetThreads - 1) /
                                  detail::kDirectQuartetThreads);
        build_fock_direct_quartet_packed_persistent_kernel<Unrestricted, AngularOrder>
            <<<std::min(capacity_workers, persistent_worker_blocks), detail::kDirectQuartetThreads,
               0, stream>>>(batch, order_tile_count, order_tiles,
                            persistent_task_heads + AngularOrder, screening_tolerance,
                            schwarz_bounds, density, active, fock);
      } else if constexpr (AngularOrder == kFusedPsssAngularOrder) {
        const unsigned capacity_workers =
            static_cast<unsigned>((capacities[AngularOrder] + detail::kDirectQuartetThreads - 1) /
                                  detail::kDirectQuartetThreads);
        build_fock_direct_psss_persistent_kernel<Unrestricted>
            <<<std::min(capacity_workers, persistent_worker_blocks), detail::kDirectQuartetThreads,
               0, stream>>>(batch, order_tile_count, order_tiles,
                            persistent_task_heads + AngularOrder, screening_tolerance,
                            schwarz_bounds, density, active, fock);
      } else if constexpr (AngularOrder == kFusedOrderTwoAngularOrder) {
        const unsigned capacity_workers =
            static_cast<unsigned>((capacities[AngularOrder] + detail::kDirectQuartetThreads - 1) /
                                  detail::kDirectQuartetThreads);
        build_fock_direct_order2_persistent_kernel<Unrestricted>
            <<<std::min(capacity_workers, persistent_worker_blocks), detail::kDirectQuartetThreads,
               0, stream>>>(batch, order_tile_count, order_tiles,
                            persistent_task_heads + AngularOrder, screening_tolerance,
                            schwarz_bounds, density, active, fock, generated_fock_shell_class_mask);
      } else if constexpr (AngularOrder < kPersistentFockAngularOrderCount) {
        const unsigned capacity_blocks = static_cast<unsigned>(
            capacities[AngularOrder] * detail::direct_quartet_subtiles_per_tile(AngularOrder));
        build_fock_direct_quartet_persistent_kernel<Unrestricted, AngularOrder>
            <<<std::min(capacity_blocks, persistent_worker_blocks), detail::kDirectQuartetThreads,
               0, stream>>>(batch, order_tile_count, order_tiles,
                            persistent_task_heads + AngularOrder, screening_tolerance,
                            schwarz_bounds, density, active, fock, generated_fock_shell_class_mask);
      } else {
        build_fock_direct_quartet_kernel<Unrestricted, AngularOrder>
            <<<static_cast<unsigned>(capacities[AngularOrder] *
                                     detail::direct_quartet_subtiles_per_tile(AngularOrder)),
               detail::kDirectQuartetThreads, 0, stream>>>(
                batch, order_tile_count, order_tiles, screening_tolerance, schwarz_bounds, density,
                active, fock, generated_fock_shell_class_mask);
      }
    }
    launch_angular_fock_quartets<Unrestricted, EvalScalar, AngularOrder + 1>(
        stream, capacities, offsets, batch, active_tile_counts, active_tiles,
        generic_order5_tile_count, generic_order5_tiles, persistent_task_heads,
        persistent_worker_blocks, screening_tolerance, schwarz_bounds, density, active, fock,
        generated_fock_shell_class_mask);
  }
}

void dispatch_angular_fock_quartets(
    bool unrestricted, bool mixed_precision, cudaStream_t stream,
    const std::array<std::size_t, detail::kDirectQuartetAngularOrderCount>& capacities,
    const std::array<std::uint32_t, detail::kDirectQuartetAngularOrderCount + 1>& offsets,
    DeviceBatch batch, const std::uint32_t* active_tile_counts,
    const ActiveShellQuartetTile* active_tiles, const std::uint32_t* generic_order5_tile_count,
    const ActiveShellQuartetTile* generic_order5_tiles, std::uint32_t* persistent_task_heads,
    unsigned persistent_worker_blocks, double screening_tolerance, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* fock,
    const std::uint64_t* generated_fock_shell_class_mask) {
  if (unrestricted) {
    if (mixed_precision) {
      launch_angular_fock_quartets<true, MixedPrecisionFloat>(
          stream, capacities, offsets, batch, active_tile_counts, active_tiles,
          generic_order5_tile_count, generic_order5_tiles, persistent_task_heads,
          persistent_worker_blocks, screening_tolerance, schwarz_bounds, density, active, fock,
          generated_fock_shell_class_mask);
    } else {
      launch_angular_fock_quartets<true, double>(
          stream, capacities, offsets, batch, active_tile_counts, active_tiles,
          generic_order5_tile_count, generic_order5_tiles, persistent_task_heads,
          persistent_worker_blocks, screening_tolerance, schwarz_bounds, density, active, fock,
          generated_fock_shell_class_mask);
    }
  } else {
    if (mixed_precision) {
      launch_angular_fock_quartets<false, MixedPrecisionFloat>(
          stream, capacities, offsets, batch, active_tile_counts, active_tiles,
          generic_order5_tile_count, generic_order5_tiles, persistent_task_heads,
          persistent_worker_blocks, screening_tolerance, schwarz_bounds, density, active, fock,
          generated_fock_shell_class_mask);
    } else {
      launch_angular_fock_quartets<false, double>(
          stream, capacities, offsets, batch, active_tile_counts, active_tiles,
          generic_order5_tile_count, generic_order5_tiles, persistent_task_heads,
          persistent_worker_blocks, screening_tolerance, schwarz_bounds, density, active, fock,
          generated_fock_shell_class_mask);
    }
  }
}

}  // namespace vibeqc::scf::cuda_execution
