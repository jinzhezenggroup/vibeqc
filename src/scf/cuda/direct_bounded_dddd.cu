#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/direct_bounded_dddd.hpp"
#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_fock_quartet.cuh"
#include "scf/cuda/direct_force_quartet.cuh"
#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/direct_queue_index.cuh"
#include "scf/cuda/direct_queue_profile.cuh"
#include "scf/cuda/direct_screening.cuh"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

/**
 * Stream only canonical dddd work from class-major shell-pair segments.
 *
 * This is an exact class-specific route, not the bounded generic fallback:
 * its outer domain is the dd-pair triangle and every accepted quartet is
 * consumed immediately.  It replaces the currently unreliable generated
 * dddd value/gradient consumer while retaining O(N_shell^2) topology storage
 * and zero whole-topology scan when all production classes are covered.
 */
template <bool Unrestricted, DirectScreeningPurpose Purpose, bool Force>
__global__
__launch_bounds__(detail::kDirectQuartetThreads) void bounded_direct_dddd_streaming_kernel(
    DeviceBatch batch, const GeneratedShellPairStream* topology_pointer, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* output,
    std::uint32_t* bra_head, DeviceShellClassProfileEntry* profile,
    unsigned long long* fp64_work_count) {
  static_assert(detail::kDirectQuartetThreads == 32);
  constexpr std::uint32_t kSkip = 0U;
  constexpr std::uint32_t kConsume = 1U;
  constexpr std::uint32_t kFinished = 2U;
  constexpr std::size_t kSubtilesPerTile =
      detail::direct_quartet_subtiles_per_tile(kDdddAngularOrder);

  __shared__ ActiveShellQuartetTile task;
  __shared__ std::uint32_t queue_count;
  __shared__ std::uint32_t candidate_ordinal;
  __shared__ std::uint32_t stream_state;
  __shared__ std::uint32_t tile_count;

  const unsigned lane = threadIdx.x;
  const GeneratedShellPairStream& topology = *topology_pointer;
  const std::size_t stride = static_cast<std::size_t>(topology.batch_size) + 1U;
  const std::size_t dd_pair_class = 5U;
  const auto* density_bounds =
      reinterpret_cast<const ShellPairDensityBounds*>(topology.shell_pair_density_bounds);

  if (lane == 0U) queue_count = 1U;
  __syncwarp();
  while (true) {
    if (lane == 0U) {
      candidate_ordinal = atomicAdd(bra_head, 1U);
      std::uint64_t remaining = candidate_ordinal;
      stream_state = kFinished;
      for (std::int32_t system = 0; system < topology.batch_size; ++system) {
        const std::uint32_t pair_begin =
            topology.pair_class_offsets[dd_pair_class * stride + static_cast<std::size_t>(system)];
        const std::uint32_t pair_end =
            topology
                .pair_class_offsets[dd_pair_class * stride + static_cast<std::size_t>(system) + 1U];
        const std::uint64_t pair_count = pair_end - pair_begin;
        const std::uint64_t system_candidates = pair_count * (pair_count + 1U) / 2U;
        if (remaining >= system_candidates) {
          remaining -= system_candidates;
          continue;
        }

        std::size_t bra_local = 0U;
        std::size_t ket_local = 0U;
        decode_lower_triangle(static_cast<std::size_t>(remaining), bra_local, ket_local);
        const std::uint32_t bra_pair = topology.pair_order[pair_begin + bra_local];
        const std::uint32_t ket_pair = topology.pair_order[pair_begin + ket_local];
        bool keep = (active == nullptr || active[system] != 0U) &&
                    topology.shell_pair_bounds[bra_pair] * topology.shell_pair_bounds[ket_pair] >=
                        screening_tolerance;
        if (keep) {
          keep = direct_shell_quartet_survives_screening<Unrestricted, Purpose>(
              batch, bra_pair, ket_pair, screening_tolerance, topology.shell_pair_bounds,
              density_bounds);
        }
        stream_state = keep ? kConsume : kSkip;
        if (keep) {
          if (fp64_work_count != nullptr) atomicAdd(fp64_work_count, 1ULL);
          task = {bra_pair, ket_pair, 0U};
          const std::size_t first_count = shell_ao_pair_count(batch, bra_pair);
          const std::size_t second_count = shell_ao_pair_count(batch, ket_pair);
          const std::size_t ao_quartets = bra_pair == ket_pair
                                              ? first_count * (first_count + 1U) / 2U
                                              : first_count * second_count;
          tile_count = static_cast<std::uint32_t>(
              (ao_quartets + detail::kDirectQuartetTileSize - 1U) / detail::kDirectQuartetTileSize);
        }
        break;
      }
    }
    __syncwarp();
    if (stream_state == kFinished) return;
    if (stream_state != kConsume) continue;

    if constexpr (Force) {
      if (lane == 0U) {
        profile_bounded_direct_shell_quartet(batch, task, profile);
      }
      __syncwarp();
    }
    for (std::uint32_t tile = 0U; tile < tile_count; ++tile) {
      if (lane == 0U) task.tile = tile;
      __syncwarp();
      for (std::size_t subtile = 0U; subtile < kSubtilesPerTile; ++subtile) {
        if constexpr (Force) {
          contract_two_electron_force_quartet_subtile<Unrestricted, kDdddAngularOrder>(
              batch, &queue_count, &task, screening_tolerance, schwarz_bounds, density, active,
              output, 0U, subtile, lane);
        } else {
          contract_fock_direct_quartet_subtile<Unrestricted, kDdddAngularOrder>(
              batch, &queue_count, &task, screening_tolerance, schwarz_bounds, density, active,
              output, nullptr, subtile, lane);
        }
      }
      __syncwarp();
    }
  }
}

void launch_bounded_direct_dddd_streaming_kernel(
    bool unrestricted, DirectScreeningPurpose purpose, bool force, dim3 grid, dim3 block,
    std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    const GeneratedShellPairStream* topology_pointer, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* output,
    std::uint32_t* bra_head, DeviceShellClassProfileEntry* profile,
    unsigned long long* fp64_work_count) {
  if (unrestricted == true) {
    if (purpose == DirectScreeningPurpose::Fock) {
      if (force == false) {
        bounded_direct_dddd_streaming_kernel<true, DirectScreeningPurpose::Fock, false>
            <<<grid, block, shared_bytes, stream>>>(batch, topology_pointer, screening_tolerance,
                                                    schwarz_bounds, density, active, output,
                                                    bra_head, profile, fp64_work_count);
      } else {
        bounded_direct_dddd_streaming_kernel<true, DirectScreeningPurpose::Fock, true>
            <<<grid, block, shared_bytes, stream>>>(batch, topology_pointer, screening_tolerance,
                                                    schwarz_bounds, density, active, output,
                                                    bra_head, profile, fp64_work_count);
      }
    } else {
      bounded_direct_dddd_streaming_kernel<true, DirectScreeningPurpose::Force, true>
          <<<grid, block, shared_bytes, stream>>>(batch, topology_pointer, screening_tolerance,
                                                  schwarz_bounds, density, active, output, bra_head,
                                                  profile, fp64_work_count);
    }
  } else {
    if (purpose == DirectScreeningPurpose::Fock) {
      if (force == false) {
        bounded_direct_dddd_streaming_kernel<false, DirectScreeningPurpose::Fock, false>
            <<<grid, block, shared_bytes, stream>>>(batch, topology_pointer, screening_tolerance,
                                                    schwarz_bounds, density, active, output,
                                                    bra_head, profile, fp64_work_count);
      } else {
        bounded_direct_dddd_streaming_kernel<false, DirectScreeningPurpose::Fock, true>
            <<<grid, block, shared_bytes, stream>>>(batch, topology_pointer, screening_tolerance,
                                                    schwarz_bounds, density, active, output,
                                                    bra_head, profile, fp64_work_count);
      }
    } else {
      bounded_direct_dddd_streaming_kernel<false, DirectScreeningPurpose::Force, true>
          <<<grid, block, shared_bytes, stream>>>(batch, topology_pointer, screening_tolerance,
                                                  schwarz_bounds, density, active, output, bra_head,
                                                  profile, fp64_work_count);
    }
  }
}

}  // namespace vibeqc::scf::cuda_execution
