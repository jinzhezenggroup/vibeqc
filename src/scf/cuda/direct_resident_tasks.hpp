#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

/** Forward resolved queue policy with unchanged geometry, stream and buffers. */
void launch_count_ppps_resident_bra_tasks_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    std::size_t active_tile_capacity, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    std::uint64_t enabled_shell_class_mask, std::uint32_t* resident_bra_counts,
    std::uint32_t* resident_signature_counts);

/** Forward resolved queue policy with unchanged geometry, stream and buffers. */
void launch_prefix_ppps_resident_bra_tasks_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                                  cudaStream_t stream,
                                                  std::size_t total_shell_pairs,
                                                  const std::uint32_t* resident_bra_counts,
                                                  std::uint32_t* resident_bra_offsets,
                                                  std::uint32_t* resident_bra_write_counts,
                                                  GeneratedPppsResidentTask* resident_tasks);

/** Forward resolved queue policy with unchanged geometry, stream and buffers. */
void launch_prefix_ppps_resident_signature_buckets_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    std::size_t total_shell_pairs, const std::uint32_t* resident_bra_offsets,
    std::uint32_t* resident_signature_counts, std::uint32_t* resident_signature_offsets);

/** Forward resolved queue policy with unchanged geometry, stream and buffers. */
void launch_materialize_ppps_resident_bra_tasks_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    std::size_t active_tile_capacity, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    const std::uint32_t* resident_bra_offsets, std::uint32_t* resident_bra_write_counts,
    const std::uint32_t* resident_signature_offsets, std::uint32_t* resident_signature_write_counts,
    GeneratedShellTask* resident_ket_tasks, std::uint32_t* resident_ket_signatures);

}  // namespace vibeqc::scf::cuda_execution
