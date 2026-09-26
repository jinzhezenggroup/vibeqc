#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

/** Reset every streaming flag from a mask captured by value, without host storage. */
void launch_reset_bounded_generated_streaming_flags_kernel(dim3 grid, dim3 block,
                                                           std::size_t shared_bytes,
                                                           cudaStream_t stream,
                                                           std::uint64_t selected_mask,
                                                           std::uint32_t* flags);

/** Forward resolved queue policy with unchanged geometry, stream and buffers. */
void launch_scan_bounded_force_signature_counts_kernel(dim3 grid, dim3 block,
                                                       std::size_t shared_bytes,
                                                       cudaStream_t stream,
                                                       std::uint32_t* signature_counts,
                                                       std::uint32_t* signature_offsets,
                                                       std::uint32_t* block_offsets);

/** Forward resolved queue policy with unchanged geometry, stream and buffers. */
void launch_prefix_bounded_force_signature_blocks_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    std::uint32_t* signature_offsets, std::uint32_t* block_offsets, std::uint32_t* task_count);

/** Forward resolved queue policy with unchanged geometry, stream and buffers. */
void launch_prepare_bounded_generated_retry_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    std::uint32_t task_capacity, const std::uint32_t* task_offsets, std::uint32_t* task_counts,
    std::uint32_t* task_heads, std::uint32_t* overflow, std::uint32_t* retry_mask,
    std::uint32_t* retry_offsets, std::uint32_t* retry_any, bool preserve_overflow_counts);

/** Forward resolved queue policy with unchanged geometry, stream and buffers. */
void launch_normalize_bounded_generated_task_counts_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    const std::uint32_t* task_offsets, std::uint32_t* task_counts, std::uint32_t* task_heads,
    std::uint32_t* overflow, bool preserve_overflow_counts);

}  // namespace vibeqc::scf::cuda_execution
