#include <cmath>
#include <cub/block/block_scan.cuh>

#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_queue_scan.hpp"

namespace vibeqc::scf::cuda_execution {

/** Device-launchable graphs cannot borrow a call-local host upload buffer. */
__global__ void reset_bounded_generated_streaming_flags_kernel(std::uint64_t selected_mask,
                                                               std::uint32_t* flags) {
  const unsigned shell_class = blockIdx.x * blockDim.x + threadIdx.x;
  if (shell_class >= detail::kDirectQuartetShellClassCount) return;
  flags[shell_class] = (selected_mask & (std::uint64_t{1} << shell_class)) != 0U;
}

/** Scan bounded signature chunks in parallel and reset them for scatter. */
__global__ void scan_bounded_force_signature_counts_kernel(std::uint32_t* signature_counts,
                                                           std::uint32_t* signature_offsets,
                                                           std::uint32_t* block_offsets) {
  using BlockScan = cub::BlockScan<std::uint32_t, kBoundedForceSignatureScanThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  const unsigned signature = blockIdx.x * kBoundedForceSignatureScanThreads + threadIdx.x;
  const std::uint32_t count =
      signature < kBoundedForceSignatureBucketCount ? signature_counts[signature] : 0U;
  std::uint32_t local_offset = 0U;
  std::uint32_t block_total = 0U;
  BlockScan(scan_storage).ExclusiveSum(count, local_offset, block_total);
  if (signature < kBoundedForceSignatureBucketCount) {
    signature_offsets[signature] = local_offset;
    signature_counts[signature] = 0U;
  }
  if (threadIdx.x == 0U) block_offsets[blockIdx.x] = block_total;
}

/** Complete the chunk prefix and publish the bounded page task count. */
__global__ void prefix_bounded_force_signature_blocks_kernel(std::uint32_t* signature_offsets,
                                                             std::uint32_t* block_offsets,
                                                             std::uint32_t* task_count) {
  using BlockScan = cub::BlockScan<std::uint32_t, kBoundedForceSignatureScanThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  const std::uint32_t count =
      threadIdx.x < kBoundedForceSignatureScanBlockCount ? block_offsets[threadIdx.x] : 0U;
  std::uint32_t block_offset = 0U;
  std::uint32_t page_total = 0U;
  BlockScan(scan_storage).ExclusiveSum(count, block_offset, page_total);
  if (threadIdx.x < kBoundedForceSignatureScanBlockCount) {
    block_offsets[threadIdx.x] = block_offset;
  }
  __syncthreads();
  for (unsigned signature = threadIdx.x; signature < kBoundedForceSignatureBucketCount;
       signature += blockDim.x) {
    signature_offsets[signature] += block_offsets[signature / kBoundedForceSignatureScanThreads];
  }
  if (threadIdx.x == 0U) *task_count = page_total;
}

/**
 * Normalize the first generated wave and plan an exact overflow-only retry.
 *
 * The retry reuses the complete task arena after successful first-wave
 * consumers drain it. Exact observed counts define the second-wave slices;
 * if their sum still exceeds the arena, proportional slices preserve useful
 * generated work while the remaining classes are completed by exact-class
 * paged compaction.
 */
__global__ void prepare_bounded_generated_retry_kernel(
    std::uint32_t task_capacity, const std::uint32_t* task_offsets, std::uint32_t* task_counts,
    std::uint32_t* task_heads, std::uint32_t* overflow, std::uint32_t* retry_mask,
    std::uint32_t* retry_offsets, std::uint32_t* retry_any, bool preserve_overflow_counts) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  std::uint64_t retry_total = 0;
  for (unsigned shell_class = 0; shell_class < detail::kDirectQuartetShellClassCount;
       ++shell_class) {
    const std::uint32_t capacity = task_offsets[shell_class + 1U] - task_offsets[shell_class];
    const bool retry = overflow[shell_class] != 0U || task_counts[shell_class] > capacity;
    retry_mask[shell_class] = retry ? 1U : 0U;
    if (retry) retry_total += task_counts[shell_class];
  }

  std::uint64_t assigned = 0;
  if (retry_total > task_capacity) {
    for (unsigned shell_class = 0; shell_class < detail::kDirectQuartetShellClassCount;
         ++shell_class) {
      if (retry_mask[shell_class] == 0U) continue;
      assigned +=
          static_cast<std::uint64_t>(task_capacity) * task_counts[shell_class] / retry_total;
    }
  }
  std::uint64_t extra =
      retry_total > task_capacity ? static_cast<std::uint64_t>(task_capacity) - assigned : 0U;
  std::uint64_t cursor = 0;
  for (unsigned shell_class = 0; shell_class < detail::kDirectQuartetShellClassCount;
       ++shell_class) {
    retry_offsets[shell_class] = static_cast<std::uint32_t>(cursor);
    if (retry_mask[shell_class] != 0U) {
      std::uint64_t capacity =
          retry_total <= task_capacity
              ? task_counts[shell_class]
              : static_cast<std::uint64_t>(task_capacity) * task_counts[shell_class] / retry_total;
      if (extra != 0U) {
        ++capacity;
        --extra;
      }
      cursor += capacity;
    }
    if (!preserve_overflow_counts && retry_mask[shell_class] != 0U) {
      task_counts[shell_class] = 0U;
    }
    task_heads[shell_class] = 0U;
    overflow[shell_class] = retry_mask[shell_class];
  }
  retry_offsets[detail::kDirectQuartetShellClassCount] = static_cast<std::uint32_t>(cursor);
  *retry_any = retry_total == 0U ? 0U : 1U;
}

/** Disable only generated classes that exceeded their fixed arena slice. */
__global__ void normalize_bounded_generated_task_counts_kernel(const std::uint32_t* task_offsets,
                                                               std::uint32_t* task_counts,
                                                               std::uint32_t* task_heads,
                                                               std::uint32_t* overflow,
                                                               bool preserve_overflow_counts) {
  const unsigned shell_class = blockIdx.x * blockDim.x + threadIdx.x;
  if (shell_class >= detail::kDirectQuartetShellClassCount) return;
  const std::uint32_t capacity = task_offsets[shell_class + 1U] - task_offsets[shell_class];
  if (overflow[shell_class] != 0U || task_counts[shell_class] > capacity) {
    if (!preserve_overflow_counts) task_counts[shell_class] = 0U;
    overflow[shell_class] = 1U;
  }
  task_heads[shell_class] = 0U;
}

void launch_reset_bounded_generated_streaming_flags_kernel(dim3 grid, dim3 block,
                                                           std::size_t shared_bytes,
                                                           cudaStream_t stream,
                                                           std::uint64_t selected_mask,
                                                           std::uint32_t* flags) {
  reset_bounded_generated_streaming_flags_kernel<<<grid, block, shared_bytes, stream>>>(
      selected_mask, flags);
}

void launch_scan_bounded_force_signature_counts_kernel(dim3 grid, dim3 block,
                                                       std::size_t shared_bytes,
                                                       cudaStream_t stream,
                                                       std::uint32_t* signature_counts,
                                                       std::uint32_t* signature_offsets,
                                                       std::uint32_t* block_offsets) {
  scan_bounded_force_signature_counts_kernel<<<grid, block, shared_bytes, stream>>>(
      signature_counts, signature_offsets, block_offsets);
}

void launch_prefix_bounded_force_signature_blocks_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    std::uint32_t* signature_offsets, std::uint32_t* block_offsets, std::uint32_t* task_count) {
  prefix_bounded_force_signature_blocks_kernel<<<grid, block, shared_bytes, stream>>>(
      signature_offsets, block_offsets, task_count);
}

void launch_prepare_bounded_generated_retry_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    std::uint32_t task_capacity, const std::uint32_t* task_offsets, std::uint32_t* task_counts,
    std::uint32_t* task_heads, std::uint32_t* overflow, std::uint32_t* retry_mask,
    std::uint32_t* retry_offsets, std::uint32_t* retry_any, bool preserve_overflow_counts) {
  prepare_bounded_generated_retry_kernel<<<grid, block, shared_bytes, stream>>>(
      task_capacity, task_offsets, task_counts, task_heads, overflow, retry_mask, retry_offsets,
      retry_any, preserve_overflow_counts);
}

void launch_normalize_bounded_generated_task_counts_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    const std::uint32_t* task_offsets, std::uint32_t* task_counts, std::uint32_t* task_heads,
    std::uint32_t* overflow, bool preserve_overflow_counts) {
  normalize_bounded_generated_task_counts_kernel<<<grid, block, shared_bytes, stream>>>(
      task_offsets, task_counts, task_heads, overflow, preserve_overflow_counts);
}

}  // namespace vibeqc::scf::cuda_execution
