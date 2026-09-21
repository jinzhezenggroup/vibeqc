#pragma once

#include <cstddef>

namespace vibeqc::runtime {

/**
 * Runtime-enriched CUDA resource facts shared by compiler/runtime policies.
 *
 * Zero means "unknown" for optional resource facts. Product names are
 * intentionally absent: schedule legality is derived from resources rather
 * than GPU marketing identity.
 */
struct CudaTargetInfo {
  int compute_capability_major{};
  int compute_capability_minor{};
  unsigned warp_size{};
  unsigned maximum_threads_per_block{};
  unsigned maximum_threads_per_sm{};
  unsigned maximum_blocks_per_sm{};
  std::size_t registers_per_sm{};
  std::size_t shared_memory_per_block{};
  std::size_t shared_memory_per_block_optin{};
  std::size_t shared_memory_per_sm{};
  unsigned multiprocessor_count{};
  std::size_t total_global_memory{};
};

template <class DeviceProperties>
CudaTargetInfo cuda_target_info_from_properties(const DeviceProperties& properties) noexcept {
  CudaTargetInfo target;
  target.compute_capability_major = properties.major;
  target.compute_capability_minor = properties.minor;
  target.warp_size = static_cast<unsigned>(properties.warpSize);
  target.maximum_threads_per_block = static_cast<unsigned>(properties.maxThreadsPerBlock);
  target.maximum_threads_per_sm = static_cast<unsigned>(properties.maxThreadsPerMultiProcessor);
  target.maximum_blocks_per_sm = static_cast<unsigned>(properties.maxBlocksPerMultiProcessor);
  target.registers_per_sm = static_cast<std::size_t>(properties.regsPerMultiprocessor);
  target.shared_memory_per_block = static_cast<std::size_t>(properties.sharedMemPerBlock);
  target.shared_memory_per_block_optin =
      static_cast<std::size_t>(properties.sharedMemPerBlockOptin);
  target.shared_memory_per_sm = static_cast<std::size_t>(properties.sharedMemPerMultiprocessor);
  target.multiprocessor_count = static_cast<unsigned>(properties.multiProcessorCount);
  target.total_global_memory = static_cast<std::size_t>(properties.totalGlobalMem);
  return target;
}

}  // namespace vibeqc::runtime
