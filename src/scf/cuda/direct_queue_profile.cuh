#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_queue_index.cuh"

namespace vibeqc::scf::cuda_execution {

/** Accumulate the same final-density shell-class ledger as exact compaction. */
__device__ inline void profile_bounded_direct_shell_quartet(DeviceBatch batch,
                                                            const ActiveShellQuartetTile& task,
                                                            DeviceShellClassProfileEntry* profile) {
  if (profile == nullptr) return;
  const std::int32_t shells[4] = {
      batch.shell_pair_first[task.first_pair],
      batch.shell_pair_second[task.first_pair],
      batch.shell_pair_first[task.second_pair],
      batch.shell_pair_second[task.second_pair],
  };
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[shells[0]], batch.shell_angular[shells[1]],
      batch.shell_angular[shells[2]], batch.shell_angular[shells[3]]);
  if (shell_class >= detail::kDirectQuartetShellClassCount) return;
  const std::size_t first_count = shell_ao_pair_count(batch, task.first_pair);
  const std::size_t second_count = shell_ao_pair_count(batch, task.second_pair);
  const std::size_t ao_quartets = task.first_pair == task.second_pair
                                      ? first_count * (first_count + 1) / 2
                                      : first_count * second_count;
  const std::size_t tiles =
      (ao_quartets + detail::kDirectQuartetTileSize - 1) / detail::kDirectQuartetTileSize;
  unsigned long long primitive_quartets = static_cast<unsigned long long>(ao_quartets);
  for (const std::int32_t shell : shells) {
    primitive_quartets *= static_cast<unsigned long long>(batch.shell_primitive_offsets[shell + 1] -
                                                          batch.shell_primitive_offsets[shell]);
  }
  DeviceShellClassProfileEntry& entry = profile[shell_class];
  atomicAdd(&entry.shell_quartets, 1ULL);
  atomicAdd(&entry.tiles, static_cast<unsigned long long>(tiles));
  atomicAdd(&entry.ao_quartets, static_cast<unsigned long long>(ao_quartets));
  atomicAdd(&entry.primitive_quartets, primitive_quartets);
}

}  // namespace vibeqc::scf::cuda_execution
