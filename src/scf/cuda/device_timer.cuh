#pragma once

#include <cstdint>

namespace vibeqc::scf::cuda_execution {

/** Read the stream profiling clock without host synchronization. */
__device__ __forceinline__ std::uint64_t globaltimer_nanoseconds() {
  std::uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

}  // namespace vibeqc::scf::cuda_execution
