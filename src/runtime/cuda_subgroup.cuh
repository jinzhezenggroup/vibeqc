#ifndef VIBEQC_RUNTIME_CUDA_SUBGROUP_CUH
#define VIBEQC_RUNTIME_CUDA_SUBGROUP_CUH

#include <cuda_runtime.h>

namespace vibeqc::runtime {

/** Sum fixed channel arrays within a power-of-two subgroup.
 * The caller captures mask before divergent work and keeps every participating
 * subgroup complete. Only subgroup leaders consume the resulting sums; other
 * lanes hold partial reductions. No shared memory or block barrier is needed.
 */
template <unsigned Rows, unsigned Width, unsigned Capacity, unsigned Channels>
__device__ void subgroup_sum(double (&values)[Capacity][Channels], unsigned mask) {
  static_assert(Rows <= Capacity && Width && Width <= 32 && !(Width & (Width - 1)));
#pragma unroll
  for (unsigned shift = Width / 2; shift; shift /= 2)
#pragma unroll
    for (unsigned row = 0; row < Rows; ++row)
#pragma unroll
      for (unsigned channel = 0; channel < Channels; ++channel)
        values[row][channel] += __shfl_down_sync(mask, values[row][channel], shift, Width);
}

}  // namespace vibeqc::runtime
#endif
