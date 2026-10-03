"""Bounded FP64 triangular Gram schedule for resident occupied DF exchange.

This lowers K = weight * U.T @ U; it neither chooses occupied orbitals nor
changes the final-state lease. The runtime lends disjoint, already charged
scratch and keeps SYRK when the shape/capacity gate is not satisfied.
"""

from __future__ import annotations

GRAM_TILE = 32
GRAM_CHUNK = 32768
MAX_SPLITS = 64


def occupied_gram_splits(n: int, reduction: int, capacity: int) -> int:
    """Return the number of bounded reduction slices, or zero for BLAS.

    Short products retain the provider's low launch overhead. The large-product
    domain gives each triangular output tile independent reduction slices without
    duplicating the upper triangle. Capacity counts doubles, not bytes.
    """
    limit = (1 << 31) - 1
    if n < 384 or n > limit // n or reduction < GRAM_CHUNK or reduction > limit:
        return 0
    splits = (reduction + GRAM_CHUNK - 1) // GRAM_CHUNK
    return splits if splits <= MAX_SPLITS and splits <= capacity // (n * n) else 0


def emit_occupied_gram() -> str:
    """Emit the host gate and allocation-free CUDA implementation together."""
    return r"""
namespace generativeqc::scf::generated {
inline constexpr std::size_t df_occupied_gram_tile = 32;
inline constexpr std::size_t df_occupied_gram_chunk = 32768;
inline std::size_t df_occupied_gram_splits(
    std::size_t n, std::size_t reduction, std::size_t capacity) noexcept {
  constexpr auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (n < 384 || n > limit / n || reduction < df_occupied_gram_chunk || reduction > limit)
    return 0;
  const auto splits = 1 + (reduction - 1) / df_occupied_gram_chunk;
  return splits <= 64 && splits <= capacity / (n * n) ? splits : 0;
}
#if defined(__CUDACC__)
/** Each lower output tile owns all its partials. The 32x32 shared panels
 * reuse contiguous reduction values for four FP64 accumulators per thread.
 * Padding covers ragged AO/reduction tails without out-of-bounds reads.
 * No atomics or stream-global state: graph replay has the same reduction order.
 */
__global__ void df_occupied_gram_partials(
    int n, int reduction, const double* input, double* partials) {
  constexpr int tile = 32, panel = 32, chunk = 32768;
  const int tile_i = blockIdx.x, tile_j = blockIdx.y;
  if (tile_i < tile_j) return;
  const int x = threadIdx.x, y = threadIdx.y, lane = y * 16 + x;
  const int begin = blockIdx.z * chunk;
  const int end = min(reduction, begin + chunk);
  __shared__ double a[tile][panel + 1], b[tile][panel + 1];
  double c00 = 0, c01 = 0, c10 = 0, c11 = 0;
  for (int offset = begin; offset < end; offset += panel) {
    for (int element = lane; element < tile * panel; element += 256) {
      const int row = element / panel, column = element % panel;
      const int i = tile_i * tile + row, j = tile_j * tile + row;
      a[row][column] = i < n && offset + column < end
          ? input[static_cast<std::size_t>(i) * reduction + offset + column] : 0;
      b[row][column] = j < n && offset + column < end
          ? input[static_cast<std::size_t>(j) * reduction + offset + column] : 0;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < panel; ++k) {
      const double a0 = a[y][k], a1 = a[y + 16][k];
      const double b0 = b[x][k], b1 = b[x + 16][k];
      c00 = fma(a0, b0, c00); c01 = fma(a0, b1, c01);
      c10 = fma(a1, b0, c10); c11 = fma(a1, b1, c11);
    }
    __syncthreads();
  }
  const double values[4] = {c00, c01, c10, c11};
#pragma unroll
  for (int ihalf = 0; ihalf < 2; ++ihalf) {
#pragma unroll
    for (int jhalf = 0; jhalf < 2; ++jhalf) {
      const int i = tile_i * tile + y + ihalf * 16;
      const int j = tile_j * tile + x + jhalf * 16;
      if (i < n && j < n && i >= j)
        partials[(static_cast<std::size_t>(blockIdx.z) * n + j) * n + i] =
            values[2 * ihalf + jhalf];
    }
  }
}

/** Read only initialized lower partials in a fixed ascending slice order.
 * Both output triangles use that same sum; weight is applied exactly once.
 */
__global__ void df_occupied_gram_reduce(
    std::size_t n, std::size_t splits, double weight,
    const double* partials, double* output) {
  const auto element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= n * n) return;
  const auto i = element % n, j = element / n;
  const auto pair = i >= j ? j * n + i : i * n + j;
  double value = 0;
  for (std::size_t split = 0; split < splits; ++split)
    value += partials[split * n * n + pair];
  output[element] = weight * value;
}
#endif
}  // namespace generativeqc::scf::generated
"""
