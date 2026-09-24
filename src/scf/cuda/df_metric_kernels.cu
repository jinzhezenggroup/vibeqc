#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "scf/cuda/df_metric_kernels.hpp"

namespace vibeqc::scf::cuda_df {

// Existing DF arithmetic and reduction order; host orchestration compiles separately.
__global__ void symmetrize_metrics_kernel(std::size_t dimension, std::size_t tiles,
                                          double* metrics) {
  // Enumerate only authoritative upper-triangle tiles. One lane decodes the
  // compact tile pair; every lane then executes the unchanged elementwise
  // average/mirror operation. Diagonal tiles still reject their lower half.
  __shared__ std::size_t tile_row, tile_column;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    const auto pair = static_cast<std::size_t>(blockIdx.x);
    std::size_t low = 0, high = tiles;
    while (low + 1 < high) {
      const auto mid = (low + high) / 2;
      if (mid * (mid + 1) / 2 <= pair)
        low = mid;
      else
        high = mid;
    }
    tile_column = low;
    tile_row = pair - low * (low + 1) / 2;
  }
  __syncthreads();
  const std::size_t column = tile_column * blockDim.x + threadIdx.x;
  const std::size_t row = tile_row * blockDim.y + threadIdx.y;
  const std::size_t system = blockIdx.z;
  if (row >= dimension || column >= dimension || row > column) return;
  const std::size_t offset = system * dimension * dimension;
  const std::size_t first = offset + row * dimension + column;
  const std::size_t second = offset + column * dimension + row;
  const double symmetric = 0.5 * (metrics[first] + metrics[second]);
  metrics[first] = symmetric;
  metrics[second] = symmetric;
}

__global__ void scale_eigenvectors_kernel(std::size_t matrix_elements, std::size_t dimension,
                                          const double* eigenvectors, const double* scales,
                                          double* scaled_eigenvectors) {
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= matrix_elements) return;
  const std::size_t local = element % (dimension * dimension);
  const std::size_t system = element / (dimension * dimension);
  const std::size_t column = local / dimension;
  scaled_eigenvectors[element] = eigenvectors[element] * scales[system * dimension + column];
}

/** Scale a bounded pair stripe while reusing the eigendirection denominator.
 *
 * Projections are column-major [eigendirection,pair]. Adjacent threads own
 * adjacent eigendirections, so each pair iteration remains coalesced. The
 * second grid dimension supplies enough independent pair stripes to occupy the
 * device without recomputing sqrt(eigenvalue) for every projected element.
 * There is no reduction or cross-thread arithmetic: every output element keeps
 * the exact existing division by the same FP64 denominator.
 */
__global__ void scale_metric_projection_kernel(std::size_t dimension, std::size_t pairs,
                                               const double* eigenvalues, bool square_root,
                                               double* projected) {
  const auto direction = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto group = static_cast<std::size_t>(blockIdx.y);
  if (direction >= dimension || group >= pairs) return;
  const auto value = eigenvalues[direction];
  const auto denominator = square_root ? sqrt(value) : value;
  for (auto pair = group; pair < pairs; pair += static_cast<std::size_t>(gridDim.y))
    projected[pair * dimension + direction] /= denominator;
}

// Preserve the caller's original domain when rectangular tiles or a clipped
// launch cannot be represented by the compact square-tile enumeration.
__global__ void symmetrize_metrics_fallback_kernel(std::size_t dimension, double* metrics) {
  const std::size_t column = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t row = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  const std::size_t system = blockIdx.z;
  if (row >= dimension || column >= dimension || row > column) return;
  const std::size_t offset = system * dimension * dimension;
  const std::size_t first = offset + row * dimension + column;
  const std::size_t second = offset + column * dimension + row;
  const double symmetric = 0.5 * (metrics[first] + metrics[second]);
  metrics[first] = symmetric;
  metrics[second] = symmetric;
}

void launch_symmetrize_metrics_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                      cudaStream_t stream, std::size_t dimension, double* metrics) {
  // Widen before multiplying, including on a 32-bit host. The compact grid
  // must fit CUDA's x dimension and preserve both axes of the original domain.
  const auto tiles = static_cast<std::uint64_t>(grid.x);
  const auto tile_pairs = tiles * (tiles + 1) / 2;
  if (!grid.x || grid.x != grid.y || !block.x || block.x != block.y || block.z != 1 ||
      tile_pairs > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    symmetrize_metrics_fallback_kernel<<<grid, block, shared_bytes, stream>>>(dimension, metrics);
    return;
  }
  symmetrize_metrics_kernel<<<dim3(static_cast<unsigned>(tile_pairs), 1, grid.z), block,
                              shared_bytes, stream>>>(dimension, tiles, metrics);
}
void launch_scale_eigenvectors_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                      cudaStream_t stream, std::size_t matrix_elements,
                                      std::size_t dimension, const double* eigenvectors,
                                      const double* scales, double* scaled_eigenvectors) {
  scale_eigenvectors_kernel<<<grid, block, shared_bytes, stream>>>(
      matrix_elements, dimension, eigenvectors, scales, scaled_eigenvectors);
}

void launch_scale_metric_projection(cudaStream_t stream, std::size_t dimension, std::size_t pairs,
                                    const double* eigenvalues, bool square_root,
                                    double* projected) {
  constexpr std::size_t kPairGroups = 32;
  const auto groups = std::min(pairs, kPairGroups);
  scale_metric_projection_kernel<<<dim3(static_cast<unsigned>((dimension + 255) / 256),
                                        static_cast<unsigned>(groups)),
                                   256, 0, stream>>>(dimension, pairs, eigenvalues, square_root,
                                                     projected);
}
}  // namespace vibeqc::scf::cuda_df
