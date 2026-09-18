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
__global__ void symmetrize_metrics_kernel(std::size_t dimension, double* metrics) {
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

__global__ void scale_metric_projection_kernel(std::size_t dimension, std::size_t elements,
                                               const double* eigenvalues, bool square_root,
                                               double* projected) {
  const auto k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= elements) return;
  const auto value = eigenvalues[k % dimension];
  projected[k] /= square_root ? sqrt(value) : value;
}

void launch_symmetrize_metrics_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                      cudaStream_t stream, std::size_t dimension, double* metrics) {
  symmetrize_metrics_kernel<<<grid, block, shared_bytes, stream>>>(dimension, metrics);
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
  const auto elements = dimension * pairs;
  scale_metric_projection_kernel<<<static_cast<unsigned>((elements + 255) / 256), 256, 0, stream>>>(
      dimension, elements, eigenvalues, square_root, projected);
}
}  // namespace vibeqc::scf::cuda_df
