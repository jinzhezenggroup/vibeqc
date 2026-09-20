#include <math_constants.h>

#include <cmath>

#include "scf/cuda/matrix_index.cuh"
#include "scf/cuda/scf_matrix_kernels.hpp"

namespace vibeqc::scf::cuda_execution {

__global__ void copy_matrix_kernel(std::size_t elements, const double* source,
                                   double* destination) {
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element < elements) destination[element] = source[element];
}

/** Copy complete per-system matrices selected by a device-resident mask. */
__global__ void copy_selected_matrices_kernel(std::int32_t batch_size,
                                              std::int32_t matrices_per_system, std::int32_t nbf,
                                              const std::uint8_t* selected, const double* source,
                                              double* destination) {
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * nbf;
  const std::size_t matrix_count = static_cast<std::size_t>(batch_size) * matrices_per_system;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= matrix_count * matrix_size) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system = state / static_cast<std::size_t>(matrices_per_system);
  if (selected[system] != 0) destination[element] = source[element];
}

__global__ void extract_matrix_diagonals_kernel(std::int32_t batch_size,
                                                std::int32_t matrices_per_system, std::int32_t nbf,
                                                const std::uint8_t* selected,
                                                const double* matrices, double* diagonals) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_count = static_cast<std::size_t>(batch_size) * matrices_per_system;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= matrix_count * n) return;
  const std::size_t state = element / n;
  const std::size_t system = state / static_cast<std::size_t>(matrices_per_system);
  if (selected[system] == 0) return;
  const std::size_t column = element % n;
  const std::size_t matrix_offset = state * n * n;
  diagonals[element] = matrices[matrix_offset + matrix_index(column, column, n)];
}

__global__ void build_orthogonalizer_kernel(std::int32_t batch_size, std::int32_t nbf,
                                            const double* eigenvectors, const double* eigenvalues,
                                            const std::uint8_t* active, double* orthogonalizer,
                                            std::uint8_t* failed) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch_size) * matrix_size) return;
  const std::int32_t system = static_cast<std::int32_t>(element / matrix_size);
  if (active[system] == 0) return;
  const std::size_t local = element % matrix_size;
  const std::size_t row = local % n;
  const std::size_t column = local / n;
  const double* vectors = eigenvectors + static_cast<std::size_t>(system) * matrix_size;
  const double* values = eigenvalues + static_cast<std::size_t>(system) * n;
  double result = 0.0;
  for (std::size_t orbital = 0; orbital < n; ++orbital) {
    if (!(values[orbital] > 1.0e-10)) {
      failed[system] = 1;
      return;
    }
    result += vectors[matrix_index(row, orbital, n)] * vectors[matrix_index(column, orbital, n)] /
              sqrt(values[orbital]);
  }
  orthogonalizer[element] = result;
}

__global__ void matrix_product_kernel(std::int32_t batch_size, std::int32_t nbf, const double* left,
                                      bool transpose_left, const double* right,
                                      const std::uint8_t* active, double* output, double scale) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch_size) * matrix_size) return;
  const std::int32_t system = static_cast<std::int32_t>(element / matrix_size);
  if (active != nullptr && active[system] == 0) return;
  const std::size_t local = element % matrix_size;
  const std::size_t row = local % n;
  const std::size_t column = local / n;
  const std::size_t offset = static_cast<std::size_t>(system) * matrix_size;
  double value = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    const std::size_t left_index =
        transpose_left ? matrix_index(k, row, n) : matrix_index(row, k, n);
    value += left[offset + left_index] * right[offset + matrix_index(k, column, n)];
  }
  output[element] = scale * value;
}

__global__ void broadcast_spin_matrix_kernel(std::int32_t batch_size, std::int32_t spin_count,
                                             std::int32_t nbf, const double* physical_matrices,
                                             const std::uint8_t* active, double* spin_matrices) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t spin_elements = static_cast<std::size_t>(batch_size) * spin_count * matrix_size;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= spin_elements) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system = state / static_cast<std::size_t>(spin_count);
  if (active != nullptr && active[system] == 0) return;
  spin_matrices[element] = physical_matrices[system * matrix_size + element % matrix_size];
}

__global__ void spin_matrix_product_kernel(std::int32_t batch_size, std::int32_t spin_count,
                                           std::int32_t nbf, const double* left, bool left_is_spin,
                                           bool transpose_left, const double* right,
                                           bool right_is_spin, const std::uint8_t* active,
                                           double* output) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t state_count = static_cast<std::size_t>(batch_size) * spin_count;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= state_count * matrix_size) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system = state / static_cast<std::size_t>(spin_count);
  if (active != nullptr && active[system] == 0) return;
  const std::size_t local = element % matrix_size;
  const std::size_t row = local % n;
  const std::size_t column = local / n;
  const std::size_t left_offset = (left_is_spin ? state : system) * matrix_size;
  const std::size_t right_offset = (right_is_spin ? state : system) * matrix_size;
  double value = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    const std::size_t left_index =
        transpose_left ? matrix_index(k, row, n) : matrix_index(row, k, n);
    value += left[left_offset + left_index] * right[right_offset + matrix_index(k, column, n)];
  }
  output[element] = value;
}

__global__ void clear_active_matrices_kernel(std::int32_t batch_size,
                                             std::int32_t matrices_per_system, std::int32_t nbf,
                                             const std::uint8_t* active, double* matrices) {
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * nbf;
  const std::size_t matrix_count = static_cast<std::size_t>(batch_size) * matrices_per_system;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= matrix_count * matrix_size) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system = state / static_cast<std::size_t>(matrices_per_system);
  if (active[system] != 0) matrices[element] = 0.0;
}

/** Subtract the second GEMM product from the first in a batched matrix set. */
__global__ void subtract_matrix_batches_kernel(std::int32_t batch_size,
                                               std::int32_t matrices_per_system, std::int32_t nbf,
                                               const double* subtract, const std::uint8_t* active,
                                               double* minuend) {
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * nbf;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(batch_size) *
                            static_cast<std::size_t>(matrices_per_system) * matrix_size;
  if (element >= total) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system = state / static_cast<std::size_t>(matrices_per_system);
  if (active != nullptr && active[system] == 0) return;
  minuend[element] -= subtract[element];
}

void launch_copy_matrix_kernel(dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
                               std::size_t elements, const double* source, double* destination) {
  copy_matrix_kernel<<<grid, block, shared_bytes, stream>>>(elements, source, destination);
}

void launch_copy_selected_matrices_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                          cudaStream_t stream, std::int32_t batch_size,
                                          std::int32_t matrices_per_system, std::int32_t nbf,
                                          const std::uint8_t* selected, const double* source,
                                          double* destination) {
  copy_selected_matrices_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, matrices_per_system, nbf, selected, source, destination);
}

void launch_extract_matrix_diagonals_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                            cudaStream_t stream, std::int32_t batch_size,
                                            std::int32_t matrices_per_system, std::int32_t nbf,
                                            const std::uint8_t* selected, const double* matrices,
                                            double* diagonals) {
  extract_matrix_diagonals_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, matrices_per_system, nbf, selected, matrices, diagonals);
}

void launch_build_orthogonalizer_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                        cudaStream_t stream, std::int32_t batch_size,
                                        std::int32_t nbf, const double* eigenvectors,
                                        const double* eigenvalues, const std::uint8_t* active,
                                        double* orthogonalizer, std::uint8_t* failed) {
  build_orthogonalizer_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, nbf, eigenvectors, eigenvalues, active, orthogonalizer, failed);
}

void launch_matrix_product_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                  cudaStream_t stream, std::int32_t batch_size, std::int32_t nbf,
                                  const double* left, bool transpose_left, const double* right,
                                  const std::uint8_t* active, double* output, double scale) {
  matrix_product_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, nbf, left, transpose_left, right, active, output, scale);
}

void launch_broadcast_spin_matrix_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                         cudaStream_t stream, std::int32_t batch_size,
                                         std::int32_t spin_count, std::int32_t nbf,
                                         const double* physical_matrices,
                                         const std::uint8_t* active, double* spin_matrices) {
  broadcast_spin_matrix_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, spin_count, nbf, physical_matrices, active, spin_matrices);
}

void launch_spin_matrix_product_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                       cudaStream_t stream, std::int32_t batch_size,
                                       std::int32_t spin_count, std::int32_t nbf,
                                       const double* left, bool left_is_spin, bool transpose_left,
                                       const double* right, bool right_is_spin,
                                       const std::uint8_t* active, double* output) {
  spin_matrix_product_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, spin_count, nbf, left, left_is_spin, transpose_left, right, right_is_spin, active,
      output);
}

void launch_clear_active_matrices_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                         cudaStream_t stream, std::int32_t batch_size,
                                         std::int32_t matrices_per_system, std::int32_t nbf,
                                         const std::uint8_t* active, double* matrices) {
  clear_active_matrices_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, matrices_per_system, nbf, active, matrices);
}

void launch_subtract_matrix_batches_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                           cudaStream_t stream, std::int32_t batch_size,
                                           std::int32_t matrices_per_system, std::int32_t nbf,
                                           const double* subtract, const std::uint8_t* active,
                                           double* minuend) {
  subtract_matrix_batches_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, matrices_per_system, nbf, subtract, active, minuend);
}

}  // namespace vibeqc::scf::cuda_execution
