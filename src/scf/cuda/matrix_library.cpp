#include "scf/cuda/matrix_library.hpp"

#include <cstddef>

#include "scf/cuda/launch_geometry.hpp"
#include "scf/cuda/runtime_support.hpp"
#include "scf/cuda/scf_matrix_kernels.hpp"

namespace vibeqc::scf::cuda_execution {

vibeqc_status launch_matrix_product(MatrixLibraryResources resources, int batch_size, int nbf,
                                    const double* left, bool transpose_left, const double* right,
                                    const std::uint8_t* active, double* output, bool use_cublas,
                                    double scale) {
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * static_cast<std::size_t>(nbf);
  if (!use_cublas) {
    const std::size_t elements = static_cast<std::size_t>(batch_size) * matrix_size;
    const unsigned blocks = static_cast<unsigned>((elements + kCaptureSafeKernelThreads - 1) /
                                                  kCaptureSafeKernelThreads);
    launch_matrix_product_kernel(blocks, kCaptureSafeKernelThreads, 0, resources.stream_,
                                 batch_size, nbf, left, transpose_left, right, active, output,
                                 scale);
    return cuda_status(cudaPeekAtLastError());
  }

  const double alpha = scale;
  const double beta = 0.0;
  const cublasOperation_t operation = transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N;
  return blas_status(cublasDgemmStridedBatched(
      resources.blas_, operation, CUBLAS_OP_N, nbf, nbf, nbf, &alpha, left, nbf,
      static_cast<long long>(matrix_size), right, nbf, static_cast<long long>(matrix_size), &beta,
      output, nbf, static_cast<long long>(matrix_size), batch_size));
}

/**
 * Multiply system-major spin matrices while broadcasting physical operands.
 *
 * A physical matrix repeats for alpha and beta, which is not one constant
 * stride over the interleaved state array. One strided-batched GEMM per spin
 * preserves the existing [system][spin][matrix] storage without pointer lists.
 */
vibeqc_status launch_spin_matrix_product(MatrixLibraryResources resources, int batch_size,
                                         int spin_count, int nbf, const double* left,
                                         bool left_is_spin, bool transpose_left,
                                         const double* right, bool right_is_spin,
                                         const std::uint8_t* active, double* output,
                                         bool use_cublas) {
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * static_cast<std::size_t>(nbf);
  if (!use_cublas) {
    const std::size_t elements =
        static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(spin_count) * matrix_size;
    const unsigned blocks = static_cast<unsigned>((elements + kCaptureSafeKernelThreads - 1) /
                                                  kCaptureSafeKernelThreads);
    launch_spin_matrix_product_kernel(blocks, kCaptureSafeKernelThreads, 0, resources.stream_,
                                      batch_size, spin_count, nbf, left, left_is_spin,
                                      transpose_left, right, right_is_spin, active, output);
    return cuda_status(cudaPeekAtLastError());
  }

  const double alpha = 1.0;
  const double beta = 0.0;
  const cublasOperation_t operation = transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N;
  const long long physical_stride = static_cast<long long>(matrix_size);
  const long long spin_stride =
      static_cast<long long>(matrix_size * static_cast<std::size_t>(spin_count));
  for (int spin = 0; spin < spin_count; ++spin) {
    const std::size_t spin_offset = static_cast<std::size_t>(spin) * matrix_size;
    const double* spin_left = left + (left_is_spin ? spin_offset : 0);
    const double* spin_right = right + (right_is_spin ? spin_offset : 0);
    const cublasStatus_t status =
        cublasDgemmStridedBatched(resources.blas_, operation, CUBLAS_OP_N, nbf, nbf, nbf, &alpha,
                                  spin_left, nbf, left_is_spin ? spin_stride : physical_stride,
                                  spin_right, nbf, right_is_spin ? spin_stride : physical_stride,
                                  &beta, output + spin_offset, nbf, spin_stride, batch_size);
    if (status != CUBLAS_STATUS_SUCCESS) return blas_status(status);
  }
  return VIBEQC_STATUS_SUCCESS;
}

}  // namespace vibeqc::scf::cuda_execution
