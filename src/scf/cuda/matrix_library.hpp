#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstdint>

#include "vibeqc/vibeqc.h"

namespace vibeqc::scf::cuda_execution {

/** Non-owning matrix-library execution context, independent of provider/plan storage. */
struct MatrixLibraryResources {
  cudaStream_t stream_{};
  cublasHandle_t blas_{};
};

/** Use the resolved native/library route, preserving masks and column-major strides.
 * Spin products broadcast physical operands through one strided GEMM per spin.
 * The caller owns every input/output allocation and the borrowed library handles.
 * The optional scale is applied by both the native and cuBLAS routes, allowing
 * occupation normalization without a separate matrix pass.
 */
vibeqc_status launch_matrix_product(MatrixLibraryResources resources, int batch_size, int nbf,
                                    const double* left, bool transpose_left, const double* right,
                                    const std::uint8_t* active, double* output, bool use_cublas,
                                    double scale = 1.0);

vibeqc_status launch_spin_matrix_product(MatrixLibraryResources resources, int batch_size,
                                         int spin_count, int nbf, const double* left,
                                         bool left_is_spin, bool transpose_left,
                                         const double* right, bool right_is_spin,
                                         const std::uint8_t* active, double* output,
                                         bool use_cublas);

}  // namespace vibeqc::scf::cuda_execution
