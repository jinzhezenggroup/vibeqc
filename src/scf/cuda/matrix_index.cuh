#pragma once

#include <cstddef>

namespace vibeqc::scf::cuda_execution {

/** Common column-major indexing used by native matrix kernels. */
__device__ inline std::size_t matrix_index(std::size_t row, std::size_t column, std::size_t n) {
  // CUDA dense matrices are column-major so they can be submitted directly to
  // cuSOLVER without iteration-level transposes.
  return row + column * n;
}

}  // namespace vibeqc::scf::cuda_execution
