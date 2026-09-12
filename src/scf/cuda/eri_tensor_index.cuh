#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

// Retained eri tensor index contraction helpers.
// Borrow immutable metadata and density/output views; host plans own lifetime.

namespace vibeqc::scf::cuda_execution {

__device__ inline std::size_t eri_index(std::size_t i, std::size_t j, std::size_t k, std::size_t l,
                                        std::size_t n) {
  return ((i * n + j) * n + k) * n + l;
}

}  // namespace vibeqc::scf::cuda_execution
