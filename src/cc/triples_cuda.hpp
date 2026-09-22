#pragma once

#include <cstddef>

namespace vibeqc::cc::triples {

struct CudaResult {
  double energy{};
  double minimum_absolute_denominator{};
  std::size_t virtual_triples{};
  std::size_t workspace_bytes{};
};

#if VIBEQC_HAS_CUDA
CudaResult evaluate_cuda(std::size_t o, std::size_t v, const double* ovvv, const double* ovoo,
                         const double* ovov, const double* fov, const double* t1, const double* t2,
                         const double* eps_o, const double* eps_v, double denominator_threshold,
                         std::size_t max_bytes, int device);
#endif

}  // namespace vibeqc::cc::triples
