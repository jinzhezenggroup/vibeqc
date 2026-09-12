#pragma once
#include <cstddef>

#include "tensor/metrics.hpp"
namespace vibeqc::mp2::generated {
using CudaCreate = int (*)(int, void**, char*, std::size_t);
using CudaDestroy = void (*)(void*);
using CudaRun = int (*)(void*, const double*, const double*, double, double, const double*,
                        const double*, double*, vibeqc_tensor::Metrics*, char*, std::size_t);
struct CudaPlan {
  CudaCreate create;
  CudaDestroy destroy;
  CudaRun run;
  std::size_t numeric_bytes;
  std::size_t device_bytes;
  const char* equation_hash;
};
CudaPlan cuda_plan(unsigned tile, int device);
}  // namespace vibeqc::mp2::generated
