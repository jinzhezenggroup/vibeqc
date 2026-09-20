#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "cc/solver.hpp"
#include "tensor/cuda_runtime.cuh"

namespace vibeqc::cc::generated {

struct CudaState {
  std::size_t o{}, v{};
  cudaStream_t stream{};
  double *foo{}, *fov{}, *fvv{};
  double *ovov{}, *ovvo{}, *oovv{}, *ovvv{}, *ovoo{}, *oooo{}, *vvvv{};
  double *d1{}, *d2{}, *t1{}, *t2{};
  double *iteration_arena{}, *replay_arena{};
  int* error{};
};

struct DeviceIterationOutputs {
  double* energy{};
  double* r1{};
  double* r2{};
  double* next_t1{};
  double* next_t2{};
};
struct DeviceReplayOutputs {
  double* energy{};
  double* r1{};
  double* r2{};
};

DeviceIterationOutputs run_iteration_cuda(CudaState& state);
DeviceReplayOutputs run_replay_cuda(CudaState& state);

}  // namespace vibeqc::cc::generated
