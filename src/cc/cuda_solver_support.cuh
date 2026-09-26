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
  double *iteration_arena{}, *replay_arena{}, *response_arena{};
  double *bar_correlation_energy{}, *bar_singles_residual{}, *bar_doubles_residual{};
  double *bar_foo{}, *bar_fov{}, *bar_fvv{};
  double *bar_ovov{}, *bar_ovvo{}, *bar_oovv{}, *bar_ovvv{}, *bar_ovoo{}, *bar_oooo{}, *bar_vvvv{};
  double *bar_reference_electronic_energy{}, *bar_fock{}, *d_rotation{};
  double *density{}, *g{}, *h{}, *rotation{};
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
struct DeviceLambdaOutputs {
  double* t1{};
  double* t2{};
};
struct DeviceParameterOutput {
  double* values{};
};
struct DeviceHamiltonianOutputs {
  double* hcore{};
  double* eri{};
  double* overlap{};
  double* rotation_gradient{};
  double* stationarity{};
  double* orbital_rhs{};
};
struct DeviceOrbitalJvpOutput {
  double* d_fov{};
};

DeviceIterationOutputs run_iteration_cuda(CudaState& state);
DeviceReplayOutputs run_replay_cuda(CudaState& state);
DeviceLambdaOutputs run_lambda_rhs_cuda(CudaState& state);
DeviceLambdaOutputs run_lambda_transpose_cuda(CudaState& state);
DeviceLambdaOutputs run_lambda_independent_rhs_cuda(CudaState& state);
DeviceLambdaOutputs run_lambda_independent_transpose_cuda(CudaState& state);
DeviceParameterOutput run_parameter_foo_cuda(CudaState& state);
DeviceParameterOutput run_parameter_fov_cuda(CudaState& state);
DeviceParameterOutput run_parameter_fvv_cuda(CudaState& state);
DeviceParameterOutput run_parameter_ovov_cuda(CudaState& state);
DeviceParameterOutput run_parameter_ovvo_cuda(CudaState& state);
DeviceParameterOutput run_parameter_oovv_cuda(CudaState& state);
DeviceParameterOutput run_parameter_ovvv_cuda(CudaState& state);
DeviceParameterOutput run_parameter_ovoo_cuda(CudaState& state);
DeviceParameterOutput run_parameter_oooo_cuda(CudaState& state);
DeviceParameterOutput run_parameter_vvvv_cuda(CudaState& state);
DeviceHamiltonianOutputs run_hamiltonian_weights_cuda(CudaState& state);
DeviceHamiltonianOutputs run_fock_weights_cuda(CudaState& state);
DeviceOrbitalJvpOutput run_orbital_jvp_cuda(CudaState& state);

}  // namespace vibeqc::cc::generated
