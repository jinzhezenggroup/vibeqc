#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "runtime/resource_cuda.cuh"

namespace vibeqc::scf::cuda_df {

/** Persistent replay owner retained by a prepared DF plan.
 * Inputs refresh per solve; solver workspaces and graph handles survive until
 * incompatible topology/options require rebuilding. Teardown selects its device.
 */
struct DeviceSolver {
  cusolverDnHandle_t handle{};
  syevjInfo_t jacobi{};
  cusolverDnParams_t parameters{};
  double* workspace{};
  void* host_workspace{};
  std::size_t workspace_bytes{};
  std::size_t host_workspace_bytes{};
  bool xsyev{};
  int lwork{};
  ~DeviceSolver() {
    (void)runtime::resource_cuda_free(workspace);
    std::free(host_workspace);
    if (parameters != nullptr) (void)cusolverDnDestroyParams(parameters);
    if (jacobi != nullptr) (void)cusolverDnDestroySyevjInfo(jacobi);
    if (handle != nullptr) (void)cusolverDnDestroy(handle);
  }
};

struct DeviceIterationGraph {
  int device_id{-1};
  cudaStream_t stream{};
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  void reset() noexcept {
    if (device_id >= 0) (void)cudaSetDevice(device_id);
    if (executable != nullptr) {
      (void)cudaGraphExecDestroy(executable);
      executable = nullptr;
    }
    if (graph != nullptr) {
      (void)cudaGraphDestroy(graph);
      graph = nullptr;
    }
  }
  ~DeviceIterationGraph() { reset(); }
};

struct PersistentScfState {
  int device_id{-1};
  bool unrestricted{};
  std::size_t batch_size{};
  std::size_t nbf{};
  std::size_t expected{};
  std::vector<void*> allocations;
  DeviceSolver solver;
  DeviceIterationGraph graph;
  bool graph_replay{};
  unsigned max_iterations{};
  double energy_tolerance{};
  double density_tolerance{};

  // A solve always seeds from its arbitrary input D with dense K. Only the
  // coefficients that constructed the next canonical density are retained.
  // Inactive items keep both D and factors. Occupation changes rebuild this
  // owner and its captured GEMM shapes; buffers never borrow d_temporary.
  bool occupied_exchange{};
  std::vector<std::int32_t> factor_alpha_ranks, factor_beta_ranks;
  std::size_t alpha_factor_rank{}, beta_factor_rank{};
  double* d_alpha_factor{};
  double* d_beta_factor{};
  std::uint32_t* d_alpha_factor_generation{};
  std::uint32_t* d_beta_factor_generation{};
  int* d_factor_error{};

  // Shared RHF state.
  double* d_hcore{};
  double* d_orthogonalizer{};
  double* d_density{};
  double* d_next_density{};
  double* d_fock{};
  double* d_temporary{};
  double* d_eigenvalues{};
  std::int32_t* d_occupied{};
  double* d_nuclear{};
  double* d_energy{};
  double* d_previous_energy{};
  double* d_energy_change{};
  double* d_density_rms{};
  std::uint8_t* d_active{};
  std::uint8_t* d_converged{};
  std::uint32_t* d_iterations{};
  int* d_info{};

  // Additional UHF state.
  double* d_alpha_density{};
  double* d_beta_density{};
  double* d_next_alpha{};
  double* d_next_beta{};
  double* d_alpha_fock{};
  double* d_beta_fock{};
  double* d_alpha_eigenvalues{};
  double* d_beta_eigenvalues{};
  std::int32_t* d_alpha_occupied{};
  std::int32_t* d_beta_occupied{};
  int* d_alpha_info{};
  int* d_beta_info{};

  ~PersistentScfState() {
    if (device_id >= 0) (void)cudaSetDevice(device_id);
    for (void* pointer : allocations) (void)runtime::resource_cuda_free(pointer);
  }
};

void destroy_persistent_scf_state(void*& opaque) noexcept;

}  // namespace vibeqc::scf::cuda_df
