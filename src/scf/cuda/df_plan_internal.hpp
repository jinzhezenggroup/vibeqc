#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "scf/cuda_density_fitting.hpp"

namespace vibeqc::scf {

namespace cuda_df {
std::uint64_t next_factor_basis_identity() noexcept;
}

/** Private storage owner shared by DF setup, J/K and response adapters.
 * Public callers retain the existing opaque handle. The integral source and
 * forward eigensystem have exactly one owner; execution functions borrow them.
 */
struct CudaDensityFittingJkPlan {
  // Process-unique lifetime token binds orbital factors to this immutable
  // geometry/basis/metric source; dimensions and recycled addresses cannot.
  std::uint64_t factor_basis_identity{cuda_df::next_factor_basis_identity()};
  int device_id{-1};
  double metric_relative_threshold{};
  std::size_t batch_size{};
  std::size_t nbf{};
  std::size_t naux{};
  std::size_t matrix_elements{};
  std::size_t tensor_elements_per_system{};
  std::size_t auxiliary_tile{};
  std::size_t ao_pair_tile{};
  std::size_t row_tile{};
  cudaStream_t stream{};
  cublasHandle_t blas{};
  cusolverDnHandle_t solver{};
  cusolverDnParams_t solver_parameters{};
  double* three_center{};
  double* primary_density{};
  double* secondary_density{};
  double* total_density{};
  double* auxiliary_density{};
  double* coulomb{};
  double* alpha_exchange{};
  double* beta_exchange{};
  double* auxiliary_tile_values{};
  double* exchange_intermediate{};
  double* exchange_contributions{};
  double* exchange_tile_output{};
  double* exchange_density_column_major{};
  // The source plan keeps its forward eigensystem for the spectral force
  // reverse map. Ownership moves out of setup; no extra setup allocation or
  // host copy of M/M+ is needed, including for rank-deficient metrics.
  double* metric_eigenvectors{};
  double* metric_eigenvalues{};
  std::vector<std::uint8_t> metric_response_valid;
  // Partial tiles stream values; full tiles permit source-backed residency.
  // The source and its metric policy are immutable for this plan's lifetime.
  // X is shared by materialization/J/K and the spectral force response.
  bool streamed{};
  CudaDensityFittingIntegralSource* integral_source{};
  double* inverse_square_roots{};
  std::vector<double> streamed_raw_three_center;
  std::vector<double> streamed_inverse_square_roots;
  // Opaque persistent SCF state.  The definition lives with the device
  // solver state owner so this private plan layout never exposes CUDA graph
  // or cuSOLVER implementation types to callers.
  void* persistent_scf_state{};
};

}  // namespace vibeqc::scf
