#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <optional>

#include "scf/cuda_density_fitting.hpp"
#include "scf/cuda_density_fitting_final_state.hpp"

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
  // Survives persistent-state rebuilds, so topology/occupation changes cannot
  // restart the epoch and authorize a previous device solve's final frame.
  std::uint64_t final_state_solve_epoch{};
  // Frozen at creation: lazy SCF factors may only use a plan that reserved
  // their capacity. High-level caches rebuild when the policy changes.
  bool occupied_scf_reserved{};
  // HF cache identity: property changes alter the value/response partition.
  // Direct fixed-tile consumers keep the default compatibility value zero.
  std::size_t scf_value_budget_bytes{};
  // High-level SCF planning includes this history before selecting K panels.
  // Compatibility fixed-point callers retain zero and allocate no DIIS state.
  unsigned scf_diis_history{};
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
  // Resident occupied K uses one scratch tensor; dense K fits its Q panels
  // in the other two. The former contribution buffer retains raw A[Q,mu,nu],
  // including discarded metric directions, from setup until destruction.
  // Never reuse it as J/K output under the resident exchange policy.
  bool resident_raw_valid{};
  // An exclusive lease on U[mu,i,Q] in auxiliary_tile_values. Only the
  // validated final physical RHF K publishes it. Every scratch writer and
  // new solve revokes it before submission, including unsuccessful attempts.
  std::optional<CudaDfFinalStateToken> final_projection_token;
  // Reconstructing raw projections from whitened U is valid only when every
  // metric direction survived the forward cutoff. Rank boundaries still use
  // the existing spectral-response validity check.
  std::vector<std::uint8_t> metric_full_rank;
  bool resident_exchange_enabled{};
  bool triangular_exchange{};
  bool flat_dense_exchange{};
  bool cooperative_diis{};
  // The prepared HF owner binds the exact immutable host allocations from
  // which setup copied A. Their vector storage survives moves into/out of the
  // prepared cache. Standalone tensor callers have no such source lease and
  // retain upload semantics. No host tensor is copied for this binding.
  const double* response_host_raw{};
  const core::Atom *response_orbital_atoms{}, *response_auxiliary_atoms{};
  const core::Shell *response_orbital_shells{}, *response_auxiliary_shells{};
  vibeqc_basis_representation response_orbital_representation{},
      response_auxiliary_representation{};
  double* exchange_tile_output{};
  double* exchange_density_column_major{};
  // Every value plan keeps its forward eigensystem for the spectral force
  // reverse map. Ownership moves out of setup; no extra setup allocation or
  // host copy of M/M+ is needed, including for rank-deficient metrics.
  double* metric_eigenvectors{};
  double* metric_eigenvalues{};
  std::vector<std::uint8_t> metric_response_valid;
  // True means B is regenerated, not merely that contraction Q is partial.
  // Retained generated B can share bounded full-AO-row contraction scratch.
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
  // Ordinary AO setup/final eigen operations share the existing handles while
  // retaining one bounded scratch frame, separate from captured SCF state.
  void* ordinary_eigensystem{};
  // Serialized FP64 final validation/projection storage, separate from graphs.
  void* final_validation{};
};

}  // namespace vibeqc::scf
