#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_plan_setup.hpp"
#include "scf/cuda/df_scf_state.hpp"

namespace vibeqc::scf::cuda_df {

std::uint64_t next_factor_basis_identity() noexcept {
  static std::atomic<std::uint64_t> next{1};
  // A wrapped counter must never recycle an identity. Zero is ineligible;
  // saturation conservatively disables the path for future plans.
  auto value = next.load(std::memory_order_relaxed);
  while (value != 0) {
    if (next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed)) return value;
  }
  return 0;
}

void release(CudaDensityFittingJkPlan& plan) noexcept {
  if (plan.device_id >= 0) (void)cudaSetDevice(plan.device_id);
  destroy_persistent_scf_state(plan.persistent_scf_state);
  destroy_cuda_density_fitting_integral_source(plan.integral_source);
  (void)runtime::resource_cuda_free(plan.inverse_square_roots);
  (void)runtime::resource_cuda_free(plan.metric_eigenvectors);
  (void)runtime::resource_cuda_free(plan.metric_eigenvalues);
  (void)runtime::resource_cuda_free(plan.three_center);
  (void)runtime::resource_cuda_free(plan.primary_density);
  (void)runtime::resource_cuda_free(plan.secondary_density);
  (void)runtime::resource_cuda_free(plan.total_density);
  (void)runtime::resource_cuda_free(plan.auxiliary_density);
  (void)runtime::resource_cuda_free(plan.coulomb);
  (void)runtime::resource_cuda_free(plan.alpha_exchange);
  (void)runtime::resource_cuda_free(plan.beta_exchange);
  (void)runtime::resource_cuda_free(plan.auxiliary_tile_values);
  (void)runtime::resource_cuda_free(plan.exchange_intermediate);
  (void)runtime::resource_cuda_free(plan.exchange_contributions);
  (void)runtime::resource_cuda_free(plan.exchange_tile_output);
  (void)runtime::resource_cuda_free(plan.exchange_density_column_major);
  if (plan.solver_parameters != nullptr) {
    (void)cusolverDnDestroyParams(plan.solver_parameters);
  }
  if (plan.solver != nullptr) (void)cusolverDnDestroy(plan.solver);
  if (plan.blas != nullptr) (void)cublasDestroy(plan.blas);
  if (plan.stream != nullptr) (void)cudaStreamDestroy(plan.stream);
  plan = {};
}

vibeqc_status fail_plan(CudaDensityFittingJkPlan* plan, vibeqc_status status) {
  if (plan != nullptr) {
    release(*plan);
    delete plan;
  }
  return status;
}

void destroy_persistent_scf_state(void*& opaque) noexcept {
  delete static_cast<PersistentScfState*>(opaque);
  opaque = nullptr;
}

}  // namespace vibeqc::scf::cuda_df
