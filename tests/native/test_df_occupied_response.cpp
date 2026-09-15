/** Adversarial factor identity checks independent of molecular force parity.
 * A tiny zero-integral model makes every fallback force exactly zero while
 * the execution diagnostic proves which validated response route ran.
 */
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "molecule/basis.hpp"
#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_scf_state.hpp"
#include "scf/cuda_density_fitting_final_state.hpp"
#include "scf/cuda_df_gradient.hpp"

namespace {
using namespace vibeqc::scf;
void require(bool value, const std::string& detail) {
  if (!value) throw std::runtime_error(detail);
}
void check(cudaError_t error) { require(error == cudaSuccess, cudaGetErrorString(error)); }
/** Exercise automatic token inspection without allocating full 768-AO tensors.
 * An unresolved metric stops execution after selection, before the placeholder
 * scratch addresses can be used by a device consumer.
 */
void malformed_automatic_token() {
  setenv("VIBEQC_DF_RESPONSE_STORAGE", "auto", 1);
  setenv("VIBEQC_DF_RESPONSE_SPACE", "auto", 1);
  CudaDensityFittingJkPlan plan;
  plan.device_id = 0;
  plan.batch_size = 1;
  plan.nbf = plan.naux = plan.row_tile = plan.auxiliary_tile = 768;
  double unused = 0;
  plan.auxiliary_tile_values = plan.exchange_intermediate = plan.exchange_contributions = &unused;
  plan.metric_response_valid = {0};
  vibeqc::core::System orbital;
  orbital.atoms = {{1, {0, 0, 0}}};
  orbital.shells.assign(768, {0, 0, {{1, 1}}});
  const std::vector<DensityFittingDensityResponse> terms{{{}, 1, .25}};
  std::vector<double> derivative;
  std::string detail;
  CudaDfFinalStateToken empty;
  const auto status = execute_cuda_density_fitting_generated_force_response(
      &plan, 0, orbital, orbital, {}, {}, terms, 0, 4U << 20, 0, derivative, detail, nullptr,
      &empty);
  require(status == VIBEQC_STATUS_NUMERICAL_FAILURE &&
              detail == "DF metric rank crossing: retained/discarded subspaces are unresolved",
          "malformed automatic token did not reach the metric guard");
}
void lifecycle(bool uhf) {
  setenv("VIBEQC_DF_EXCHANGE", "occupied", 1);
  setenv("VIBEQC_DF_RESPONSE_STORAGE", "jk-scratch", 1);
  setenv("VIBEQC_DF_RESPONSE_SPACE", "occupied", 1);
  CudaDensityFittingJkPlan* raw = nullptr;
  std::vector<CudaDensityFittingMetricDiagnostic> diagnostics;
  std::string detail;
  require(create_cuda_density_fitting_jk_plan(0, 1, 2, 1, {1}, {0, 0, 0, 0}, 1e-10, 1, &raw,
                                              diagnostics, detail) == VIBEQC_STATUS_SUCCESS,
          detail);
  std::unique_ptr<CudaDensityFittingJkPlan, decltype(&destroy_cuda_density_fitting_jk_plan)> plan(
      raw, destroy_cuda_density_fitting_jk_plan);
  std::vector<double> density, beta;
  std::vector<CudaDensityFittingDeviceScfItem> records;
  const auto solve = [&] {
    const auto status =
        uhf ? run_cuda_density_fitting_uhf_device_scf(plan.get(), {-1, 0, 0, 2}, {1, 0, 0, 1},
                                                      {1, 0, 0, 0}, {0, 0, 0, 0}, {1}, {0}, {0}, 8,
                                                      1e-12, 1e-10, density, beta, records, detail)
            : run_cuda_density_fitting_rhf_device_scf(plan.get(), {-1, 0, 0, 2}, {1, 0, 0, 1},
                                                      {2, 0, 0, 0}, {1}, {0}, 8, 1e-12, 1e-10,
                                                      density, records, detail);
    require(status == VIBEQC_STATUS_SUCCESS && records[0].converged, detail);
  };
  solve();
  CudaDfFinalStateToken token;
  require(
      cuda_density_fitting_final_state_token(plan.get(), 0, token, detail) == VIBEQC_STATUS_SUCCESS,
      detail);
  vibeqc::core::System orbital;
  orbital.atoms = {{1, {0, 0, -0.7}}, {1, {0, 0, 0.7}}};
  orbital.shells = {{0, 0, {{1, 1}}}, {1, 0, {{1, 1}}}};
  auto auxiliary = orbital;
  auxiliary.shells.resize(1);
  require(vibeqc::molecule::validate_and_normalize(orbital, detail) == VIBEQC_STATUS_SUCCESS,
          detail);
  require(vibeqc::molecule::validate_and_normalize(auxiliary, detail) == VIBEQC_STATUS_SUCCESS,
          detail);
  const auto force = [&](const CudaDfFinalStateToken* expected, bool occupied) {
    auto total = density;
    if (uhf)
      for (std::size_t k = 0; k < total.size(); ++k) total[k] += beta[k];
    std::vector<DensityFittingDensityResponse> terms =
        uhf ? std::vector<DensityFittingDensityResponse>{{total, 1, 0},
                                                         {density, 0, .5},
                                                         {beta, 0, .5}}
            : std::vector<DensityFittingDensityResponse>{{density, 1, .25}};
    DfGradientResources resources;
    std::vector<double> derivative;
    const std::vector<double> raw_values(4), metric{1};
    const auto status = execute_cuda_density_fitting_generated_force_response(
        plan.get(), 0, orbital, auxiliary, raw_values, metric, terms, 0, 4U << 20, 0, derivative,
        detail, &resources, expected);
    require(status == VIBEQC_STATUS_SUCCESS, detail);
    require(resources.occupied_response == occupied, "wrong response factor selection");
    require(derivative == std::vector<double>(6), "zero-integral response changed");
  };
  force(&token, true);
  force(nullptr, false);  // Even identical external density has no authorization.
  for (unsigned fault = 0; fault < 6; ++fault) {
    auto stale = token;
    if (fault == 0) ++stale.identity.factor.basis;
    if (fault == 1) ++stale.identity.solve_epoch;
    if (fault == 2) ++stale.identity.factor.density_generation;
    if (fault == 3) ++stale.identity.factor.orbital_generation;
    if (fault == 4) stale.identity.occupied[0] = 0;
    if (fault == 5) stale.identity.model.metric_relative_threshold *= 2;
    force(&stale, false);
  }
  density[0] += 1e-13;
  force(&token, false);
  density[0] -= 1e-13;
  // Restore through a solve instead of assuming floating addition reverses.
  solve();
  force(&token, false);  // Epoch changes even when D and dimensions are identical.
  require(
      cuda_density_fitting_final_state_token(plan.get(), 0, token, detail) == VIBEQC_STATUS_SUCCESS,
      detail);
  force(&token, true);
  auto* state = static_cast<cuda_df::PersistentScfState*>(plan->persistent_scf_state);
  check(cudaMemsetAsync(state->d_alpha_factor_generation, 0, sizeof(std::uint32_t), plan->stream));
  check(cudaStreamSynchronize(plan->stream));
  force(&token, false);
}
}  // namespace
int main() {
  try {
    malformed_automatic_token();
    lifecycle(false);
    lifecycle(true);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
