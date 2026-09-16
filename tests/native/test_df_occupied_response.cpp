/** Adversarial factor identity checks independent of molecular force parity.
 * A tiny zero-integral model makes every fallback force exactly zero while
 * the execution diagnostic proves which validated response route ran.
 */
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_scf_state.hpp"
#include "scf/cuda_density_fitting_final_state.hpp"
#include "scf/cuda_density_fitting_integrals.hpp"
#include "scf/cuda_df_gradient.hpp"
#include "scf/density_fitting.hpp"

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

/** Physical raw-integral derivative oracle for packed storage. Unequal AO and
 * auxiliary counts plus one-Q scratch cover direct raw projection, bounded
 * projection, empty spin, and discarded metric directions with the same data.
 */
void packed_response(bool uhf, unsigned beta_occupied, std::size_t capacity, double cutoff,
                     std::int32_t occupied = 1, std::size_t qtile = 1) {
  setenv("VIBEQC_DF_EXCHANGE", "occupied", 1);
  setenv("VIBEQC_DF_RESPONSE_STORAGE", "auto", 1);
  setenv("VIBEQC_DF_RESPONSE_SPACE", "occupied", 1);
  setenv("VIBEQC_DF_FINAL_EXCHANGE", "occupied", 1);
  setenv("VIBEQC_DF_FINAL_PROJECTION", "reuse", 1);
  setenv("VIBEQC_DF_WEIGHTED_EXECUTION", "generic", 1);
  setenv("VIBEQC_DF_DERIVATIVE_PAIRS", "auto", 1);
  vibeqc::core::System orbital;
  orbital.atoms = {{1, {0, 0, -.7}}, {1, {0, 0, .7}}};
  orbital.shells = {{0, 0, {{1, 1}}}, {0, 0, {{.3, 1}}}, {1, 0, {{1, 1}}}, {1, 0, {{.3, 1}}}};
  auto auxiliary = orbital;
  auxiliary.shells.push_back({0, 0, {{3, 1}}});
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(orbital, detail) == VIBEQC_STATUS_SUCCESS,
          detail);
  require(vibeqc::molecule::validate_and_normalize(auxiliary, detail) == VIBEQC_STATUS_SUCCESS,
          detail);
  const auto oracle = vibeqc::integrals::build_density_fitting_integrals(orbital, auxiliary);
  CudaDensityFittingIntegralSource* source{};
  std::vector<double> metric;
  std::size_t n{}, a{};
  require(create_cuda_density_fitting_integral_source(0, {orbital}, {auxiliary}, &source, metric, n,
                                                      a, detail) == VIBEQC_STATUS_SUCCESS,
          detail);
  CudaDensityFittingJkPlan* raw{};
  std::vector<CudaDensityFittingMetricDiagnostic> diagnostics;
  require(create_cuda_density_fitting_jk_plan_from_source(
              0, &source, 1, n, a, metric, cutoff, qtile, n * n, &raw, diagnostics, detail, true,
              {DfPairStorage::SymmetricLower, capacity}) == VIBEQC_STATUS_SUCCESS,
          detail);
  std::unique_ptr<CudaDensityFittingJkPlan, decltype(&destroy_cuda_density_fitting_jk_plan)> plan(
      raw, destroy_cuda_density_fitting_jk_plan);
  if (cutoff > 1e-10)
    require(!plan->metric_full_rank[0], "packed truncation fixture retained all directions");
  std::vector<double> h(n * n), x(n * n), initial(n * n), initial_beta(n * n), density, beta;
  for (std::size_t i = 0; i < n; ++i) {
    h[i * n + i] = -20 + 15.0 * i;
    x[i * n + i] = 1;
  }
  for (std::int32_t i = 0; i < occupied; ++i) initial[i * n + i] = uhf ? 1 : 2;
  for (unsigned i = 0; i < beta_occupied; ++i) initial_beta[i * n + i] = 1;
  std::vector<CudaDensityFittingDeviceScfItem> records;
  const auto status =
      uhf ? run_cuda_density_fitting_uhf_device_scf(
                plan.get(), h, x, initial, initial_beta, {occupied},
                {static_cast<std::int32_t>(beta_occupied)}, {0}, 80, 1e-12, 1e-10, density, beta,
                records, detail)
          : run_cuda_density_fitting_rhf_device_scf(plan.get(), h, x, initial, {occupied}, {0}, 80,
                                                    1e-12, 1e-10, density, records, detail);
  require(status == VIBEQC_STATUS_SUCCESS && records[0].converged,
          "packed response solve: " + detail);
  CudaDfFinalStateToken token;
  require(
      cuda_density_fitting_final_state_token(plan.get(), 0, token, detail) == VIBEQC_STATUS_SUCCESS,
      detail);
  if (!uhf) {
    std::vector<double> j, k;
    bool used{};
    require(try_cuda_density_fitting_final_rhf_jk(plan.get(), token, density, j, k, used, detail) ==
                    VIBEQC_STATUS_SUCCESS &&
                used,
            "packed final K: " + detail);
    require(plan->final_projection_token.has_value() ==
                (capacity >= static_cast<std::size_t>(occupied) && plan->metric_full_rank[0]),
            "packed final U lease exceeded actual retained capacity/rank");
  }
  const auto force = [&](const CudaDfFinalStateToken* expected, bool expected_occupied) {
    auto total = density;
    if (uhf)
      for (std::size_t i = 0; i < total.size(); ++i) total[i] += beta[i];
    const std::vector<DensityFittingDensityResponse> terms =
        uhf ? std::vector<DensityFittingDensityResponse>{{total, 1, 0},
                                                         {density, 0, .5},
                                                         {beta, 0, .5}}
            : std::vector<DensityFittingDensityResponse>{{density, 1, .25}};
    const auto reference =
        uhf ? build_density_fitting_uhf_gradient(oracle, density, beta, cutoff).derivative
            : build_density_fitting_rhf_gradient(oracle, density, cutoff).derivative;
    std::vector<double> derivative;
    DfGradientResources resources;
    require(execute_cuda_density_fitting_generated_force_response(
                plan.get(), 0, orbital, auxiliary, {}, {}, terms, 0, 4U << 20, 1, derivative,
                detail, &resources, expected) == VIBEQC_STATUS_SUCCESS,
            detail);
    require(resources.occupied_response == expected_occupied,
            "packed response selected the wrong factor route");
    require(resources.borrowed_device_bytes ==
                (expected_occupied
                     ? (plan->projection_capacity + 2 * plan->panel_capacity) * sizeof(double)
                     : 0),
            "packed response misreported unequal borrowed capacities");
    require(!resources.recomputed_value_bytes && !resources.tensor_host_to_device_bytes,
            "packed response regenerated/uploaded immutable raw values");
    require(derivative.size() == reference.size(), "packed force shape");
    for (std::size_t i = 0; i < derivative.size(); ++i)
      require(std::abs(derivative[i] - reference[i]) < 8e-10,
              "packed force differs from raw integral oracle: spin=" + std::to_string(uhf) +
                  " rank=" + std::to_string(occupied) + " capacity=" + std::to_string(capacity) +
                  " cutoff=" + std::to_string(cutoff) + " coordinate=" + std::to_string(i) +
                  " actual=" + std::to_string(derivative[i]) +
                  " expected=" + std::to_string(reference[i]));
  };
  const auto rr = static_cast<std::size_t>(occupied) * occupied;
  const bool fits =
      rr * a <= plan->panel_capacity &&
      (rr + (uhf ? beta_occupied * beta_occupied : 0)) * a <= plan->projection_capacity;
  force(&token, fits);  // Optional final-U lease is consumed exactly once.
  require(!plan->final_projection_token, "packed response failed to revoke final U lease");
  setenv("VIBEQC_DF_WEIGHTED_EXECUTION", "shell", 1);
  setenv("VIBEQC_DF_DERIVATIVE_PAIRS", "packed", 1);
  setenv("VIBEQC_DF_SHELL_SCHEDULE", "compact", 1);
  force(&token, fits);  // Raw projection includes discarded metric directions.
  force(nullptr, false);
  auto stale = token;
  ++stale.identity.solve_epoch;
  force(&stale, false);
  density[0] += 1e-13;
  force(&token, false);
  if (!uhf && capacity == 1 && cutoff == 1e-12 && occupied == 1) {
    // Invalid view provenance and unequal capacities must fail before any
    // immutable raw allocation can become a mutable response destination.
    const CudaDfMetricView metric_view{plan->inverse_square_roots, plan->metric_eigenvectors,
                                       plan->metric_eigenvalues, cutoff};
    const CudaDfPackedRawTensorView packed_view{
        plan->packed_raw, n, a, plan->stored_pair_count, plan->factor_basis_identity, metric_view};
    CudaDfResponseBuffers valid;
    valid.staging_weights = plan->auxiliary_tile_values;
    valid.raw_auxiliary_major = plan->exchange_contributions;
    valid.exchange_response = plan->exchange_intermediate;
    valid.staging_elements = plan->projection_capacity;
    valid.raw_elements = valid.exchange_elements = plan->panel_capacity;
    valid.occupied_response = true;
    valid.resident_packed_raw = packed_view;
    const auto* state = static_cast<const cuda_df::PersistentScfState*>(plan->persistent_scf_state);
    valid.occupied_factors[0] = {state->d_alpha_factor, 1, 2};
    const std::vector<DensityFittingDensityResponse> terms{{density, 1, .25}};
    for (unsigned fault = 0; fault < 8; ++fault) {
      auto view = packed_view;
      auto buffers = valid;
      if (fault == 0) view.owner_identity = 0;
      if (fault == 1) ++view.pair_count;
      if (fault == 2) view.metric.eigenvalues = nullptr;
      if (fault == 3) buffers.staging_elements = n * n - 1;
      if (fault == 4) buffers.exchange_elements = a - 1;
      if (fault == 5) buffers.raw_auxiliary_major = plan->packed_raw;
      if (fault == 6) buffers.staging_weights = buffers.exchange_response;
      if (fault == 7) ++buffers.resident_packed_raw.owner_identity;
      std::vector<double> output{123};
      require(execute_cuda_df_hf_gradient(0, reinterpret_cast<void*>(plan->stream),
                                          plan->integral_source, 0, orbital, auxiliary, {}, {}, {},
                                          terms, cutoff, 0, 4U << 20, 1, output, detail, nullptr,
                                          &metric_view, reinterpret_cast<void*>(plan->blas),
                                          &buffers, &view) == VIBEQC_STATUS_INVALID_ARGUMENT &&
                  output == std::vector<double>{123},
              "invalid packed response borrow changed output");
    }
  }
}
}  // namespace
int main() {
  try {
    malformed_automatic_token();
    lifecycle(false);
    lifecycle(true);
    for (std::size_t capacity : {0U, 1U})
      for (double cutoff : {1e-12, .2}) {
        packed_response(false, 0, capacity, cutoff);
        packed_response(true, 0, capacity, cutoff);
        packed_response(true, 1, capacity, cutoff);
      }
    for (double cutoff : {1e-12, .2}) {
      packed_response(false, 0, 2, cutoff, 2, 3);
      packed_response(true, 1, 2, cutoff, 2, 3);
      packed_response(false, 0, 2, cutoff, 2, 1);  // Insufficient rank-squared panel.
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
