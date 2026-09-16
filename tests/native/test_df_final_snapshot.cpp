#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_scf_final_state.hpp"
#include "scf/cuda_density_fitting_final_state.hpp"
#include "scf/reference/mean_field.hpp"

namespace {
using namespace vibeqc::scf;
using namespace vibeqc::scf::cuda_df;
using reference::Matrix;
void require(bool value, const std::string& detail) {
  if (!value) throw std::runtime_error(detail);
}
void checked(cudaError_t error) { require(error == cudaSuccess, cudaGetErrorString(error)); }
void exchange_policy(const char* value) {
#ifdef _WIN32
  require(_putenv_s("VIBEQC_DF_EXCHANGE", value ? value : "") == 0, "cannot set exchange policy");
#else
  require((value ? setenv("VIBEQC_DF_EXCHANGE", value, 1) : unsetenv("VIBEQC_DF_EXCHANGE")) == 0,
          "cannot set exchange policy");
#endif
}
using Plan =
    std::unique_ptr<CudaDensityFittingJkPlan, decltype(&destroy_cuda_density_fitting_jk_plan)>;

void lifecycle(bool uhf, std::size_t batch) {
  CudaDensityFittingJkPlan* raw{};
  std::vector<CudaDensityFittingMetricDiagnostic> diagnostics;
  std::string detail;
  require(create_cuda_density_fitting_jk_plan_tiled(0, batch, 2, 1, Matrix(batch, 1),
                                                    Matrix(batch * 4, 0), 1e-10, 1, 4, &raw,
                                                    diagnostics, detail) == VIBEQC_STATUS_SUCCESS,
          detail);
  Plan plan(raw, &destroy_cuda_density_fitting_jk_plan);
  const Matrix s{2, 0, 0, 4}, x{1 / std::sqrt(2.0), 0, 0, .5};
  const double cosine = std::cos(.31), sine = std::sin(.31);
  // A rotated nonidentity-metric frame detects a column/row layout mistake.
  const Matrix c{x[0] * cosine, -x[0] * sine, .5 * sine, .5 * cosine};
  const Matrix f{2 * (-cosine * cosine + 3 * sine * sine), -8 * std::sqrt(2.0) * cosine * sine,
                 -8 * std::sqrt(2.0) * cosine * sine, 4 * (-sine * sine + 3 * cosine * cosine)};
  const auto one = reference::density_from_orbitals(c, 2, 1, uhf ? 1 : 2);
  Matrix h, xs, initial;
  for (std::size_t item = 0; item < batch; ++item) {
    h.insert(h.end(), f.begin(), f.end());
    xs.insert(xs.end(), x.begin(), x.end());
    const auto d = item == 1 ? Matrix(4, 0) : one;
    initial.insert(initial.end(), d.begin(), d.end());
  }
  Matrix final, beta;
  std::vector<CudaDensityFittingDeviceScfItem> records;
  const auto run = [&](unsigned maximum) {
    if (uhf)
      return run_cuda_density_fitting_uhf_device_scf(
          plan.get(), h, xs, initial, Matrix(batch * 4, 0), std::vector<std::int32_t>(batch, 1),
          std::vector<std::int32_t>(batch, 0), Matrix(batch, .3), maximum, 1e-10, 1e-10, final,
          beta, records, detail);
    return run_cuda_density_fitting_rhf_device_scf(
        plan.get(), h, xs, initial, std::vector<std::int32_t>(batch, 1), Matrix(batch, .3), maximum,
        1e-10, 1e-10, final, records, detail);
  };
  require(run(8) == VIBEQC_STATUS_SUCCESS, detail);
  require(records.size() == batch && records[0].converged, "analytic SCF failed");
  if (batch > 1)
    require(records[0].iterations < records[1].iterations,
            "fixture did not exercise an early inactive neighbor");
  std::vector<CudaDfFinalStateToken> tokens(batch);
  for (std::size_t item = 0; item < batch; ++item) {
    require(cuda_density_fitting_final_state_token(plan.get(), item, tokens[item], detail) ==
                VIBEQC_STATUS_SUCCESS,
            detail);
    CudaDfFinalStateSnapshot snapshot;
    require(read_cuda_density_fitting_final_state(plan.get(), tokens[item], snapshot, detail) ==
                VIBEQC_STATUS_SUCCESS,
            detail);
    require(
        snapshot.candidate.identity.factor.density_generation == records[item].iterations + 1ULL &&
            snapshot.candidate.fock_density_generation == records[item].iterations,
        "snapshot confused input/output density generations");
    solver::FinalStateDiagnostic diagnostic;
    const solver::PhysicalFockFrame physical{tokens[item].identity, true,
                                             std::vector<Matrix>(uhf ? 2 : 1, f)};
    require(solver::validate_final_state(tokens[item].identity, s, f, .3, snapshot.density,
                                         physical, snapshot.candidate, {}, diagnostic, detail),
            detail);
    for (std::size_t k = 0; k < 4; ++k)
      require(std::abs(snapshot.density[0][k] - one[k]) < 1e-12 &&
                  snapshot.density[0][k] == final[item * 4 + k],
              "snapshot lost the actual returned density");
    if (uhf) require(snapshot.density[1] == Matrix(4, 0), "empty beta snapshot became nonzero");
  }
  auto* state = static_cast<PersistentScfState*>(plan->persistent_scf_state);
  // Simulate later inactive GEMMs/eigensolves overwriting all scratch with
  // NaNs. Launching the actual masked stores must preserve the saved frames.
  checked(cudaMemsetAsync(state->d_temporary, 0xff, batch * 4 * sizeof(double), plan->stream));
  checked(cudaMemsetAsync(uhf ? state->d_alpha_eigenvalues : state->d_eigenvalues, 0xff,
                          batch * 2 * sizeof(double), plan->stream));
  store_scf_final_frame(*plan, *state, state->d_temporary, false);
  if (uhf) {
    checked(
        cudaMemsetAsync(state->d_beta_eigenvalues, 0xff, batch * 2 * sizeof(double), plan->stream));
    store_scf_final_frame(*plan, *state, state->d_temporary, true);
  }
  checked(cudaStreamSynchronize(plan->stream));
  CudaDfFinalStateSnapshot snapshot;
  require(read_cuda_density_fitting_final_state(plan.get(), tokens[0], snapshot, detail) ==
              VIBEQC_STATUS_SUCCESS,
          "inactive launches overwrote retained C/epsilon");
  const char* original_policy = std::getenv("VIBEQC_DF_EXCHANGE");
  const std::string saved_policy = original_policy ? original_policy : "";
  for (const char* changed : {state->occupied_exchange ? "dense" : "occupied", "invalid"}) {
    exchange_policy(changed);
    const auto status =
        read_cuda_density_fitting_final_state(plan.get(), tokens[0], snapshot, detail);
    exchange_policy(original_policy ? saved_policy.c_str() : nullptr);
    require(status == VIBEQC_STATUS_INVALID_ARGUMENT && snapshot.density.empty(),
            "changed/invalid exchange policy reused the prior successful solve");
  }

  const std::vector<std::function<void(CudaDfFinalStateToken&)>> faults{
      [](auto& t) { ++t.version; },
      [](auto& t) { ++t.identity.factor.basis; },
      [](auto& t) { t.identity.factor.reference = 0; },
      [](auto& t) { ++t.identity.solve_epoch; },
      [](auto& t) { ++t.identity.factor.density_generation; },
      [](auto& t) { ++t.identity.factor.orbital_generation; },
      [](auto& t) { t.identity.occupied[0] = 0; },
      [](auto& t) { t.identity.model.metric_relative_threshold *= 2; },
      [](auto& t) { t.identity.model.screening_tolerance *= 2; }};
  for (const auto& fault : faults) {
    auto stale = tokens[0];
    fault(stale);
    require(read_cuda_density_fitting_final_state(plan.get(), stale, snapshot, detail) ==
                    VIBEQC_STATUS_INVALID_ARGUMENT &&
                snapshot.density.empty() && snapshot.candidate.spins.empty(),
            "stale token published a snapshot");
  }
  // A rejected/unsupported final-K attempt must consume an old scratch lease
  // even when no CUDA work is submitted. Otherwise a later force could see
  // an intermediate from a preceding attempt under an apparently valid token.
  plan->final_projection_token = tokens[0];
  Matrix rejected_j, rejected_k;
  bool reused = true;
  require(try_cuda_density_fitting_final_rhf_jk(plan.get(), tokens[0], {}, rejected_j, rejected_k,
                                                reused, detail) == VIBEQC_STATUS_SUCCESS &&
              !reused && !plan->final_projection_token,
          "unsupported final K preserved the old projection lease");
  // Retained device generation/info corruption fails independently of the
  // host token. Every failure must leave the output empty.
  checked(cudaMemsetAsync(state->d_final_alpha_generation, 0, sizeof(std::uint64_t), plan->stream));
  require(read_cuda_density_fitting_final_state(plan.get(), tokens[0], snapshot, detail) ==
                  VIBEQC_STATUS_NUMERICAL_FAILURE &&
              snapshot.density.empty(),
          "corrupt device generation passed");
  plan->final_projection_token = tokens[0];
  require(run(8) == VIBEQC_STATUS_SUCCESS, detail);
  require(!plan->final_projection_token, "new solve preserved the old projection lease");
  require(read_cuda_density_fitting_final_state(plan.get(), tokens[0], snapshot, detail) ==
              VIBEQC_STATUS_INVALID_ARGUMENT,
          "warm replay accepted the preceding epoch");
  CudaDfFinalStateToken recovered;
  require(cuda_density_fitting_final_state_token(plan.get(), 0, recovered, detail) ==
              VIBEQC_STATUS_SUCCESS,
          detail);
  state = static_cast<PersistentScfState*>(plan->persistent_scf_state);
  checked(cudaMemsetAsync(state->d_final_alpha_info, 1, sizeof(int), plan->stream));
  require(read_cuda_density_fitting_final_state(plan.get(), recovered, snapshot, detail) ==
                  VIBEQC_STATUS_NUMERICAL_FAILURE &&
              snapshot.density.empty(),
          "failed active solver info passed");
  plan->final_projection_token = recovered;
  require(run(0) == VIBEQC_STATUS_INVALID_ARGUMENT, "invalid solve request was accepted");
  require(!plan->final_projection_token, "invalid solve preserved the old projection lease");
  require(read_cuda_density_fitting_final_state(plan.get(), recovered, snapshot, detail) ==
              VIBEQC_STATUS_INVALID_ARGUMENT,
          "invalid replay preserved previous eligibility");
  require(run(1) == VIBEQC_STATUS_SUCCESS, detail);
  require(cuda_density_fitting_final_state_token(plan.get(), 0, recovered, detail) ==
              VIBEQC_STATUS_INVALID_ARGUMENT,
          "nonconverged item exported a frame");
  require(run(8) == VIBEQC_STATUS_SUCCESS, detail);
  require(cuda_density_fitting_final_state_token(plan.get(), 0, recovered, detail) ==
              VIBEQC_STATUS_SUCCESS,
          "independent replay did not recover");
  if (uhf) {
    // Give alpha and beta deliberately different scratch contents. Saving
    // alpha after beta, or aliasing either retained allocation, loses these
    // exact witnesses even when a physical fixture has equal spin Focks.
    state = static_cast<PersistentScfState*>(plan->persistent_scf_state);
    checked(cudaMemsetAsync(state->d_active, 1, batch, plan->stream));
    const Matrix alpha_c(batch * 4, 3), beta_c(batch * 4, 7);
    const Matrix alpha_e(batch * 2, 11), beta_e(batch * 2, 13);
    const auto upload = [&](double* destination, const Matrix& source) {
      checked(cudaMemcpyAsync(destination, source.data(), source.size() * sizeof(double),
                              cudaMemcpyHostToDevice, plan->stream));
    };
    upload(state->d_temporary, alpha_c);
    upload(state->d_alpha_eigenvalues, alpha_e);
    store_scf_final_frame(*plan, *state, state->d_temporary, false);
    upload(state->d_temporary, beta_c);
    upload(state->d_beta_eigenvalues, beta_e);
    store_scf_final_frame(*plan, *state, state->d_temporary, true);
    Matrix saved_alpha(batch * 4), saved_beta(batch * 4), saved_alpha_e(batch * 2),
        saved_beta_e(batch * 2);
    const auto download = [&](Matrix& destination, const double* source) {
      checked(cudaMemcpyAsync(destination.data(), source, destination.size() * sizeof(double),
                              cudaMemcpyDeviceToHost, plan->stream));
    };
    download(saved_alpha, state->d_final_alpha_coefficients);
    download(saved_beta, state->d_final_beta_coefficients);
    download(saved_alpha_e, state->d_final_alpha_values);
    download(saved_beta_e, state->d_final_beta_values);
    checked(cudaStreamSynchronize(plan->stream));
    require(saved_alpha == alpha_c && saved_beta == beta_c && saved_alpha_e == alpha_e &&
                saved_beta_e == beta_e,
            "UHF alpha/beta final frames alias shared scratch");
  }
  plan->final_state_solve_epoch = std::numeric_limits<std::uint64_t>::max();
  require(cuda_density_fitting_solve_epoch(plan.get()) == 0,
          "saturated epoch authorized a host-recovery final state");
  require(run(8) == VIBEQC_STATUS_NUMERICAL_FAILURE, "solve epoch wrapped around");
  require(read_cuda_density_fitting_final_state(plan.get(), recovered, snapshot, detail) ==
              VIBEQC_STATUS_INVALID_ARGUMENT,
          "epoch exhaustion preserved eligibility");
}
}  // namespace

int main() {
  try {
    for (const bool uhf : {false, true})
      for (const std::size_t batch : {1U, 4U}) lifecycle(uhf, batch);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
