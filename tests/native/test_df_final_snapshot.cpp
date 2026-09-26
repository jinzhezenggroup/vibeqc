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
    solver::FinalStateLimits export_limits;
    export_limits.require_canonicality = true;
    const solver::PhysicalFockFrame physical{tokens[item].identity, true,
                                             std::vector<Matrix>(uhf ? 2 : 1, f)};
    require(
        solver::validate_final_state(tokens[item].identity, s, f, .3, snapshot.density, physical,
                                     snapshot.candidate, export_limits, diagnostic, detail),
        detail);
    // Exercise the normal no-C-download provider against the independent
    // CPU products, then compare the actual W supplied to force consumers.
    const auto operations = cuda_density_fitting_final_state_operations(plan.get());
    auto retained = snapshot.candidate;
    retained.spins.assign(uhf ? 2 : 1, {});
    solver::FinalStateDiagnostic device_diagnostic;
    require(solver::validate_final_state(tokens[item].identity, s, f, .3, snapshot.density,
                                         physical, retained, export_limits, device_diagnostic,
                                         detail, &operations),
            detail);
    for (auto field :
         {&solver::FinalStateDiagnostic::energy, &solver::FinalStateDiagnostic::maximum_commutator,
          &solver::FinalStateDiagnostic::density_rms,
          &solver::FinalStateDiagnostic::maximum_trace_error,
          &solver::FinalStateDiagnostic::maximum_idempotency_error,
          &solver::FinalStateDiagnostic::maximum_density_error,
          &solver::FinalStateDiagnostic::maximum_canonical_error})
      require(std::abs(device_diagnostic.*field - diagnostic.*field) < 1e-12,
              "device diagnostics disagree with independent CPU products");
    for (std::size_t spin = 0; spin < diagnostic.eigenframes.size(); ++spin)
      for (auto field : {&solver::EigenFrameDiagnostic::maximum_eigen_residual,
                         &solver::EigenFrameDiagnostic::scaled_eigen_residual,
                         &solver::EigenFrameDiagnostic::maximum_metric_error})
        require(std::abs(device_diagnostic.eigenframes[spin].*field -
                         diagnostic.eigenframes[spin].*field) < 1e-12,
                "device eigen diagnostics disagree with independent CPU products");
    const auto selected = solver::select_final_state(
        tokens[item].identity, s, f, x, .3, snapshot.density, &retained,
        [&](const auto&, const auto&) { return physical; },
        [&](const auto& matrix, const auto* overlap, const auto* orthogonalizer,
            auto n) -> reference::EigenResult {
          require(matrix == f && overlap && *overlap == s && orthogonalizer &&
                      *orthogonalizer == x && n == 2,
                  "fixed-point probe lost its physical F/S/X inputs");
          // Independent analytic projector for the rotated physical fixture.
          // Validation requires this solve without changing the retained D.
          return {{-1, 3}, c};
        },
        {}, true, false, &operations);
    require(selected.state.has_value() && selected.reused, selected.detail);
    require(selected.density_updates == 0 && selected.fock_evaluations == 1 &&
                selected.fixed_point_checks == 1 && selected.fixed_point_rejections == 0 &&
                selected.eigen_solves == (uhf ? 2U : 1U) &&
                selected.fixed_point_eigen_solves == selected.eigen_solves,
            "stationary retained state was corrected or its physical probe was not counted");
    for (std::size_t spin = 0; spin < snapshot.density.size(); ++spin) {
      const auto expected = reference::energy_weighted_density(
          c, {-1, 3}, 2, tokens[item].identity.occupied[spin], uhf ? 1 : 2);
      for (std::size_t k = 0; k < 4; ++k)
        require(std::abs(selected.state->weighted_density[spin][k] - expected[k]) < 1e-12,
                "device W differs from the force-state oracle");
    }
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
  const auto rejects_device_frame = [&](const CudaDfFinalStateToken& token) {
    const auto operations = cuda_density_fitting_final_state_operations(plan.get());
    solver::FinalFrameCandidate retained{token.identity,
                                         token.identity.factor.density_generation - 1, true,
                                         std::vector<reference::EigenResult>(uhf ? 2 : 1)};
    std::vector<Matrix> densities{Matrix(final.begin(), final.begin() + 4)};
    if (uhf) densities.emplace_back(beta.begin(), beta.begin() + 4);
    const auto selected = solver::select_final_state(
        token.identity, s, f, x, .3, densities, &retained,
        [&](const auto&, const auto&) {
          return solver::PhysicalFockFrame{token.identity, true,
                                           std::vector<Matrix>(uhf ? 2 : 1, f)};
        },
        [](const auto&, const auto*, const auto*, auto) -> reference::EigenResult {
          throw std::runtime_error("corrupt owner must not enter correction");
        },
        {}, true, false, &operations);
    return !selected.state && selected.status == solver::FinalStateStatus::ProviderFailure &&
           selected.eigen_solves == 0;
  };
  // Retained device generation/info corruption fails independently of the
  // host token. Every failure must leave the output empty.
  checked(cudaMemsetAsync(state->d_final_alpha_generation, 0, sizeof(std::uint64_t), plan->stream));
  require(rejects_device_frame(tokens[0]), "device validation accepted a corrupt generation");
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
  require(rejects_device_frame(recovered), "device validation accepted failed solver info");
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
/** A closed-form nonidentity-overlap determinant tests one-step admission
 * independently of a molecular oracle, including unpublished/stale/failed
 * states and exact input matching. There are no two-electron contributions. */
void warm_replay() {
  exchange_policy("occupied");
#ifdef _WIN32
  _putenv_s("VIBEQC_DF_FINAL_EXCHANGE", "occupied");
#else
  setenv("VIBEQC_DF_FINAL_EXCHANGE", "occupied", 1);
#endif
  CudaDensityFittingJkPlan* raw{};
  std::vector<CudaDensityFittingMetricDiagnostic> diagnostics;
  std::string detail;
  require(
      create_cuda_density_fitting_jk_plan_tiled(0, 1, 2, 1, Matrix{1}, Matrix(4, 0), 1e-10, 1, 4,
                                                &raw, diagnostics, detail) == VIBEQC_STATUS_SUCCESS,
      detail);
  Plan plan(raw, &destroy_cuda_density_fitting_jk_plan);
  const Matrix h{-2, 0, 0, 12}, s{2, 0, 0, 4}, x{1 / std::sqrt(2.), 0, 0, .5};
  // This unbudgeted analytic tensor fixture explicitly admits the two-slot
  // history normally reserved by the high-level molecular plan builder.
  plan->scf_diis_history = 2;
  Matrix input{1, 0, 0, 0}, final;
  std::vector<CudaDensityFittingDeviceScfItem> records;
  const auto matches = [&] {
    return cuda_density_fitting_rhf_warm_matches(plan.get(), input, h, s, x, 1, .3);
  };
  const auto run = [&](unsigned maximum) {
    return run_cuda_density_fitting_rhf_device_scf(plan.get(), h, x, input, {1}, {.3}, maximum,
                                                   1e-12, 1e-10, final, records, detail, s, 2);
  };
  const auto publish = [&] {
    CudaDfFinalStateToken token;
    require(cuda_density_fitting_final_state_token(plan.get(), 0, token, detail) ==
                VIBEQC_STATUS_SUCCESS,
            detail);
    const auto fock =
        evaluate_cuda_density_fitting_final_fock(plan.get(), token.identity, {final}, h);
    CudaDfFinalStateSnapshot snapshot;
    require(read_cuda_density_fitting_final_state(plan.get(), token, snapshot, detail) ==
                VIBEQC_STATUS_SUCCESS,
            detail);
    // CPU matrix products and the analytic -1.7 Eh energy are independent
    // checks before the internal caller authorizes retention.
    solver::FinalStateDiagnostic diagnostic;
    require(
        solver::validate_final_state(token.identity, s, h, .3, {final}, {token.identity, true, {h}},
                                     snapshot.candidate, {}, diagnostic, detail),
        detail);
    require(std::abs(diagnostic.energy + 1.7) < 1e-12, "wrong analytic warm energy");
    prepare_cuda_density_fitting_rhf_warm_state(plan.get(), token, final, h, s, x, 1, .3);
    auto stale = token;
    ++stale.identity.factor.density_generation;
    commit_cuda_density_fitting_rhf_warm_state(plan.get(), stale);
    require(!matches(), "unpublished or stale-token warm state escaped");
    commit_cuda_density_fitting_rhf_warm_state(plan.get(), token);
  };
  require(!matches(), "cold plan manufactured a warm baseline");
  require(run(8) == VIBEQC_STATUS_SUCCESS, detail);
  require(records[0].converged && records[0].iterations > 1,
          "cold determinant skipped its missing energy baseline");
  input = final;
  publish();
  require(matches(), "strict accepted determinant did not become eligible");
  auto changed = input;
  changed[0] += 1e-14;
  require(!cuda_density_fitting_rhf_warm_matches(plan.get(), changed, h, s, x, 1, .3),
          "nearby imported density passed exact warm admission");
  auto shifted_h = h;
  shifted_h[0] += 1e-14;
  require(!cuda_density_fitting_rhf_warm_matches(plan.get(), input, shifted_h, s, x, 1, .3) &&
              !cuda_density_fitting_rhf_warm_matches(plan.get(), input, h, x, x, 1, .3) &&
              !cuda_density_fitting_rhf_warm_matches(plan.get(), input, h, s, s, 1, .3) &&
              !cuda_density_fitting_rhf_warm_matches(plan.get(), input, h, s, x, 0, .3) &&
              !cuda_density_fitting_rhf_warm_matches(plan.get(), input, h, s, x, 1, .4),
          "changed Hamiltonian/metric/projector/occupation/nuclear energy reused a baseline");
  ++plan->factor_basis_identity;
  require(!matches(), "another immutable source reused the warm frame");
  --plan->factor_basis_identity;
  for (int repeat = 0; repeat < 3; ++repeat) {
    require(run(1) == VIBEQC_STATUS_SUCCESS && records[0].converged && records[0].iterations == 1,
            "qualified frozen warm replay did not finish in exactly one iteration");
    require(!matches(), "unfinalized solve published warm eligibility");
    publish();
    require(matches(), "frozen input was lost when the latest frame advanced");
  }
  require(run(0) == VIBEQC_STATUS_INVALID_ARGUMENT && !matches(),
          "failed solve preserved a prior warm entry");
  require(run(1) == VIBEQC_STATUS_SUCCESS && !records[0].converged && !matches(),
          "failed/nonconverged solve manufactured first-step convergence");
#ifdef _WIN32
  _putenv_s("VIBEQC_DF_FINAL_EXCHANGE", "");
#else
  unsetenv("VIBEQC_DF_FINAL_EXCHANGE");
#endif
}
}  // namespace

int main() {
  try {
    for (const bool uhf : {false, true})
      for (const std::size_t batch : {1U, 4U}) lifecycle(uhf, batch);
    warm_replay();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
