/** Exercise the production density factor and dense fallback against an
 * independent four-index RI contraction, including invalid and mixed seeds.
 * This executable requires a real device allocated by Slurm.
 */
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "scf/cuda/df_jk_internal.hpp"
#include "scf/cuda/df_scf_factor.hpp"
#include "scf/cuda_density_fitting_final_state.hpp"

namespace {
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void check(vibeqc_status status, const std::string& detail) {
  if (status != VIBEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
}
void cuda_check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}  // namespace

int main() {
  if (!std::getenv("SLURM_JOB_ID")) return 77;
  using namespace vibeqc::scf;
  using namespace vibeqc::scf::cuda_df;
  try {
    require(std::getenv("SLURM_JOB_ID"), "density seed test requires Slurm");
    (void)setenv("VIBEQC_DF_EXCHANGE", "occupied", 1);
    (void)setenv("VIBEQC_DF_SEED_EXCHANGE", "dense", 1);
    constexpr std::size_t n = 8, a = 5, capacity = 2;
    std::vector<double> metric(a * a), b(n * n * a), h(n * n), x(n * n), d(n * n);
    for (std::size_t q = 0; q < a; ++q) metric[q * a + q] = 1;
    for (std::size_t i = 0; i < n; ++i) {
      h[i * n + i] = -1.0 / (i + 1);
      x[i * n + i] = 1;
      for (std::size_t j = 0; j < n; ++j)
        for (std::size_t q = 0; q < a; ++q)
          b[(i * n + j) * a + q] = 0.03 * std::cos((i + j + 1) * (q + 1));
    }
    CudaDensityFittingJkPlan* raw = nullptr;
    std::vector<CudaDensityFittingMetricDiagnostic> diagnostics;
    std::string detail;
    check(create_cuda_density_fitting_jk_plan(0, 1, n, a, metric, b, 1e-10, a, &raw, diagnostics,
                                              detail),
          detail);
    std::unique_ptr<CudaDensityFittingJkPlan, decltype(&destroy_cuda_density_fitting_jk_plan)> plan(
        raw, destroy_cuda_density_fitting_jk_plan);
    std::vector<double> final;
    std::vector<CudaDensityFittingDeviceScfItem> records;
    check(run_cuda_density_fitting_rhf_device_scf(raw, h, x, d, {capacity}, {0}, 1, 1e-12, 1e-10,
                                                  final, records, detail),
          detail);
    auto& state = *static_cast<PersistentScfState*>(raw->persistent_scf_state);
    std::cout << std::setprecision(17);
    for (int fixture = 0; fixture < 8; ++fixture) {
      // Two orthonormal, dense columns with fractional/idempotent occupations.
      // Mixed PSD can fit capacity or require an extra independent direction.
      const double w0 = fixture == 1 ? 0.7 : 2.0, w1 = fixture == 1 ? 0.2 : 2.0;
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
          const double u = 1.0 / std::sqrt(static_cast<double>(n));
          const double vi = (i % 2 ? -1 : 1) * u, vj = (j % 2 ? -1 : 1) * u;
          d[i * n + j] = fixture == 2 ? 0 : w0 * u * u + w1 * vi * vj;
          const double zi = (i % 4 < 2 ? -1 : 1) * u;
          const double zj = (j % 4 < 2 ? -1 : 1) * u;
          if (fixture == 3) d[i * n + j] += 1e-5 * zi * zj;
          if (fixture == 4) d[i * n + j] -= 1e-5 * zi * zj;
          if (fixture == 5) d[i * n + j] += 2e-14 * zi * zj;
          if (fixture == 6 && i == j) d[i * n + j] += 0.01;
        }
      if (fixture == 7) d[1] += 1e-5;
      // The internal SCF adapter takes column-major D. Keep the asymmetric
      // rejection fixture honest instead of accidentally comparing K[D^T].
      std::vector<double> column_major(n * n);
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) column_major[i + j * n] = d[i * n + j];
      cuda_check(cudaMemcpyAsync(state.d_density, column_major.data(), d.size() * sizeof(double),
                                 cudaMemcpyHostToDevice, raw->stream));
      bool accepted = false;
      std::size_t rank = 0;
      check(factor_density_for_exchange(*raw, state, state.d_density, accepted, rank, detail),
            detail);
      const bool expected = fixture <= 2 || fixture == 5;
      require(accepted == expected, "density factor acceptance differs from scientific contract");
      if (accepted) require(rank == (fixture == 2 ? 0 : capacity), "unexpected retained rank");
      std::vector<double> reconstructed(n * n), factor(n * capacity);
      if (accepted) {
        cuda_check(cudaMemcpy(factor.data(), state.d_alpha_factor, factor.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < n; ++i)
          for (std::size_t j = 0; j < n; ++j)
            for (std::size_t k = 0; k < rank; ++k)
              reconstructed[i * n + j] += factor[i + k * n] * factor[j + k * n];
      }
      std::vector<double> j, dense;
      check(execute_cuda_density_fitting_rhf_jk(raw, d, j, dense, detail), detail);
      (void)setenv("VIBEQC_DF_SEED_EXCHANGE", "factor", 1);
      check(reset_scf_factors(*raw, state, detail), detail);
      check(build_scf_occupied_jk(*raw, state, state.d_density, nullptr, false, detail), detail);
      require(state.density_seed_used == expected, "seed fallback did not execute expected branch");
      std::vector<double> actual(n * n), actual_j(n * n), oracle(n * n);
      // The internal device adapter is asynchronous on a nonblocking stream.
      // Its readback must use that same stream, including the triangle mirror.
      cuda_check(cudaMemcpyAsync(actual.data(), raw->alpha_exchange, actual.size() * sizeof(double),
                                 cudaMemcpyDeviceToHost, raw->stream));
      cuda_check(cudaMemcpyAsync(actual_j.data(), raw->coulomb, actual_j.size() * sizeof(double),
                                 cudaMemcpyDeviceToHost, raw->stream));
      cuda_check(cudaStreamSynchronize(raw->stream));
      double maximum = 0, sum = 0, fock_max = 0, reconstruction_max = 0;
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t jcol = 0; jcol < n; ++jcol) {
          const auto ij = i * n + jcol;
          for (std::size_t k = 0; k < n; ++k)
            for (std::size_t l = 0; l < n; ++l)
              for (std::size_t q = 0; q < a; ++q)
                oracle[ij] += b[(i * n + k) * a + q] * b[(jcol * n + l) * a + q] * d[k * n + l];
          if (std::abs(actual[ij] - oracle[ij]) >= 1e-12) {
            std::cerr << std::setprecision(17) << "fixture " << fixture << " entry " << ij
                      << " factor=" << actual[ij] << " dense=" << dense[ij]
                      << " oracle=" << oracle[ij] << '\n';
            throw std::runtime_error("seed K differs from independent RI oracle");
          }
          const double error = actual[ij] - dense[ij];
          maximum = std::max(maximum, std::abs(error));
          sum += error * error;
          fock_max = std::max(fock_max, std::abs(actual_j[ij] - j[ij] - 0.5 * error));
          reconstruction_max = std::max(reconstruction_max, std::abs(reconstructed[ij] - d[ij]));
        }
      require(maximum < 1e-12 && fock_max < 1e-12, "seed K/Fock gate failed");
      if (accepted) require(reconstruction_max <= 1e-12, "density reconstruction gate failed");
      std::cout << "{\"fixture\":" << fixture << ",\"accepted\":" << accepted
                << ",\"rank\":" << rank << ",\"k_max\":" << maximum
                << ",\"k_rms\":" << std::sqrt(sum / (n * n)) << ",\"fock_max\":" << fock_max
                << ",\"reconstruction_max\":";
      if (accepted)
        std::cout << reconstruction_max;
      else
        std::cout << "null";
      std::cout << "}\n";
      // The one-step solver must overwrite algebraic factor provenance and
      // respect max_iterations=1 even when its seed is accepted.
      check(run_cuda_density_fitting_rhf_device_scf(raw, h, x, d, {capacity}, {0}, 1, 1e-12, 1e-10,
                                                    final, records, detail),
            detail);
      require(records[0].iterations == 1 && !records[0].converged, "seed exceeded iteration limit");
    }
    (void)setenv("VIBEQC_DF_FINAL_EXCHANGE", "occupied", 1);
    check(run_cuda_density_fitting_rhf_device_scf(raw, h, x, final, {capacity}, {0}, 100, 1e-12,
                                                  1e-10, d, records, detail),
          detail);
    require(records[0].converged, "final-state fixture did not converge");
    CudaDfFinalStateToken token;
    check(cuda_density_fitting_final_state_token(raw, 0, token, detail), detail);
    CudaDfFinalStateSnapshot snapshot;
    check(read_cuda_density_fitting_final_state(raw, token, snapshot, detail), detail);
    std::vector<double> expected_j, expected_k, actual_j, actual_k;
    check(execute_cuda_density_fitting_rhf_jk(raw, d, expected_j, expected_k, detail), detail);
    bool used = false;
    check(try_cuda_density_fitting_final_rhf_jk(raw, token, d, actual_j, actual_k, used, detail),
          detail);
    require(used, "matching retained final factor was rejected");
    for (std::size_t k = 0; k < n * n; ++k) {
      require(std::abs(actual_k[k] - expected_k[k]) < 1e-12, "final retained K differs from dense");
      require(std::abs(actual_j[k] - expected_j[k]) < 1e-12, "final retained J differs from dense");
    }
    (void)setenv("VIBEQC_DF_FINAL_EXCHANGE", "auto", 1);
    check(try_cuda_density_fitting_final_rhf_jk(raw, token, d, actual_j, actual_k, used, detail),
          detail);
    require(!used, "automatic final K escaped its qualified workload domain");
    (void)setenv("VIBEQC_DF_FINAL_EXCHANGE", "occupied", 1);
    auto changed = d;
    changed[0] += 1e-14;
    check(try_cuda_density_fitting_final_rhf_jk(raw, token, changed, actual_j, actual_k, used,
                                                detail),
          detail);
    require(!used, "changed final density reused old factor");
    auto stale = token;
    ++stale.identity.solve_epoch;
    check(try_cuda_density_fitting_final_rhf_jk(raw, stale, d, actual_j, actual_k, used, detail),
          detail);
    require(!used, "stale final token reused old factor");
    const int failed_solver = 1, successful_solver = 0;
    cuda_check(cudaMemcpy(state.d_final_alpha_info, &failed_solver, sizeof(failed_solver),
                          cudaMemcpyHostToDevice));
    check(try_cuda_density_fitting_final_rhf_jk(raw, token, d, actual_j, actual_k, used, detail),
          detail);
    require(!used, "failed final eigensolver reused its factor");
    cuda_check(cudaMemcpy(state.d_final_alpha_info, &successful_solver, sizeof(successful_solver),
                          cudaMemcpyHostToDevice));
    const std::uint64_t bad_generation = 0;
    cuda_check(cudaMemcpy(state.d_final_alpha_generation, &bad_generation, sizeof(bad_generation),
                          cudaMemcpyHostToDevice));
    check(try_cuda_density_fitting_final_rhf_jk(raw, token, d, actual_j, actual_k, used, detail),
          detail);
    require(!used, "stale device generation reused old factor");
    std::cout << "{\"final_retained_identity_gates\":\"passed\"}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
