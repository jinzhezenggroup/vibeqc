// Validation adapter: stage downloads transactionally and exercise retained-frame
// lifetime, provenance refusal and callbacks throwing after queued transfers.
#include <algorithm>
#include <array>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>

#include "cc/df_source_response.hpp"
#include "cc/lambda_response.hpp"
#include "posthf/capacity.hpp"
#include "posthf/raw_source.hpp"
#include "runtime/cuda_resources.cuh"

namespace {
// Compose the production native source -> CCSD -> Lambda -> factor pullback.
// The supplied Fock is held fixed, isolating correlation-source derivatives
// from the separate conventional-reference stationarity/Pulay chain.
generativeqc::cc::DFFactorResponseResult solve_factors(generativeqc::cc::DFSourceResult& fitted,
                                                       const double* fock) {
  using namespace generativeqc;
  const auto o = fitted.nocc, v = fitted.nvir, n = o + v;
  cc::Problem p;
  p.nocc = o;
  p.nvir = v;
  p.naux = fitted.naux;
  p.df_source_identity = fitted.source_identity;
  p.reference_retained_bytes = fitted.retained_source_bytes;
  p.df_boo = std::move(fitted.boo);
  p.df_bov = std::move(fitted.bov);
  p.df_bvv = std::move(fitted.bvv);
  p.ovov = std::move(fitted.ovov);
  p.ovvo = std::move(fitted.ovvo);
  p.oovv = std::move(fitted.oovv);
  p.ovoo = std::move(fitted.ovoo);
  p.oooo = std::move(fitted.oooo);
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j) p.foo.push_back(fock[i * n + j]);
  for (std::size_t a = 0; a < v; ++a)
    for (std::size_t b = 0; b < v; ++b) p.fvv.push_back(fock[(o + a) * n + o + b]);
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t a = 0; a < v; ++a) {
      p.fov.push_back(fock[i * n + o + a]);
      p.d1.push_back(fock[i * n + i] - fock[(o + a) * n + o + a]);
      p.initial_t1.push_back(p.fov.back() / p.d1.back());
    }
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b) {
          p.d2.push_back(p.d1[i * v + a] + p.d1[j * v + b]);
          p.initial_t2.push_back(p.ovov[((i * v + a) * o + j) * v + b] / p.d2.back());
        }
  cc::SolverOptions options;
  options.max_bytes = 1ULL << 30;
  options.energy_tolerance = 1e-13;
  options.residual_tolerance = 1e-11;
  const auto solved = cc::solve_cuda(p, options, 0);
  if (!solved.converged()) throw std::runtime_error(solved.reason);
  cc::LambdaOptions response;
  response.max_bytes = 1ULL << 30;
  response.gmres.absolute_tolerance = 1e-12;
  const auto r = cc::solve_lambda_parameter_response_cuda(p, solved, 0, response);
  return cc::pullback_df_factors_cuda(o, v, p.naux,
                                      {p.df_boo, p.df_bov, p.df_bvv, r.ovov, r.ovvo, r.oovv, r.ovoo,
                                       r.oooo, r.df_bov, r.df_bvv, p.df_source_identity},
                                      1ULL << 30, 0,
                                      fitted.retained_source_bytes + cc::problem_host_bytes(p));
}
}  // namespace

extern "C" int df_source_metric_response_probe(void* opaque, const double* coefficients,
                                               std::size_t occupied, double threshold,
                                               const double* const* input, double* const* output,
                                               std::size_t budget, std::size_t caller_bytes,
                                               int failure, std::size_t* counts, char* error,
                                               std::size_t error_size) noexcept {
  using namespace generativeqc;
  try {
    const auto& raw = *static_cast<posthf::RawSource*>(opaque);
    const auto n = raw.nbf(), q = raw.naux(), v = n - occupied;
    const auto nn = n * n, qq = q * q, full = nn * q;
    hf::PhysicalReference ref;
    ref.nbf = n;
    ref.nocc = occupied;
    ref.coefficients.assign(coefficients, coefficients + nn);
    auto fitted = cc::build_df_source_cuda(raw.orbital(), raw.auxiliary(), ref, 1ULL << 30,
                                           threshold, 0, 0, failure != 1);
    const auto forward_peak = fitted.numeric_capacity_bytes;
    const auto retained = fitted.retained_source_bytes;
    const auto rank = fitted.metric_rank;
    auto state = fitted.response_state;
    cc::DFFactorResponseResult seeds;
    seeds.source_identity = fitted.source_identity;
    const std::array<std::size_t, 3> sizes{q * occupied * occupied, q * occupied * v, q * v * v};
    const std::array<std::vector<double>*, 3> sectors{&seeds.boo, &seeds.bov, &seeds.bvv};
    if (failure == 16)
      seeds = solve_factors(fitted, input[0]);
    else
      for (std::size_t i = 0; i < 3; ++i) sectors[i]->assign(input[i], input[i] + sizes[i]);
    if (failure == 2) {
      // Identical extents and geometry, but a different frame and owner token.
      for (auto& x : ref.coefficients) x *= -1;
      auto wrong =
          cc::build_df_source_cuda(raw.orbital(), raw.auxiliary(), ref, 1ULL << 30, threshold, 0);
      seeds.source_identity = wrong.source_identity;
    }
    if (failure == 3) seeds.source_identity = 0;
    if (failure == 6) seeds.bov.pop_back();
    // Neither a forward result nor the caller's C storage is needed afterwards.
    // Poison then free host C, so an accidental borrowed host frame is exposed.
    std::fill(ref.coefficients.begin(), ref.coefficients.end(), 0.0);
    std::vector<double>().swap(ref.coefficients);
    fitted = {};
    std::vector<double> bar_raw(full), bar_c(nn), bar_metric(qq);
    const auto external =
        caller_bytes +
        sizeof(double) * (2 * (full + nn + qq) + nn + sizes[0] + sizes[1] + sizes[2]) +
        posthf::source_capacity(raw.orbital()) + posthf::source_capacity(raw.auxiliary());
    std::size_t rows = 0;
    auto run = [&](int injected) {
      rows = 0;
      return cc::pullback_df_source_cuda(
          state, seeds,
          [&](std::size_t mu, const double* row, cudaStream_t stream) {
            if (mu != rows++) throw std::runtime_error("source response row order mismatch");
            runtime::cuda_resource_check(cudaMemcpyAsync(bar_raw.data() + mu * n * q, row,
                                                         n * q * sizeof(double),
                                                         cudaMemcpyDeviceToHost, stream));
            if (injected == 4) throw std::runtime_error("injected raw consume failure");
          },
          [&](const double* dc, const double* dm, cudaStream_t stream) {
            runtime::cuda_resource_check(cudaMemcpyAsync(bar_c.data(), dc, nn * sizeof(double),
                                                         cudaMemcpyDeviceToHost, stream));
            runtime::cuda_resource_check(cudaMemcpyAsync(bar_metric.data(), dm, qq * sizeof(double),
                                                         cudaMemcpyDeviceToHost, stream));
            if (injected == 5) throw std::runtime_error("injected metric finish failure");
          },
          budget, external);
    };
    if (failure == 8 || failure == 9) {
      bool caught = false;
      try {
        (void)run(failure == 8 ? 4 : 5);
      } catch (const std::runtime_error&) {
        caught = true;
      }
      if (!caught) throw std::runtime_error("expected injected callback failure");
    }
    const auto result = run(failure);
    // A successful repeat certifies that source rows and the retained frame
    // survived the first reverse traversal without consuming or mutating them.
    const auto first_raw = bar_raw, first_c = bar_c, first_metric = bar_metric;
    (void)run(0);
    if (first_raw != bar_raw || first_c != bar_c || first_metric != bar_metric)
      throw std::runtime_error("retained source response changed on replay");
    const std::array<const std::vector<double>*, 3> values{&bar_raw, &bar_c, &bar_metric};
    for (std::size_t i = 0; i < 3; ++i) std::copy(values[i]->begin(), values[i]->end(), output[i]);
    const std::size_t work[]{result.numeric_capacity_bytes,
                             result.retained_source_bytes,
                             result.h2d_bytes,
                             result.embedding_arena_bytes,
                             result.metric_workspace_bytes,
                             result.transform.source_rows,
                             result.transform.source_values,
                             result.transform.output_rows,
                             result.transform.output_values,
                             result.transform.gemms,
                             result.transform.contraction_summands,
                             forward_peak,
                             retained,
                             rank};
    std::copy(std::begin(work), std::end(work), counts);
    return 0;
  } catch (const std::exception& e) {
    if (error && error_size) std::snprintf(error, error_size, "%s", e.what());
    return 1;
  }
}
