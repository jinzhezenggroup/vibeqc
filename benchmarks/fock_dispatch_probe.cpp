/** Same-source before/after dispatch probe for #202.
 * Compile against each revision's headers/library with -O3 -DNDEBUG.
 * Synthetic symmetric positive tensors isolate contraction dispatch; complete
 * molecular energy/force endpoints are measured by benchmark_fock_strategies.py.
 * GPU calls return host matrices and synchronize before the clock is stopped.
 */
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "scf/cuda_density_fitting.hpp"
#include "scf/density_fitting.hpp"
#include "scf/fock_build.hpp"
#if __has_include("scf/fock_provider.hpp")
#include "scf/cuda_fock_provider.hpp"
#define VIBEQC_PROBE_HAS_PROVIDER 1
#else
#define VIBEQC_PROBE_HAS_PROVIDER 0
#endif

using namespace vibeqc::scf;
using Clock = std::chrono::steady_clock;

namespace {
void checked(vibeqc_status status, const std::string& detail) {
  if (status != VIBEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
}
void array(const std::vector<double>& values) {
  std::cout << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << values[i];
  }
  std::cout << ']';
}

void probe(std::size_t n, bool fitted, bool cuda) {
  const auto matrix = n * n;
  const auto auxiliary = n + 2;
  DensityFittingScfData data;
  data.raw.nbf = data.one_electron.nbf = n;
  data.raw.naux = auxiliary;
  data.raw.metric.resize(auxiliary * auxiliary);
  data.raw.three_center.resize(matrix * auxiliary);
  data.metric_relative_threshold = 1e-10;
  std::vector<double> density(matrix);
  for (std::size_t p = 0; p < auxiliary; ++p) data.raw.metric[p * auxiliary + p] = 1;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      density[i * n + j] = (i == j ? 1.0 : 0.1) / n;
      for (std::size_t p = 0; p < auxiliary; ++p)
        data.raw.three_center[(i * n + j) * auxiliary + p] = 1.0 / (1 + i + j + p) / auxiliary;
    }
  data.three_center = orthonormalize_density_fitting_three_center(
      data.raw.three_center, n, factor_density_fitting_metric(data.raw.metric, auxiliary));
  if (!fitted) {
    data.one_electron.eri.resize(matrix * matrix);
    for (std::size_t a = 0; a < matrix; ++a)
      for (std::size_t b = 0; b < matrix; ++b)
        for (std::size_t p = 0; p < auxiliary; ++p)
          data.one_electron.eri[a * matrix + b] +=
              data.raw.three_center[a * auxiliary + p] * data.raw.three_center[b * auxiliary + p];
  }
  auto spec = make_hf_fock_spec(
      FockSpin::Restricted, fitted ? FockApproximation::DensityFitted : FockApproximation::Exact);
  spec.derivative_order = 0;
  const auto strategy = resolve_fock_build(spec, FockBackend::Cpu);
  CudaDensityFittingJkPlan* cuda_raw = nullptr;
  std::vector<CudaDensityFittingMetricDiagnostic> diagnostics;
  std::string detail;
  if (cuda)
    checked(create_cuda_density_fitting_jk_plan(0, 1, n, auxiliary, data.raw.metric,
                                                data.raw.three_center, 1e-10, 0, &cuda_raw,
                                                diagnostics, detail),
            detail);
  std::unique_ptr<CudaDensityFittingJkPlan, decltype(&destroy_cuda_density_fitting_jk_plan)> owner(
      cuda_raw, destroy_cuda_density_fitting_jk_plan);
  std::function<DirectJkMatrices()> build;
#if VIBEQC_PROBE_HAS_PROVIDER
  std::unique_ptr<CpuFockPlanView> cpu_plan;
  std::unique_ptr<CudaFockPlanView> cuda_plan;
  if (cuda) {
    const CudaFockProviderView view(cuda_raw, data);
    cuda_plan = std::make_unique<CudaFockPlanView>(resolve_fock_build(spec, FockBackend::Cuda), n,
                                                   0, view, view);
    build = [&] { return cuda_plan->build(density); };
  } else {
    const auto view = fitted ? CpuFockProviderView(data) : CpuFockProviderView(data.one_electron);
    cpu_plan = std::make_unique<CpuFockPlanView>(strategy, n, 0, view, view);
    build = [&] { return cpu_plan->build(density); };
  }
#else
  if (cuda) {
    build = [&] {
      DirectJkMatrices out;
      out.nbf = n;
      checked(execute_cuda_density_fitting_rhf_jk(cuda_raw, density, out.coulomb,
                                                  out.exchange_alpha, detail),
              detail);
      return out;
    };
  } else if (fitted) {
    build = [&] {
      auto out = build_density_fitting_rhf_jk(data.three_center, density);
      return DirectJkMatrices{n, std::move(out.coulomb), std::move(out.exchange), {}};
    };
  } else {
    build = [&] { return build_exact_direct_jk(strategy, n, data.one_electron.eri, density); };
  }
#endif
  DirectJkMatrices out;
  for (int i = 0; i < 5; ++i) out = build();
  const int repeats = cuda ? 100 : 500;
  std::vector<double> samples;
  for (int sample = 0; sample < 9; ++sample) {
    const auto start = Clock::now();
    for (int i = 0; i < repeats; ++i) out = build();
    samples.push_back(std::chrono::duration<double>(Clock::now() - start).count() / repeats);
  }
  std::cout << "{\"nbf\":" << n << ",\"naux\":" << auxiliary << ",\"approximation\":\""
            << (fitted ? "density_fitted" : "exact") << "\",\"repeats_per_sample\":" << repeats
            << ",\"seconds\":";
  array(samples);
  std::cout << ",\"coulomb\":";
  array(out.coulomb);
  std::cout << ",\"exchange\":";
  array(out.exchange_alpha);
  std::cout << '}';
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const bool cuda = argc == 2 && std::string(argv[1]) == "cuda";
    if (cuda && !std::getenv("SLURM_JOB_ID"))
      throw std::runtime_error("CUDA probe must run inside a Slurm GPU allocation");
    std::cout << std::setprecision(17) << "{\"provider_boundary\":" << VIBEQC_PROBE_HAS_PROVIDER
              << ",\"rows\":[";
    bool first = true;
    for (std::size_t n : {2, 16})
      for (bool fitted : {false, true}) {
        if (cuda && !fitted) continue;  // No independent direct CUDA API at baseline.
        if (!first) std::cout << ',';
        first = false;
        probe(n, fitted, cuda);
      }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
