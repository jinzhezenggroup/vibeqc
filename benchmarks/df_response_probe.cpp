/** Synchronous DF response component benchmark. Execute only inside Slurm.
 * Arguments: case mode budget auxiliary-cap spin repeats. The CPU raw oracle
 * remains allocated for independent error checks; process high-water therefore
 * includes that oracle and is not an estimate of a tensor-free SCF endpoint.
 */
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "scf/cuda_density_fitting.hpp"
#include "scf/cuda_df_gradient.hpp"
#include "scf/density_fitting.hpp"

int main(int argc, char** argv) {
  using namespace vibeqc;
  try {
    if (!std::getenv("SLURM_JOB_ID") || !std::getenv("CUDA_VISIBLE_DEVICES") || argc != 7)
      throw std::invalid_argument(
          "run through finite Slurm: probe case mode budget cap rhf|uhf repeats");
    const std::string name = argv[1], mode = argv[2], spin = argv[5];
    const auto budget = std::stoull(argv[3]), cap = std::stoull(argv[4]);
    const int repeats = std::stoi(argv[6]);
    const bool source = mode.ends_with("_source");
    if ((name != "sp8" && name != "sdf18") || (spin != "rhf" && spin != "uhf") || repeats < 5 ||
        (mode != "generated_source" && mode != "generated_resident"))
      throw std::invalid_argument(
          "unsupported probe arguments; reference response requires the archived source");
    if (!budget) throw std::invalid_argument("generated response requires a positive byte budget");
    core::System orbital;
    orbital.atoms = {{1, {0.0, 0.0, -0.7}}, {1, {0.1, 0.2, 0.7}}};
    orbital.shells = name == "sp8" ? std::vector<core::Shell>{{0, 0, {{1.2, 1.0}}},
                                                              {0, 1, {{0.7, 1.0}}},
                                                              {1, 0, {{1.2, 1.0}}},
                                                              {1, 1, {{0.7, 1.0}}}}
                                   : std::vector<core::Shell>{{0, 0, {{1.5, 1.0}}},
                                                              {0, 2, {{0.8, 1.0}}},
                                                              {0, 3, {{0.6, 1.0}}},
                                                              {1, 0, {{1.2, 1.0}}}};
    std::string detail;
    const auto check = [&](vibeqc_status status) {
      if (status != VIBEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
    };
    check(molecule::validate_and_normalize(orbital, detail));
    const auto auxiliary = orbital;
    const auto raw = integrals::build_density_fitting_integrals(orbital, auxiliary);
    const auto n = raw.nbf, a = raw.naux;
    constexpr double threshold = 1e-10;
    std::vector<double> alpha(n * n), beta(n * n), total(n * n), rhf(n * n);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        alpha[i * n + j] = 0.0004 * (i + 1) * (j + 1);
        beta[i * n + j] = ((i + j) % 2 ? -0.0001 : 0.0001) * (i + 1) * (j + 1);
        total[i * n + j] = alpha[i * n + j] + beta[i * n + j];
        rhf[i * n + j] = 2 * alpha[i * n + j];
      }
    const auto expected =
        spin == "rhf"
            ? scf::build_density_fitting_rhf_gradient(raw, rhf, threshold).derivative
            : scf::build_density_fitting_uhf_gradient(raw, alpha, beta, threshold).derivative;
    scf::CudaDensityFittingJkPlan* handle = nullptr;
    std::vector<scf::CudaDensityFittingMetricDiagnostic> diagnostics;
    if (source) {
      scf::CudaDensityFittingIntegralSource* integral_source = nullptr;
      std::vector<double> metrics;
      std::size_t source_n = 0, source_a = 0;
      check(scf::create_cuda_density_fitting_integral_source(
          0, {orbital}, {auxiliary}, &integral_source, metrics, source_n, source_a, detail));
      const auto status = scf::create_cuda_density_fitting_jk_plan_from_source(
          0, &integral_source, 1, n, a, metrics, threshold, std::min<std::size_t>(3, a), n, &handle,
          diagnostics, detail);
      scf::destroy_cuda_density_fitting_integral_source(integral_source);
      check(status);
    } else {
      check(scf::create_cuda_density_fitting_jk_plan(0, 1, n, a, raw.metric, raw.three_center,
                                                     threshold, a, &handle, diagnostics, detail));
    }
    std::unique_ptr<scf::CudaDensityFittingJkPlan,
                    decltype(&scf::destroy_cuda_density_fitting_jk_plan)>
        plan(handle, &scf::destroy_cuda_density_fitting_jk_plan);
    const std::vector<scf::DensityFittingDensityResponse> terms =
        spin == "rhf" ? std::vector<scf::DensityFittingDensityResponse>{{rhf, 1, .25}}
                      : std::vector<scf::DensityFittingDensityResponse>{
                            {total, 1, 0}, {alpha, 0, .5}, {beta, 0, .5}};
    scf::DfGradientResources resources;
    std::vector<double> actual;
    const auto execute = [&] {
      check(scf::execute_cuda_density_fitting_generated_force_response(
          plan.get(), 0, orbital, auxiliary, raw.three_center, raw.metric, terms, 0, budget, cap,
          actual, detail, &resources));
    };
    execute();
    std::vector<double> timings;
    double maximum_error = 0;
    for (int repeat = 0; repeat < repeats; ++repeat) {
      const auto start = std::chrono::steady_clock::now();
      execute();
      timings.push_back(
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
              .count());
      if (actual.size() != expected.size()) throw std::runtime_error("gradient shape mismatch");
      for (std::size_t i = 0; i < actual.size(); ++i)
        maximum_error = std::max(maximum_error, std::abs(actual[i] - expected[i]));
    }
    if (maximum_error > 2e-9 || (resources.host_bytes > budget || resources.device_bytes > budget))
      throw std::runtime_error("response error or staging budget gate failed");
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage)) throw std::runtime_error("getrusage failed");
    std::cout << std::setprecision(17) << "{\"source_identity\":\"" << vibeqc_get_source_identity()
              << "\",\"case\":\"" << name << "\",\"mode\":\"" << mode << "\",\"spin\":\"" << spin
              << "\",\"budget\":" << budget << ",\"maximum_error\":" << maximum_error
              << ",\"slurm_job_id\":\"" << std::getenv("SLURM_JOB_ID")
              << "\",\"raw_derivative_bytes\":"
              << (raw.metric_derivative.size() + raw.three_center_derivative.size()) *
                     sizeof(double)
              << ",\"process_peak_host_bytes_including_oracle\":" << usage.ru_maxrss * 1024ULL
              << ",\"plan_device_resident_bytes\":" << diagnostics[0].device_resident_bytes
              << ",\"response_resources\":";
    std::cout << "{\"host_bytes\":" << resources.host_bytes
              << ",\"device_bytes\":" << resources.device_bytes
              << ",\"h2d_bytes\":" << resources.host_to_device_bytes
              << ",\"d2h_bytes\":" << resources.device_to_host_bytes
              << ",\"synchronizations\":" << resources.stream_synchronizations
              << ",\"uploads\":" << resources.uploads << ",\"derivative_tiles\":" << resources.tiles
              << ",\"weight_tile_elements\":" << resources.weight_tile_elements
              << ",\"auxiliary_weight_tile\":" << resources.auxiliary_weight_tile
              << ",\"value_slices\":" << resources.value_slices << "}";
    std::cout << ",\"milliseconds\":[";
    for (std::size_t i = 0; i < timings.size(); ++i) std::cout << (i ? "," : "") << timings[i];
    std::cout << "]}\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
