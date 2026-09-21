#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "d4_eeq_oracle_fixtures.hpp"
#include "dft/dispersion/d4_runtime.hpp"

#if VIBEQC_HAS_CUDA
#include <cuda_runtime.h>
extern "C" void d4_cuda_fail_after_coordinate_upload_for_test_v1();
#endif

using namespace d4_eeq_tests;
using namespace vibeqc::dft::dispersion;

namespace {

bool near(double a, double b, double tolerance) { return std::fabs(a - b) <= tolerance; }

bool cuda_available() {
#if VIBEQC_HAS_CUDA
  int devices = 0;
  return cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
#else
  return false;
#endif
}

std::unique_ptr<D4Plan> plan_for(const EEQOracleFixture& f, vibeqc_backend backend) {
  std::vector<std::uint32_t> offsets{0, static_cast<std::uint32_t>(f.atoms)};
  std::vector<std::int32_t> z(f.z.begin(), f.z.begin() + f.atoms);
  std::vector<double> xyz(f.xyz.begin(), f.xyz.begin() + 3 * f.atoms);
  std::string detail;
  vibeqc_status status{};
  auto plan = D4Plan::prepare(backend, 0, std::move(offsets), std::move(z),
                              std::vector<double>{f.total_charge}, std::move(xyz), f.parameters,
                              f.profile, 64u << 20, detail, status);
  if (!plan || status != VIBEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
  return plan;
}

void execute_one(D4Plan& plan, const std::vector<double>& xyz, bool gradient,
                 std::vector<D4Status>& statuses, std::vector<double>& components,
                 std::vector<double>& gradients, std::vector<double>& charges) {
  const std::uint8_t active = 1;
  const std::uint8_t requested = gradient ? 1 : 0;
  std::string detail;
  const auto status = plan.execute(xyz, std::span(&active, 1), std::span(&requested, 1), statuses,
                                   components, gradients, charges, detail);
  if (status != VIBEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
}

void independent_oracles(vibeqc_backend backend) {
  for (const auto& f : kEEQOracleFixtures) {
    auto plan = plan_for(f, backend);
    std::vector<double> xyz(f.xyz.begin(), f.xyz.begin() + 3 * f.atoms);
    std::vector<D4Status> statuses;
    std::vector<double> components, gradients, charges;
    execute_one(*plan, xyz, true, statuses, components, gradients, charges);
    if (statuses != std::vector<D4Status>{D4Status::success})
      throw std::runtime_error(std::string(f.name) + ": production status failed");
    if (components.size() != 2 || !near(components[0] + components[1], f.energy, 2e-13))
      throw std::runtime_error(std::string(f.name) + ": production energy mismatch");
    for (int i = 0; i < f.atoms; ++i)
      if (!near(charges[i], f.charges[i], 1e-10))
        throw std::runtime_error(std::string(f.name) + ": production charge mismatch");
    for (int i = 0; i < 3 * f.atoms; ++i)
      if (!near(gradients[i], f.gradient[i], 2e-12))
        throw std::runtime_error(std::string(f.name) + ": production gradient mismatch");
  }
}

void finite_difference(vibeqc_backend backend) {
  const auto& f = kEEQOracleFixtures[1];
  auto plan = plan_for(f, backend);
  std::vector<double> xyz(f.xyz.begin(), f.xyz.begin() + 3 * f.atoms);
  std::vector<D4Status> statuses;
  std::vector<double> components, gradient, charges;
  execute_one(*plan, xyz, true, statuses, components, gradient, charges);
  for (double h : {1e-4, 2e-5}) {
    double maximum = 0.0;
    for (int coordinate = 0; coordinate < 3 * f.atoms; ++coordinate) {
      auto plus = xyz, minus = xyz;
      plus[coordinate] += h;
      minus[coordinate] -= h;
      std::vector<double> ep, em, ignored_gradient, ignored_charges;
      execute_one(*plan, plus, false, statuses, ep, ignored_gradient, ignored_charges);
      execute_one(*plan, minus, false, statuses, em, ignored_gradient, ignored_charges);
      const double numerical = ((ep[0] + ep[1]) - (em[0] + em[1])) / (2 * h);
      maximum = std::max(maximum, std::fabs(numerical - gradient[coordinate]));
    }
    if (maximum > 2e-9)
      throw std::runtime_error("production D4 finite-difference gradient gate failed");
  }
}

#if VIBEQC_HAS_CUDA
void changed_geometry_failure_recovery() {
  const auto& f = kEEQOracleFixtures[0];
  auto plan = plan_for(f, VIBEQC_BACKEND_CUDA);
  std::vector<double> original(f.xyz.begin(), f.xyz.begin() + 3 * f.atoms);
  std::vector<D4Status> statuses;
  std::vector<double> baseline, gradients, charges;
  execute_one(*plan, original, false, statuses, baseline, gradients, charges);

  auto changed = original;
  changed[0] += 0.037;
  const std::uint8_t active = 1;
  const std::uint8_t requested = 0;
  std::string detail;
  std::vector<double> failed_components, failed_gradient, failed_charges;
  d4_cuda_fail_after_coordinate_upload_for_test_v1();
  const auto failed =
      plan->execute(changed, std::span(&active, 1), std::span(&requested, 1), statuses,
                    failed_components, failed_gradient, failed_charges, detail);
  if (failed != VIBEQC_STATUS_CUDA_ERROR ||
      detail.find("after coordinate upload") == std::string::npos)
    throw std::runtime_error("D4 post-upload CUDA failure injection did not trigger");

  std::vector<double> recovered_components, recovered_gradient, recovered_charges;
  execute_one(*plan, original, false, statuses, recovered_components, recovered_gradient,
              recovered_charges);
  if (recovered_components.size() != baseline.size())
    throw std::runtime_error("D4 recovery changed the result shape");
  for (std::size_t i = 0; i < baseline.size(); ++i)
    if (!near(recovered_components[i], baseline[i], 2e-13))
      throw std::runtime_error("D4 failed-upload recovery reused poisoned device coordinates");

  const auto bytes = original.size() * sizeof(double);
  const auto& counters = plan->counters();
  if (counters.execution_count != 3 || counters.unchanged_geometry_replays != 1 ||
      counters.changed_geometry_replays != 2 || counters.coordinate_h2d_bytes != 2 * bytes)
    throw std::runtime_error("D4 failed-upload recovery did not force coordinate re-upload");
}
#endif

}  // namespace

int main(int argc, char** argv) {
  try {
    const bool device = argc > 1 && std::string(argv[1]) == "cuda";
    if (device && !cuda_available()) return 77;
    const auto backend = device ? VIBEQC_BACKEND_CUDA : VIBEQC_BACKEND_CPU_REFERENCE;
    independent_oracles(backend);
    finite_difference(backend);
#if VIBEQC_HAS_CUDA
    if (device) changed_geometry_failure_recovery();
#endif
    std::puts(device ? "production D4 CUDA oracle/FD gates passed"
                     : "production D4 CPU oracle/FD gates passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
