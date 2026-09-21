#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/dispersion/d4_runtime.hpp"
#include "generated_method_parameters.hpp"

#if VIBEQC_HAS_CUDA
#include <cuda_runtime.h>
#endif

using namespace vibeqc::dft::dispersion;

namespace {
bool cuda_available() {
#if VIBEQC_HAS_CUDA
  int devices = 0;
  return cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
#else
  return false;
#endif
}
}  // namespace

int main(int argc, char**) {
  try {
    const bool device = argc > 1;
    if (device && !cuda_available()) return 77;
    constexpr std::size_t systems = 4100;
    std::vector<std::uint32_t> offsets(systems + 1);
    std::iota(offsets.begin(), offsets.end(), 0);
    std::vector<std::int32_t> numbers(systems, 1);
    std::vector<double> total_charges(systems, 0.0);
    std::vector<double> coordinates(3 * systems, 0.0);
    const auto p = ::vibeqc::generated::method_parameters::pbeD4();
    D4Parameters parameters{D4ReferenceModel::eeq, p.s6,          p.s8,         p.s9, p.a1, p.a2,
                            p.cn_cutoff,           p.pair_cutoff, p.atm_cutoff, p.ga, p.gc};
    std::string detail;
    vibeqc_status status{};
    auto plan = D4Plan::prepare(device ? VIBEQC_BACKEND_CUDA : VIBEQC_BACKEND_CPU_REFERENCE, 0,
                                offsets, numbers, total_charges, coordinates, parameters,
                                D4EEQProfile::standard, 64u << 20, detail, status);
    if (!plan || status != VIBEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
    const auto slots = device ? std::uint32_t{32} : std::uint32_t{1};
    const auto expected_workspace =
        device ? (slots * (eeq2019_workspace_elements(1) + 3u) + 27u * systems) * sizeof(double)
               : complete_d4_eeq_workspace_elements(1) * sizeof(double);
    if (plan->resources().worker_blocks != slots || plan->resources().workspace_slots != slots ||
        plan->resources().workspace_bytes != expected_workspace)
      throw std::runtime_error("D4 workspace is not bounded by worker slots");
    const auto minimum_execution_host =
        3 * systems * sizeof(double) + 2 * systems * sizeof(std::uint8_t) +
        systems * sizeof(vibeqc_status) + systems * sizeof(D4Status) +
        2 * systems * sizeof(double) + 3 * systems * sizeof(double) + systems * sizeof(double);
    if (plan->resources().execution_host_bytes < minimum_execution_host)
      throw std::runtime_error("D4 public replay staging is missing from host resource accounting");

    std::vector<std::uint8_t> active(systems, 1), requested(systems, 1);
    std::vector<D4Status> statuses;
    std::vector<double> components, gradients, eeq_charges;
    for (int replay = 0; replay < 2; ++replay) {
      if (replay) {
        coordinates[0] = std::numeric_limits<double>::quiet_NaN();
        active[1] = 0;
        requested[2] = 0;
      }
      status = plan->execute(coordinates, active, requested, statuses, components, gradients,
                             eeq_charges, detail);
      if (status != VIBEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
      for (std::size_t i = 0; i < systems; ++i) {
        const auto expected = replay && i == 0 ? D4Status::invalid_argument : D4Status::success;
        if (statuses[i] != expected)
          throw std::runtime_error("D4 ragged peer-local status isolation failed");
        if (i != 0 && (components[2 * i] != 0.0 || components[2 * i + 1] != 0.0))
          throw std::runtime_error("one-atom D4 correction must be zero");
      }
      if (!std::all_of(gradients.begin(), gradients.end(),
                       [](double value) { return value == 0.0; }))
        throw std::runtime_error("D4 failed/skipped gradient publication was not zero");
    }
    const auto& counters = plan->counters();
    if (counters.execution_count != 2 || counters.unchanged_geometry_replays != 1 ||
        counters.changed_geometry_replays != 1)
      throw std::runtime_error("D4 replay counters are inconsistent");
    if (device && (counters.kernel_launches != 18 ||
                   counters.coordinate_h2d_bytes != coordinates.size() * sizeof(double)))
      throw std::runtime_error("D4 CUDA replay movement/launch accounting is inconsistent");
    std::cout << "4100 D4 systems: bounded worker workspace, replay and isolation passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
