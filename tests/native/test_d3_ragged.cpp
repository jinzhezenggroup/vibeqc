#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "dft/dispersion/d3_runtime.hpp"

int main(int argc, char**) {
  using namespace vibeqc::dft::dispersion;
  try {
    const bool device = argc > 1;
    constexpr std::size_t atoms = 4100;
    std::vector<std::uint32_t> offsets(atoms + 1);
    std::iota(offsets.begin(), offsets.end(), 0);
    std::vector<std::int32_t> numbers(atoms, 1);
    std::vector<double> coordinates(3 * atoms, 0.0);
    D3Parameters parameters{1.0, 0.95948085, 0.38574991, 4.80688534, 0.0};
    std::string detail;
    vibeqc_status status{};
    auto plan =
        D3Plan::prepare(device ? VIBEQC_BACKEND_CUDA : VIBEQC_BACKEND_CPU_REFERENCE, 0, offsets,
                        numbers, coordinates, parameters, 64U << 20, detail, status);
    if (!plan || status != VIBEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
    const auto expected_workspace = 16 * sizeof(double) * (device ? atoms : 1);
    if (plan->resources().workspace_bytes != expected_workspace)
      throw std::runtime_error("aggregate CUDA workspace incorrectly uses the per-system cap");
    std::vector<std::uint8_t> active(atoms, 1), requested(atoms, 1);
    std::vector<D3Status> statuses;
    std::vector<double> energies, gradients;
    for (int replay = 0; replay < 2; ++replay) {
      if (replay) {
        coordinates[0] = std::numeric_limits<double>::quiet_NaN();
        active[1] = 0;
        requested[2] = 0;
      }
      status = plan->execute(coordinates, active, requested, statuses, energies, gradients, detail);
      if (status != VIBEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
      for (std::size_t i = 0; i < atoms; ++i) {
        const auto expected = replay && i == 0 ? D3Status::invalid_argument : D3Status::success;
        if (statuses[i] != expected || energies[i] != 0.0)
          throw std::runtime_error("ragged status/energy isolation failed");
      }
      if (!std::all_of(gradients.begin(), gradients.end(), [](double v) { return v == 0.0; }))
        throw std::runtime_error("skipped/failed gradient slots must remain zero");
    }
    std::cout << "4100 independent systems, aggregate workspace and masked replay passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
