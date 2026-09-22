#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vibeqc/vibeqc.h"

namespace {

struct Context {
  vibeqc_context* value{};
  Context() {
    vibeqc_context_descriptor descriptor{sizeof(vibeqc_context_descriptor), VIBEQC_ABI_VERSION, 0,
                                         VIBEQC_BACKEND_CUDA};
    if (vibeqc_context_create(&descriptor, &value) != VIBEQC_STATUS_SUCCESS)
      throw std::runtime_error("failed to create CUDA context");
  }
  ~Context() { vibeqc_context_destroy(value); }
};

struct Fleet {
  std::vector<std::vector<std::int32_t>> numbers;
  std::vector<std::vector<double>> coordinates;
  std::vector<vibeqc_d3_system_descriptor> descriptors;
};

Fleet make_fleet(std::size_t systems, std::size_t atoms) {
  Fleet fleet;
  fleet.numbers.reserve(systems);
  fleet.coordinates.reserve(systems);
  fleet.descriptors.reserve(systems);
  constexpr std::int32_t species[] = {6, 8, 7, 1};
  for (std::size_t system = 0; system < systems; ++system) {
    fleet.numbers.emplace_back(atoms);
    fleet.coordinates.emplace_back(3 * atoms);
    auto& z = fleet.numbers.back();
    auto& xyz = fleet.coordinates.back();
    for (std::size_t atom = 0; atom < atoms; ++atom) {
      z[atom] = species[(atom + system) % 4];
      const double col = static_cast<double>(atom % 4);
      const double row = static_cast<double>((atom / 4) % 4);
      const double layer = static_cast<double>(atom / 16);
      xyz[3 * atom] = 2.35 * col + 0.003 * static_cast<double>(system % 7);
      xyz[3 * atom + 1] = 2.55 * row + 0.004 * static_cast<double>(system % 5);
      xyz[3 * atom + 2] = 2.75 * layer + 0.002 * static_cast<double>(system % 11);
    }
  }
  for (std::size_t system = 0; system < systems; ++system) {
    fleet.descriptors.push_back(vibeqc_d3_system_descriptor{
        sizeof(vibeqc_d3_system_descriptor), VIBEQC_ABI_VERSION, fleet.numbers[system].data(),
        fleet.coordinates[system].data(), static_cast<std::uint32_t>(atoms)});
  }
  return fleet;
}

vibeqc_d3_bj_descriptor model() {
  return {sizeof(vibeqc_d3_bj_descriptor),
          VIBEQC_ABI_VERSION,
          VIBEQC_D3_DAMPING_BJ,
          1.0,
          0.7875,
          0.4289,
          4.4407,
          0.0,
          0.0,
          0.0,
          0.0,
          512u << 20,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0};
}

double percentile(std::vector<double> values, double fraction) {
  std::sort(values.begin(), values.end());
  const std::size_t index =
      std::min(values.size() - 1, static_cast<std::size_t>(fraction * values.size()));
  return values[index];
}

void run_mode(vibeqc_d3_batch* batch, const Fleet& fleet, bool gradient, int warmups,
              int iterations) {
  const std::size_t systems = fleet.descriptors.size();
  const std::size_t atoms = fleet.numbers.front().size();
  std::vector<std::vector<double>> gradients(systems);
  std::vector<vibeqc_d3_batch_item_result_descriptor> results(systems);
  for (std::size_t system = 0; system < systems; ++system) {
    gradients[system].resize(gradient ? 3 * atoms : 0);
    results[system] = vibeqc_d3_batch_item_result_descriptor{
        sizeof(vibeqc_d3_batch_item_result_descriptor),
        VIBEQC_ABI_VERSION,
        VIBEQC_STATUS_INTERNAL_ERROR,
        0.0,
        gradient ? gradients[system].data() : nullptr,
        gradient ? static_cast<std::uint32_t>(gradients[system].size()) : 0u,
        VIBEQC_BACKEND_CUDA};
  }

  auto execute = [&] {
    const auto status = vibeqc_d3_batch_execute(batch, nullptr, 0, results.data(),
                                                static_cast<std::uint32_t>(results.size()));
    if (status != VIBEQC_STATUS_SUCCESS) throw std::runtime_error("benchmark batch failed");
    for (const auto& result : results)
      if (result.status != VIBEQC_STATUS_SUCCESS || result.executed_backend != VIBEQC_BACKEND_CUDA)
        throw std::runtime_error("benchmark item failed");
  };

  for (int i = 0; i < warmups; ++i) execute();
  std::vector<double> milliseconds;
  milliseconds.reserve(iterations);
  double checksum = 0.0;
  for (int i = 0; i < iterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    execute();
    const auto stop = std::chrono::steady_clock::now();
    milliseconds.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    for (std::size_t system = 0; system < systems; ++system) {
      checksum += results[system].energy;
      if (gradient) checksum += gradients[system][(i + system) % gradients[system].size()];
    }
  }

  std::cout << std::setprecision(12) << "mode=" << (gradient ? "gradient" : "energy")
            << " systems=" << systems << " atoms=" << atoms
            << " median_ms=" << percentile(milliseconds, 0.5)
            << " p95_ms=" << percentile(milliseconds, 0.95) << " checksum=" << checksum << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::size_t systems = argc > 1 ? std::stoul(argv[1]) : 64;
    const std::size_t atoms = argc > 2 ? std::stoul(argv[2]) : 64;
    const int iterations = argc > 3 ? std::stoi(argv[3]) : 60;
    const int warmups = argc > 4 ? std::stoi(argv[4]) : 10;
    if (systems == 0 || atoms < 2 || iterations < 3 || warmups < 0)
      throw std::runtime_error("invalid benchmark arguments");

    const auto fleet = make_fleet(systems, atoms);
    Context context;
    auto parameters = model();
    vibeqc_d3_batch* batch{};
    const auto status = vibeqc_d3_batch_prepare(
        context.value, fleet.descriptors.data(),
        static_cast<std::uint32_t>(fleet.descriptors.size()), &parameters, &batch);
    if (status != VIBEQC_STATUS_SUCCESS) {
      const auto* detail = vibeqc_context_get_last_detail(context.value);
      throw std::runtime_error(detail ? detail : "benchmark prepare failed");
    }
    run_mode(batch, fleet, true, warmups, iterations);
    run_mode(batch, fleet, false, warmups, iterations);
    vibeqc_d3_batch_destroy(batch);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
