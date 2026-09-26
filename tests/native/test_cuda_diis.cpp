/** Independent dot oracle for populated, wrapped and retired DIIS histories.
 * Poisoned unused slots catch reads outside the chronological live window;
 * an inactive neighbor must leave its entire output reservation untouched.
 */
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "scf/cuda/scf_diis_kernels.hpp"

namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template <class T>
struct Device {
  T* data{};
  explicit Device(const std::vector<T>& values) {
    check(cudaMalloc(&data, values.size() * sizeof(T)));
    check(cudaMemcpy(data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  ~Device() { cudaFree(data); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
};

void check_history(unsigned spins, unsigned old_count, unsigned head) {
  constexpr unsigned n = 67, history = 8, batch = 2;
  const std::size_t size = n * n * spins, parts = (size + 4095) / 4096;
  const double poison = std::numeric_limits<double>::quiet_NaN();
  const auto count = std::min(old_count + 1, history);
  const auto first = (head + 1 + history - count) % history;
  const auto live = [&](unsigned slot) { return (slot + history - first) % history < count; };
  std::vector<double> current(batch * size), stored(batch * history * size, poison);
  for (std::size_t i = 0; i < size; ++i) {
    current[i] = std::sin(0.017 * (i + 1));
    for (unsigned slot = 0; slot < history; ++slot)
      if (live(slot) && slot != head)
        stored[slot * size + i] = std::cos(0.011 * (slot + 1) * (i + 1));
  }
  std::vector<double> result(batch * history * history * parts, poison);
  Device<double> d_current(current), d_stored(stored), d_result(result);
  Device<std::uint8_t> d_active(std::vector<std::uint8_t>{1, 0});
  Device<std::uint32_t> d_count(std::vector<std::uint32_t>{old_count, old_count});
  Device<std::uint32_t> d_head(std::vector<std::uint32_t>{head, head});
  vibeqc::scf::cuda_execution::launch_diis_dot_partials(
      nullptr, batch, n, spins, history, d_current.data, d_stored.data, d_active.data, d_count.data,
      d_head.data, parts, d_result.data);
  check(cudaGetLastError());
  check(cudaMemcpy(result.data(), d_result.data, result.size() * sizeof(double),
                   cudaMemcpyDeviceToHost));
  for (unsigned system = 0; system < batch; ++system)
    for (unsigned row = 0; row < history; ++row)
      for (unsigned column = 0; column < history; ++column) {
        const auto offset = ((system * history + row) * history + column) * parts;
        const bool used = system == 0 && count >= 2 && live(row) && live(column);
        if (!used) {
          for (std::size_t part = 0; part < parts; ++part)
            if (!std::isnan(result[offset + part]))
              throw std::runtime_error("DIIS wrote an inactive/unpopulated partial");
          continue;
        }
        const auto* left = row == head ? current.data() : stored.data() + row * size;
        const auto* right = column == head ? current.data() : stored.data() + column * size;
        // Long-double accumulation is independent of the GPU reduction tree.
        long double expected = 0;
        for (std::size_t i = 0; i < size; ++i)
          expected += static_cast<long double>(left[i]) * right[i];
        double actual = 0;
        for (std::size_t part = 0; part < parts; ++part) actual += result[offset + part];
        if (!std::isfinite(actual) || std::abs(actual - expected) > 2e-11L)
          throw std::runtime_error("DIIS partials differ from the independent ring-history dot");
      }
}
}  // namespace

int main() {
  if (!std::getenv("SLURM_JOB_ID")) return 77;
  try {
    check(cudaSetDevice(0));
    for (unsigned spins : {1U, 2U})
      for (unsigned count : {0U, 1U, 2U, 6U, 7U, 8U})
        for (unsigned head : {0U, 3U, 7U}) check_history(spins, count, head);
    std::cout << "validated deterministic DIIS partials and retired circular histories\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
