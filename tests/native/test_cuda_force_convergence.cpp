/** Physical stationarity must survive tiny DIIS steps, localized spin errors,
 * nonfinite entries and inactive neighbors. Run only in a Slurm allocation. */
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "scf/cuda/scf_convergence_kernels.hpp"

namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template <class T>
struct Device {
  T* data{};
  std::size_t size{};
  explicit Device(const std::vector<T>& values) : size(values.size()) {
    check(cudaMalloc(&data, size * sizeof(T)));
    check(cudaMemcpy(data, values.data(), size * sizeof(T), cudaMemcpyHostToDevice));
  }
  ~Device() { cudaFree(data); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  std::vector<T> read() const {
    std::vector<T> values(size);
    check(cudaMemcpy(values.data(), data, size * sizeof(T), cudaMemcpyDeviceToHost));
    return values;
  }
};

void check_convergence(unsigned spins, bool retain, bool require_physical, bool approximate,
                       double tolerance) {
  using namespace vibeqc::scf::cuda_execution;
  constexpr unsigned n = 8, batch = 6;
  const std::size_t size = spins * n * n;
  const double bound = std::min(1e-8, tolerance);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> residual(batch * size, 0);
  const double peaks[]{bound / 2, bound, 3 * bound, nan, std::numeric_limits<double>::infinity(),
                       nan};
  for (unsigned i = 0; i < batch; ++i) residual[(i + 1) * size - 1] = peaks[i];
  Device<double> d_residual(residual), density(std::vector<double>(batch * size, 0.25)),
      next(std::vector<double>(batch * size, 0.25 + 1e-12)), energy(std::vector<double>(batch, -1)),
      previous(std::vector<double>(batch, -1)), change(std::vector<double>(batch, nan)),
      rms(std::vector<double>(batch, nan));
  Device<std::uint8_t> active(std::vector<std::uint8_t>{1, 1, 1, 1, 1, 0}),
      converged(std::vector<std::uint8_t>(batch, 0));
  Device<std::uint32_t> iterations(std::vector<std::uint32_t>(batch, 1));
  Device<std::uint32_t> census(std::vector<std::uint32_t>{0, 0, 1, 1, 0, 0});
  const auto launch =
      spins == 1 ? launch_update_convergence_kernel : launch_update_uhf_convergence_kernel;
  launch(retain, batch, 32, 0, nullptr, batch, n, 1e-12, tolerance, false, energy.data,
         previous.data, next.data, density.data, active.data, converged.data, iterations.data,
         change.data, rms.data, require_physical ? d_residual.data : nullptr,
         approximate ? census.data : nullptr);
  check(cudaGetLastError());
  const auto flags = converged.read();
  const auto counts = iterations.read();
  const auto density_out = density.read();
  for (unsigned i = 0; i < batch; ++i) {
    const bool expected = i < 5 && (!require_physical || i < 2 || (approximate && i < 4));
    if (flags[i] != expected || counts[i] != (i == 5 ? 1U : 2U))
      throw std::runtime_error("tiny density step bypassed the physical force criterion");
    const double expected_density = i == 5 || (retain && expected) ? 0.25 : 0.25 + 1e-12;
    if (density_out[i * size] != expected_density)
      throw std::runtime_error("physical convergence changed density-retention semantics");
  }

  // The rebuilt force determinant must be checked separately: iteration
  // acceptance says nothing about a later projection's physical residual.
  Device<std::uint8_t> final_active(std::vector<std::uint8_t>{1, 1, 1, 1, 1, 0}),
      final_converged(std::vector<std::uint8_t>{1, 1, 1, 1, 1, 0});
  Device<std::uint32_t> tested(std::vector<std::uint32_t>{0});
  launch_validate_force_residual_kernel(nullptr, batch, spins, n, tolerance, d_residual.data,
                                        final_active.data, final_converged.data, tested.data);
  check(cudaGetLastError());
  if (final_converged.read() != std::vector<std::uint8_t>{1, 1, 0, 0, 0, 0} ||
      final_active.read() != std::vector<std::uint8_t>{1, 1, 0, 0, 0, 0} || tested.read()[0] != 5)
    throw std::runtime_error("force validation missed a bad determinant or lost rejected work");
}
}  // namespace

int main() {
  if (!std::getenv("SLURM_JOB_ID")) return 77;
  try {
    check(cudaSetDevice(0));
    for (unsigned spins : {1U, 2U})
      for (bool retain : {false, true})
        for (bool physical : {false, true})
          for (bool approximate : {false, true})
            for (double tolerance : {1e-10, 1e-6})
              check_convergence(spins, retain, physical, approximate, tolerance);
    std::cout << "Physical force convergence and final-state rejection passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
