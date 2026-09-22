// Independent closed-form two-plane spectral/occupation projectors. Numerical
// boundary gates are exercised through actual generated CUDA kernels.
#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "runtime/cuda_resources.cuh"
#include "scf/cuda/mean_field_setup.hpp"

namespace {
void require(bool result, const char* message) {
  if (!result) throw std::runtime_error(message);
}
}  // namespace

int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
  try {
    using namespace vibeqc::runtime;
    using namespace vibeqc::scf::cuda_execution;
    OwnedCudaStream stream(0);
    OwnedCudaBuffer<double> vectors(0, 32, stream.get()), weights(0, 8, stream.get()),
        output(0, 32, stream.get());
    OwnedCudaBuffer<int> invalid(0, 1, stream.get());
    const auto check = cuda_resource_check;
    std::array<double, 32> c{};
    // Column-major orthogonal plane rotation, duplicated for alpha and beta.
    for (std::size_t spin = 0; spin < 2; ++spin) {
      auto* block = c.data() + spin * 16;
      block[0] = 0.6;
      block[1] = 0.8;
      block[4] = -0.8;
      block[5] = 0.6;
      block[10] = block[15] = 1.0;
    }
    check(cudaMemcpy(vectors.get(), c.data(), sizeof(c), cudaMemcpyHostToDevice));
    std::array<double, 8> values{1e-10, 0.5, 1.0, 4.0, 0, 0, 0, 0};
    check(cudaMemcpy(weights.get(), values.data(), sizeof(values), cudaMemcpyHostToDevice));
    check(cudaMemset(invalid.get(), 0, sizeof(int)));
    form_overlap_weights(stream.get(), 4, weights.get(), invalid.get());
    form_weighted_projector(stream.get(), 4, 1, vectors.get(), weights.get(), output.get());
    stream.synchronize();
    int error = -1;
    check(cudaMemcpy(&error, invalid.get(), sizeof(error), cudaMemcpyDeviceToHost));
    require(error == 0, "exact overlap cutoff was incorrectly rejected");
    std::array<double, 32> actual{};
    check(cudaMemcpy(actual.data(), output.get(), 16 * sizeof(double), cudaMemcpyDeviceToHost));
    const double a = 1.0 / std::sqrt(1e-10), b = std::sqrt(2.0);
    std::array<double, 16> expected{};
    expected[0] = 0.36 * a + 0.64 * b;
    expected[1] = expected[4] = 0.48 * (a - b);
    expected[5] = 0.64 * a + 0.36 * b;
    expected[10] = 1.0;
    expected[15] = 0.5;
    for (std::size_t i = 0; i < 16; ++i)
      require(std::abs(actual[i] - expected[i]) <= 2e-11,
              "inverse-square-root projector differs from analytic rotation");

    for (double bad : {std::nextafter(1e-10, 0.0), -1.0, std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity()}) {
      values[0] = bad;
      check(cudaMemcpy(weights.get(), values.data(), sizeof(values), cudaMemcpyHostToDevice));
      check(cudaMemset(invalid.get(), 0, sizeof(int)));
      form_overlap_weights(stream.get(), 4, weights.get(), invalid.get());
      stream.synchronize();
      check(cudaMemcpy(&error, invalid.get(), sizeof(error), cudaMemcpyDeviceToHost));
      require(error == 1, "invalid overlap spectrum was accepted");
    }
    form_occupation_weights(stream.get(), 4, 2, 1, 0, weights.get());
    form_weighted_projector(stream.get(), 4, 2, vectors.get(), weights.get(), output.get());
    stream.synchronize();
    check(cudaMemcpy(actual.data(), output.get(), sizeof(actual), cudaMemcpyDeviceToHost));
    expected = {};
    expected[0] = 0.36;
    expected[1] = expected[4] = 0.48;
    expected[5] = 0.64;
    for (std::size_t i = 0; i < 16; ++i) {
      require(std::abs(actual[i] - expected[i]) < 1e-15, "alpha projector mismatch");
      require(actual[i + 16] == 0.0, "unoccupied beta density is nonzero");
    }
    form_occupation_weights(stream.get(), 4, 1, 1, 1, weights.get());
    form_weighted_projector(stream.get(), 4, 1, vectors.get(), weights.get(), output.get());
    stream.synchronize();
    check(cudaMemcpy(actual.data(), output.get(), 16 * sizeof(double), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < 16; ++i)
      require(std::abs(actual[i] - 2 * expected[i]) < 1e-15, "restricted occupation mismatch");
    for (double off_diagonal : {0.5e-8, 2e-8}) {
      expected = {};
      for (std::size_t i = 0; i < 4; ++i) expected[5 * i] = 1.0;
      expected[1] = off_diagonal;
      check(cudaMemcpy(output.get(), expected.data(), sizeof(expected), cudaMemcpyHostToDevice));
      check(cudaMemset(invalid.get(), 0, sizeof(int)));
      check_overlap_metric(stream.get(), 4, output.get(), invalid.get());
      stream.synchronize();
      check(cudaMemcpy(&error, invalid.get(), sizeof(error), cudaMemcpyDeviceToHost));
      require(error == (off_diagonal > 1e-8 ? 1 : 0), "overlap metric gate changed");
    }
    std::cout << "CUDA setup analytic projector/spectrum/occupation gates passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
