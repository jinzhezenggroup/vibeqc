#include <cuda_runtime_api.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "scf/cuda/eigensolver.hpp"
#include "scf/eigensolver_workspace.hpp"

namespace {
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void check(cudaError_t status) { require(status == cudaSuccess, cudaGetErrorString(status)); }
struct Owner {
  cudaStream_t stream{};
  std::vector<void*> buffers;
  Owner() { check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
  ~Owner() {
    (void)cudaStreamSynchronize(stream);
    for (auto* p : buffers) (void)cudaFree(p);
    (void)cudaStreamDestroy(stream);
  }
  template <class T>
  T* allocate(std::size_t count) {
    T* p{};
    check(cudaMalloc(reinterpret_cast<void**>(&p), count * sizeof(T)));
    buffers.push_back(p);
    return p;
  }
};

/** A=Q diag(d) Q^T with independently constructed Householder Q. Its analytic
 * rank-two action checks every eigenvector in O(n^2), without a second native
 * solver or a cubic host workload in the 768-AO acceptance case.
 */
void verify(int n) {
  const auto matrix = std::size_t(n) * n;
  std::vector<double> u(n), d(n), a(matrix);
  double norm = 0, ud = 0;
  for (int i = 0; i < n; ++i) {
    u[i] = 1 + i % 7;
    norm += u[i] * u[i];
  }
  for (int i = 0; i < n; ++i) {
    u[i] /= std::sqrt(norm);
    d[i] = -3 + 5.0 * i / n;
    ud += u[i] * u[i] * d[i];
  }
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      a[i + std::size_t(j) * n] =
          (i == j ? d[i] : 0) + u[i] * u[j] * (4 * ud - 2 * d[i] - 2 * d[j]);
  Owner owner;
  auto* input = owner.allocate<double>(matrix * 2);
  auto* scratch = owner.allocate<double>(matrix * 2);
  auto* values = owner.allocate<double>(n * 2);
  auto* info = owner.allocate<int>(2);
  auto* active = owner.allocate<std::uint8_t>(2);
  const std::uint8_t masks[2]{1, 1};
  check(cudaMemcpyAsync(active, masks, sizeof(masks), cudaMemcpyHostToDevice, owner.stream));
  vibeqc::scf::cuda_execution::OrdinaryStreamEigensolver solver(owner.stream, n, input, values);
  const auto bound = vibeqc::scf::ordinary_eigensolver_workspace_allowance(n);
  require(solver.device_bytes() <= bound && solver.host_bytes() <= bound,
          "solver query exceeded the shape bound");
  for (int repeat = 0; repeat < 2; ++repeat) {
    for (int spin = 0; spin < 2; ++spin)
      check(cudaMemcpyAsync(input + spin * matrix, a.data(), matrix * sizeof(double),
                            cudaMemcpyHostToDevice, owner.stream));
    require(solver.launch(2, input, scratch, values, info, active) == VIBEQC_STATUS_SUCCESS,
            "ordinary stream eigensolver failed");
    std::vector<double> v(matrix * 2), w(n * 2);
    int status[2]{-1, -1};
    check(cudaMemcpyAsync(v.data(), input, v.size() * sizeof(double), cudaMemcpyDeviceToHost,
                          owner.stream));
    check(cudaMemcpyAsync(w.data(), values, w.size() * sizeof(double), cudaMemcpyDeviceToHost,
                          owner.stream));
    check(cudaMemcpyAsync(status, info, sizeof(status), cudaMemcpyDeviceToHost, owner.stream));
    check(cudaStreamSynchronize(owner.stream));
    for (int spin = 0; spin < 2; ++spin) {
      require(status[spin] == 0, "eigensolver info reported failure");
      for (int j = 0; j < n; ++j) {
        require(std::abs(w[spin * n + j] - d[j]) < 2e-11, "independent eigenvalue mismatch");
        const auto* column = v.data() + spin * matrix + std::size_t(j) * n;
        long double uv = 0, duv = 0, length = 0, overlap = 0;
        for (int i = 0; i < n; ++i) {
          uv += u[i] * column[i];
          duv += u[i] * d[i] * column[i];
          length += column[i] * column[i];
          overlap += column[i] * ((i == j ? 1.0 : 0.0) - 2 * u[i] * u[j]);
        }
        require(std::abs(length - 1) < 2e-11 && std::abs(std::abs(overlap) - 1) < 2e-10,
                "independent normalized eigenvector mismatch");
        for (int i = 0; i < n; ++i) {
          const auto av = d[i] * column[i] + u[i] * ((4 * ud - 2 * d[i]) * uv - 2 * duv);
          require(std::abs(av - w[spin * n + j] * column[i]) < 2e-11,
                  "independent eigen residual mismatch");
        }
      }
    }
  }
  if (n > vibeqc::scf::cuda_execution::kSmallEigensolverLimit) {
    // Library providers still receive the full batch. The common dispatcher
    // must sanitize an inactive nonfinite matrix before calling cuSOLVER.
    const std::uint8_t selected[2]{1, 0};
    std::vector<double> invalid(matrix, std::numeric_limits<double>::quiet_NaN());
    check(
        cudaMemcpyAsync(active, selected, sizeof(selected), cudaMemcpyHostToDevice, owner.stream));
    check(cudaMemcpyAsync(input, a.data(), matrix * sizeof(double), cudaMemcpyHostToDevice,
                          owner.stream));
    check(cudaMemcpyAsync(input + matrix, invalid.data(), matrix * sizeof(double),
                          cudaMemcpyHostToDevice, owner.stream));
    require(solver.launch(2, input, scratch, values, info, active) == VIBEQC_STATUS_SUCCESS,
            "inactive provider state failed");
    std::vector<double> inactive(n);
    int status = -1;
    check(cudaMemcpyAsync(inactive.data(), values + n, n * sizeof(double), cudaMemcpyDeviceToHost,
                          owner.stream));
    check(cudaMemcpyAsync(&status, info + 1, sizeof(status), cudaMemcpyDeviceToHost, owner.stream));
    check(cudaStreamSynchronize(owner.stream));
    require(status == 0, "inactive sanitized provider reported failure");
    for (double value : inactive) require(value == 1.0, "inactive matrix was not sanitized");
  }
  check(cudaStreamBeginCapture(owner.stream, cudaStreamCaptureModeThreadLocal));
  require(solver.launch(2, input, scratch, values, info, active) == VIBEQC_STATUS_INVALID_ARGUMENT,
          "ordinary solver silently accepted graph capture");
  cudaGraph_t graph{};
  check(cudaStreamEndCapture(owner.stream, &graph));
  check(cudaGraphDestroy(graph));
}
}  // namespace

int main() {
  try {
    for (int n : {7, 24, 192, 768}) verify(n);
    std::cout << "ordinary eigensolver independent spectrum/residual/replay/capture PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
