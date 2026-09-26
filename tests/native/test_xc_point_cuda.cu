#include <cuda_runtime.h>

#include <stdexcept>

#include "dft/xc_point.hpp"

namespace {
void check_point_cuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct Input {
  bool pbe;
  double rho[2], gradient[2][3];
};

__global__ void point_kernel(const Input* input, vibeqc::dft::point::Value* output) {
  *output = vibeqc::dft::point::evaluate(input->pbe, input->rho, input->gradient);
}

/** Test-only transfer wrapper: this is point arithmetic evidence, separately
 * from the native device-buffer pipeline and complete SCF timing evidence. */
vibeqc::dft::point::Value evaluate_device(bool pbe, const double rho[2],
                                          const double gradient[2][3]) {
  struct Buffers {
    Input* input{};
    vibeqc::dft::point::Value* output{};
    Buffers() {
      check_point_cuda(cudaMalloc(&input, sizeof(Input)));
      try {
        check_point_cuda(cudaMalloc(&output, sizeof(*output)));
      } catch (...) {
        cudaFree(input);
        throw;
      }
    }
    ~Buffers() {
      cudaFree(output);
      cudaFree(input);
    }
  };
  static Buffers buffers;
  Input input{};
  input.pbe = pbe;
  for (unsigned s = 0; s < 2; ++s) {
    input.rho[s] = rho[s];
    for (unsigned k = 0; k < 3; ++k) input.gradient[s][k] = gradient[s][k];
  }
  check_point_cuda(cudaMemcpy(buffers.input, &input, sizeof(input), cudaMemcpyHostToDevice));
  point_kernel<<<1, 1>>>(buffers.input, buffers.output);
  check_point_cuda(cudaGetLastError());
  vibeqc::dft::point::Value result;
  check_point_cuda(cudaMemcpy(&result, buffers.output, sizeof(result), cudaMemcpyDeviceToHost));
  return result;
}
}  // namespace

#define VIBEQC_TEST_POINT_EVALUATE evaluate_device
#define main point_test_main
#include "test_xc_point.cpp"
#undef main

int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
  return point_test_main();
}
