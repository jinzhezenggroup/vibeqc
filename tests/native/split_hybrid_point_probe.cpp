// Validation executable only: no production dependency on the reference oracle.
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if VIBEQC_VALIDATE_CUDA
#include <cuda_runtime.h>
#define VIBEQC_PROBE_DEVICE __device__
#else
#define __device__
#define VIBEQC_PROBE_DEVICE
#endif
#include "m06_2x_point.cuh"
#include "mn15_point.cuh"
#if !VIBEQC_VALIDATE_CUDA
#undef __device__
#endif

VIBEQC_PROBE_DEVICE void evaluate_point(int code, const double* x, double* y) {
  if (code == 450) {
    const auto value =
        vibeqc::dft::generated::m06_2x_device(x[0], x[1], x[2], x[3], x[4], x[5], x[6]);
    y[0] = value.energy_density;
    for (unsigned i = 0; i < 7; ++i) y[i + 1] = value.feature_derivative[i];
  } else {
    const auto value =
        vibeqc::dft::generated::mn15_device(x[0], x[1], x[2], x[3], x[4], x[5], x[6]);
    y[0] = value.energy_density;
    for (unsigned i = 0; i < 7; ++i) y[i + 1] = value.feature_derivative[i];
  }
}

#if VIBEQC_VALIDATE_CUDA
void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct DeviceBuffer {
  double* data{};
  explicit DeviceBuffer(std::size_t count) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(double)));
  }
  ~DeviceBuffer() {
    if (data) cudaFree(data);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
__global__ void evaluate_points(int code, int count, const double* x, double* y) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) evaluate_point(code, x + 7 * i, y + 8 * i);
}
#endif

int main(int argc, char** argv) {
  try {
    if (argc != 5) throw std::runtime_error("usage: probe CODE COUNT INPUT OUTPUT");
    const int code = std::stoi(argv[1]);
    const int count = std::stoi(argv[2]);
    if ((code != 450 && code != 268) || count < 1 || count > 65536)
      throw std::runtime_error("invalid point probe request");
    std::vector<double> input(7 * count), output(8 * count);
    std::ifstream source(argv[3], std::ios::binary);
    if (!source.read(reinterpret_cast<char*>(input.data()), input.size() * sizeof(double)))
      throw std::runtime_error("point input is missing or truncated");
#if VIBEQC_VALIDATE_CUDA
    int devices = 0;
    check_cuda(cudaGetDeviceCount(&devices));
    if (devices < 1) throw std::runtime_error("CUDA acceptance requires a real CUDA device");
    check_cuda(cudaSetDevice(0));
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, 0));
    int driver = 0, runtime = 0;
    check_cuda(cudaDriverGetVersion(&driver));
    check_cuda(cudaRuntimeGetVersion(&runtime));
    std::cerr << "CUDA device=" << properties.name << " cc=" << properties.major << '.'
              << properties.minor << " driver=" << driver << " runtime=" << runtime << '\n';
    DeviceBuffer x(input.size()), y(output.size());
    check_cuda(
        cudaMemcpy(x.data, input.data(), input.size() * sizeof(double), cudaMemcpyHostToDevice));
    evaluate_points<<<(count + 127) / 128, 128>>>(code, count, x.data, y.data);
    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());
    check_cuda(
        cudaMemcpy(output.data(), y.data, output.size() * sizeof(double), cudaMemcpyDeviceToHost));
#else
    for (int i = 0; i < count; ++i)
      evaluate_point(code, input.data() + 7 * i, output.data() + 8 * i);
#endif
    std::ofstream destination(argv[4], std::ios::binary);
    if (!destination.write(reinterpret_cast<const char*>(output.data()),
                           output.size() * sizeof(double)))
      throw std::runtime_error("cannot write point output");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
