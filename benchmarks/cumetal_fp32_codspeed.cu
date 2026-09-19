#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr unsigned int kElements = 1U << 20;
constexpr unsigned int kThreads = 256;
constexpr unsigned int kInnerIterations = 64;
constexpr unsigned int kLaunchesPerSample = 128;
constexpr unsigned int kWarmupLaunches = 4;

extern "C" __global__ void vibeqc_cumetal_fp32_contract(const float* lhs, const float* rhs,
                                                        float* output, unsigned int size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) return;

  float x = lhs[index];
  float y = rhs[index];
  float accumulator = 0.125F;
#pragma unroll 4
  for (unsigned int iteration = 0; iteration < kInnerIterations; ++iteration) {
    accumulator += x * y;
    x = x * 0.99991F + 0.00013F;
    y = y * 1.00007F - 0.00011F;
  }
  output[index] = accumulator;
}

bool check_cuda(cudaError_t status, const char* expression) {
  if (status == cudaSuccess) return true;
  std::fprintf(stderr, "FAIL: %s: %s\n", expression, cudaGetErrorString(status));
  return false;
}

float reference_value(float x, float y) {
  float accumulator = 0.125F;
  for (unsigned int iteration = 0; iteration < kInnerIterations; ++iteration) {
    accumulator += x * y;
    x = x * 0.99991F + 0.00013F;
    y = y * 1.00007F - 0.00011F;
  }
  return accumulator;
}

class BenchmarkServer {
 public:
  bool initialize() {
    if (!check_cuda(cudaSetDevice(0), "cudaSetDevice(0)")) return false;
    if (!check_cuda(cudaGetDeviceProperties(&properties_, 0), "cudaGetDeviceProperties")) {
      return false;
    }
    if (!check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                    "cudaStreamCreateWithFlags")) {
      return false;
    }
    if (!check_cuda(cudaEventCreate(&begin_), "cudaEventCreate(begin)")) return false;
    if (!check_cuda(cudaEventCreate(&end_), "cudaEventCreate(end)")) return false;

    std::vector<float> lhs(kElements);
    std::vector<float> rhs(kElements);
    for (unsigned int index = 0; index < kElements; ++index) {
      lhs[index] = 0.5F + static_cast<float>(index % 97U) * 0.001F;
      rhs[index] = 1.25F - static_cast<float>(index % 89U) * 0.0005F;
    }

    const std::size_t bytes = static_cast<std::size_t>(kElements) * sizeof(float);
    if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&lhs_), bytes), "cudaMalloc(lhs)") ||
        !check_cuda(cudaMalloc(reinterpret_cast<void**>(&rhs_), bytes), "cudaMalloc(rhs)") ||
        !check_cuda(cudaMalloc(reinterpret_cast<void**>(&output_), bytes), "cudaMalloc(output)") ||
        !check_cuda(cudaMemcpyAsync(lhs_, lhs.data(), bytes, cudaMemcpyHostToDevice, stream_),
                    "cudaMemcpyAsync(lhs)") ||
        !check_cuda(cudaMemcpyAsync(rhs_, rhs.data(), bytes, cudaMemcpyHostToDevice, stream_),
                    "cudaMemcpyAsync(rhs)") ||
        !check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize(upload)")) {
      return false;
    }

    for (unsigned int warmup = 0; warmup < kWarmupLaunches; ++warmup) launch_once();
    if (!check_cuda(cudaGetLastError(), "warmup kernel launch") ||
        !check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize(warmup)")) {
      return false;
    }

    float actual = 0.0F;
    if (!check_cuda(cudaMemcpy(&actual, output_, sizeof(float), cudaMemcpyDeviceToHost),
                    "cudaMemcpy(validation)")) {
      return false;
    }
    const float expected = reference_value(lhs.front(), rhs.front());
    const float tolerance = 2.0e-3F * std::fmax(1.0F, std::fabs(expected));
    if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
      std::fprintf(stderr, "FAIL: FP32 validation actual=%.8g expected=%.8g tolerance=%.8g\n",
                   static_cast<double>(actual), static_cast<double>(expected),
                   static_cast<double>(tolerance));
      return false;
    }

    std::cout << "READY device=" << properties_.name << " elements=" << kElements
              << " launches=" << kLaunchesPerSample << " inner=" << kInnerIterations << std::endl;
    return true;
  }

  bool run_sample(float* elapsed_ms) {
    if (!check_cuda(cudaEventRecord(begin_, stream_), "cudaEventRecord(begin)")) return false;
    for (unsigned int launch = 0; launch < kLaunchesPerSample; ++launch) launch_once();
    if (!check_cuda(cudaGetLastError(), "benchmark kernel launch") ||
        !check_cuda(cudaEventRecord(end_, stream_), "cudaEventRecord(end)") ||
        !check_cuda(cudaEventSynchronize(end_), "cudaEventSynchronize(end)") ||
        !check_cuda(cudaEventElapsedTime(elapsed_ms, begin_, end_), "cudaEventElapsedTime")) {
      return false;
    }
    return std::isfinite(*elapsed_ms) && *elapsed_ms > 0.0F;
  }

  ~BenchmarkServer() {
    if (begin_ != nullptr) cudaEventDestroy(begin_);
    if (end_ != nullptr) cudaEventDestroy(end_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
    if (lhs_ != nullptr) cudaFree(lhs_);
    if (rhs_ != nullptr) cudaFree(rhs_);
    if (output_ != nullptr) cudaFree(output_);
  }

 private:
  void launch_once() {
    constexpr unsigned int blocks = (kElements + kThreads - 1U) / kThreads;
    vibeqc_cumetal_fp32_contract<<<blocks, kThreads, 0, stream_>>>(lhs_, rhs_, output_, kElements);
  }

  cudaDeviceProp properties_{};
  cudaStream_t stream_ = nullptr;
  cudaEvent_t begin_ = nullptr;
  cudaEvent_t end_ = nullptr;
  float* lhs_ = nullptr;
  float* rhs_ = nullptr;
  float* output_ = nullptr;
};

}  // namespace

int main() {
  BenchmarkServer server;
  if (!server.initialize()) return 1;

  std::string command;
  while (std::getline(std::cin, command)) {
    if (command == "run") {
      float elapsed_ms = 0.0F;
      if (!server.run_sample(&elapsed_ms)) return 2;
      std::cout << "OK " << elapsed_ms << std::endl;
      continue;
    }
    if (command == "quit") return 0;
    std::fprintf(stderr, "FAIL: unknown command: %s\n", command.c_str());
    return 3;
  }
  return 0;
}
