#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr unsigned int kThreads = 256;
constexpr unsigned int kComputeElements = 1U << 20;
constexpr unsigned int kMemoryElements = 1U << 22;
constexpr unsigned int kGatherElements = 1U << 20;
constexpr unsigned int kMixedElements = 1U << 20;
constexpr unsigned int kMaxElements = kMemoryElements;
constexpr unsigned int kComputeIterations = 64;
constexpr unsigned int kMixedIterations = 16;
constexpr unsigned int kWarmupLaunches = 4;

enum class Workload { Compute, Memory, Gather, Mixed };

extern "C" __global__ void vibeqc_cumetal_fp32_compute(const float* lhs, const float* rhs,
                                                       float* output, unsigned int size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) return;

  float x = lhs[index];
  float y = rhs[index];
  float accumulator = 0.125F;
#pragma unroll 4
  for (unsigned int iteration = 0; iteration < kComputeIterations; ++iteration) {
    accumulator += x * y;
    x = x * 0.99991F + 0.00013F;
    y = y * 1.00007F - 0.00011F;
  }
  output[index] = accumulator;
}

extern "C" __global__ void vibeqc_cumetal_fp32_memory(const float* lhs, const float* rhs,
                                                      float* output, unsigned int size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) return;
  output[index] = lhs[index] * 1.0001F + rhs[index] * 0.9999F;
}

extern "C" __global__ void vibeqc_cumetal_fp32_gather(const float* lhs, const float* rhs,
                                                      float* output, unsigned int size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) return;
  const unsigned int source = (index * 40503U) & (size - 1U);
  output[index] = lhs[source] * 0.625F + rhs[index] * 0.375F;
}

extern "C" __global__ void vibeqc_cumetal_fp32_mixed(const float* lhs, const float* rhs,
                                                     float* output, unsigned int size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) return;

  float x = lhs[index];
  float y = rhs[index];
  float z = lhs[(index + 1U) & (size - 1U)];
#pragma unroll 4
  for (unsigned int iteration = 0; iteration < kMixedIterations; ++iteration) {
    x = x * 0.9997F + y * 0.0003F;
    y = y * 0.9991F + z * 0.0009F;
    z = z * 0.9989F + x * 0.0011F;
  }
  output[index] = x + y + z;
}

bool check_cuda(cudaError_t status, const char* expression) {
  if (status == cudaSuccess) return true;
  std::fprintf(stderr, "FAIL: %s: %s\n", expression, cudaGetErrorString(status));
  return false;
}

const char* workload_name(Workload workload) {
  switch (workload) {
    case Workload::Compute:
      return "compute";
    case Workload::Memory:
      return "memory";
    case Workload::Gather:
      return "gather";
    case Workload::Mixed:
      return "mixed";
  }
  return "unknown";
}

bool parse_workload(const std::string& name, Workload* workload) {
  if (name == "compute") {
    *workload = Workload::Compute;
    return true;
  }
  if (name == "memory") {
    *workload = Workload::Memory;
    return true;
  }
  if (name == "gather") {
    *workload = Workload::Gather;
    return true;
  }
  if (name == "mixed") {
    *workload = Workload::Mixed;
    return true;
  }
  return false;
}

unsigned int workload_elements(Workload workload) {
  switch (workload) {
    case Workload::Compute:
      return kComputeElements;
    case Workload::Memory:
      return kMemoryElements;
    case Workload::Gather:
      return kGatherElements;
    case Workload::Mixed:
      return kMixedElements;
  }
  return 0;
}

unsigned int workload_launches(Workload workload) {
  switch (workload) {
    case Workload::Compute:
      return 128;
    case Workload::Memory:
      return 32;
    case Workload::Gather:
      return 64;
    case Workload::Mixed:
      return 96;
  }
  return 0;
}

// Independent host references are evaluated only during initialization.
float reference_value(Workload workload, unsigned int index, unsigned int size,
                      const std::vector<float>& lhs, const std::vector<float>& rhs) {
  float x = lhs[index];
  float y = rhs[index];
  switch (workload) {
    case Workload::Compute: {
      float accumulator = 0.125F;
      for (unsigned int iteration = 0; iteration < kComputeIterations; ++iteration) {
        accumulator += x * y;
        x = x * 0.99991F + 0.00013F;
        y = y * 1.00007F - 0.00011F;
      }
      return accumulator;
    }
    case Workload::Memory:
      return x * 1.0001F + y * 0.9999F;
    case Workload::Gather: {
      const unsigned int source = (index * 40503U) & (size - 1U);
      return lhs[source] * 0.625F + y * 0.375F;
    }
    case Workload::Mixed: {
      float z = lhs[(index + 1U) & (size - 1U)];
      for (unsigned int iteration = 0; iteration < kMixedIterations; ++iteration) {
        x = x * 0.9997F + y * 0.0003F;
        y = y * 0.9991F + z * 0.0009F;
        z = z * 0.9989F + x * 0.0011F;
      }
      return x + y + z;
    }
  }
  return std::nanf("");
}

bool validate_value(float actual, float expected) {
  // Preserve the original compute-series tolerance; do not admit arbitrary finite output.
  const float tolerance = 2.0e-3F * std::fmax(1.0F, std::fabs(expected));
  return std::isfinite(actual) && std::isfinite(expected) &&
         std::fabs(actual - expected) <= tolerance;
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

    std::vector<float> lhs(kMaxElements);
    std::vector<float> rhs(kMaxElements);
    for (unsigned int index = 0; index < kMaxElements; ++index) {
      lhs[index] = 0.5F + static_cast<float>(index % 97U) * 0.001F;
      rhs[index] = 1.25F - static_cast<float>(index % 89U) * 0.0005F;
    }

    const std::size_t bytes = static_cast<std::size_t>(kMaxElements) * sizeof(float);
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

    for (Workload workload :
         {Workload::Compute, Workload::Memory, Workload::Gather, Workload::Mixed}) {
      for (unsigned int warmup = 0; warmup < kWarmupLaunches; ++warmup) {
        launch_once(workload);
      }
      if (!check_cuda(cudaGetLastError(), "warmup kernel launch") ||
          !check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize(warmup)")) {
        return false;
      }

      // Include nonzero, block-boundary and wraparound indices: index zero alone
      // cannot distinguish the gather permutation from a contiguous read.
      const unsigned int size = workload_elements(workload);
      for (unsigned int index : {0U, 1U, 96U, 97U, 255U, 256U, size / 2U, size - 2U, size - 1U}) {
        float actual = 0.0F;
        if (!check_cuda(cudaMemcpy(&actual, output_ + index, sizeof(float), cudaMemcpyDeviceToHost),
                        "cudaMemcpy(validation)")) {
          return false;
        }
        const float expected = reference_value(workload, index, size, lhs, rhs);
        if (!validate_value(actual, expected)) {
          std::fprintf(stderr, "FAIL: FP32 %s validation index=%u actual=%.8g expected=%.8g\n",
                       workload_name(workload), index, static_cast<double>(actual),
                       static_cast<double>(expected));
          return false;
        }
      }
    }

    std::cout << "READY device=" << properties_.name << " cases=compute,memory,gather,mixed"
              << std::endl;
    return true;
  }

  bool run_sample(Workload workload, float* elapsed_ms) {
    if (!check_cuda(cudaEventRecord(begin_, stream_), "cudaEventRecord(begin)")) return false;
    const unsigned int launches = workload_launches(workload);
    for (unsigned int launch = 0; launch < launches; ++launch) launch_once(workload);
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
  void launch_once(Workload workload) {
    const unsigned int size = workload_elements(workload);
    const unsigned int blocks = (size + kThreads - 1U) / kThreads;
    switch (workload) {
      case Workload::Compute:
        vibeqc_cumetal_fp32_compute<<<blocks, kThreads, 0, stream_>>>(lhs_, rhs_, output_, size);
        return;
      case Workload::Memory:
        vibeqc_cumetal_fp32_memory<<<blocks, kThreads, 0, stream_>>>(lhs_, rhs_, output_, size);
        return;
      case Workload::Gather:
        vibeqc_cumetal_fp32_gather<<<blocks, kThreads, 0, stream_>>>(lhs_, rhs_, output_, size);
        return;
      case Workload::Mixed:
        vibeqc_cumetal_fp32_mixed<<<blocks, kThreads, 0, stream_>>>(lhs_, rhs_, output_, size);
        return;
    }
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
    if (command.rfind("run ", 0) == 0) {
      Workload workload{};
      const std::string name = command.substr(4);
      if (!parse_workload(name, &workload)) {
        std::fprintf(stderr, "FAIL: unknown workload: %s\n", name.c_str());
        return 3;
      }
      float elapsed_ms = 0.0F;
      if (!server.run_sample(workload, &elapsed_ms)) return 2;
      std::cout << "OK " << name << " " << elapsed_ms << std::endl;
      continue;
    }
    if (command == "quit") return 0;
    std::fprintf(stderr, "FAIL: unknown command: %s\n", command.c_str());
    return 3;
  }
  return 0;
}
