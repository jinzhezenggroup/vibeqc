// Standalone calibration probe. Run under the local Slurm GPU allocation;
// allocation, initialization, validation and warmup are outside measured batches.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void checked(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

__global__ void launch_probe() {}

// Eight independent chains expose FP64 throughput rather than one dependency's
// latency. FMA=2 operations; seed/initialization/reduction add 16 per thread.
__global__ void fma_probe(double* output, int iterations) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const double seed = 0.25 + (i % 1024) * 0.0001;
  double a = seed, b = seed + 0.1, c = seed + 0.2, d = seed + 0.3;
  double e = seed + 0.4, f = seed + 0.5, g = seed + 0.6, h = seed + 0.7;
#pragma unroll 1
  for (int j = 0; j < iterations; ++j) {
    a = __fma_rn(a, 0.9999999, 0.0000001);
    b = __fma_rn(b, 0.9999999, 0.0000002);
    c = __fma_rn(c, 0.9999999, 0.0000003);
    d = __fma_rn(d, 0.9999999, 0.0000004);
    e = __fma_rn(e, 0.9999999, 0.0000005);
    f = __fma_rn(f, 0.9999999, 0.0000006);
    g = __fma_rn(g, 0.9999999, 0.0000007);
    h = __fma_rn(h, 0.9999999, 0.0000008);
  }
  output[i] = a + b + c + d + e + f + g + h;
}

__global__ void copy_probe(const double* __restrict__ input, double* __restrict__ output,
                           std::size_t count) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    output[i] = input[i];
}

// Streaming transform family: each element executes exactly iterations FP64
// FMAs. The refined suite fits this family separately from independent chains.
__global__ void mixed_probe(const double* __restrict__ input, double* __restrict__ output,
                            std::size_t count, int iterations) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    double x = input[i];
#pragma unroll 1
    for (int j = 0; j < iterations; ++j) x = __fma_rn(x, 0.9999999, 0.0000001);
    output[i] = x;
  }
}

struct Case {
  std::string id, split, family;
  int blocks, threads, iterations;
  std::size_t elements;
};

void launch(const Case& item, const double* input, double* output) {
  if (item.family == "launch")
    launch_probe<<<item.blocks, item.threads>>>();
  else if (item.family == "fma")
    fma_probe<<<item.blocks, item.threads>>>(output, item.iterations);
  else if (item.family == "copy")
    copy_probe<<<item.blocks, item.threads>>>(input, output, item.elements);
  else
    mixed_probe<<<item.blocks, item.threads>>>(input, output, item.elements, item.iterations);
  checked(cudaGetLastError());
}

cudaFuncAttributes attributes(const Case& item) {
  cudaFuncAttributes attr{};
  if (item.family == "launch")
    checked(cudaFuncGetAttributes(&attr, launch_probe));
  else if (item.family == "fma")
    checked(cudaFuncGetAttributes(&attr, fma_probe));
  else if (item.family == "copy")
    checked(cudaFuncGetAttributes(&attr, copy_probe));
  else
    checked(cudaFuncGetAttributes(&attr, mixed_probe));
  return attr;
}

struct Timing {
  double wall, event;
};
Timing measure(const Case& item, const double* input, double* output, int launches,
               cudaEvent_t start, cudaEvent_t stop) {
  checked(cudaDeviceSynchronize());
  const auto begin = std::chrono::steady_clock::now();
  checked(cudaEventRecord(start));
  for (int i = 0; i < launches; ++i) launch(item, input, output);
  checked(cudaEventRecord(stop));
  checked(cudaEventSynchronize(stop));
  const auto end = std::chrono::steady_clock::now();
  float ms = 0;
  checked(cudaEventElapsedTime(&ms, start, stop));
  return {std::chrono::duration<double>(end - begin).count(), ms * 0.001};
}

// Probe checks use an independent host std::fma oracle at deterministic spread
// indices, including the boundaries of the output range, outside timing.
double validate(const Case& item, const double* output) {
  if (item.family == "launch") return 0.0;
  double worst = 0.0;
  const auto count =
      item.family == "fma" ? static_cast<std::size_t>(item.blocks) * item.threads : item.elements;
  for (int sample = 0; sample < 17; ++sample) {
    const std::size_t i = (count - 1) * sample / 16;
    double expected = 0.25 + (i % 1024) * 0.0001;
    if (item.family == "fma") {
      double values[8];
      for (int k = 0; k < 8; ++k) values[k] = expected + 0.1 * k;
      for (int j = 0; j < item.iterations; ++j)
        for (int k = 0; k < 8; ++k) values[k] = std::fma(values[k], 0.9999999, 0.0000001 * (k + 1));
      expected = values[0];
      for (int k = 1; k < 8; ++k) expected += values[k];
    } else if (item.family == "mixed") {
      for (int j = 0; j < item.iterations; ++j) expected = std::fma(expected, 0.9999999, 0.0000001);
    }
    double actual;
    checked(cudaMemcpy(&actual, output + i, sizeof(double), cudaMemcpyDeviceToHost));
    if (!std::isfinite(actual)) throw std::runtime_error("nonfinite probe output");
    worst = std::max(worst, std::abs(actual - expected));
  }
  if (worst > 2e-12) throw std::runtime_error("independent FP64 probe oracle failed");
  return worst;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (!std::getenv("SLURM_JOB_ID")) throw std::runtime_error("run this probe through srun");
    const int samples = argc > 1 ? std::stoi(argv[1]) : 7;
    const bool smoke = argc > 2 && std::string(argv[2]) == "--smoke";
    const bool refined = argc > 2 && std::string(argv[2]) == "--refined";
    const bool training_only = argc > 3 && std::string(argv[3]) == "--training-only";
    if (samples < 3 || samples > 31) throw std::runtime_error("samples must be 3..31");
    checked(cudaSetDevice(0));  // Slurm's visible ordinal, never a physical index override.
    cudaDeviceProp prop{};
    checked(cudaGetDeviceProperties(&prop, 0));
    int driver = 0, runtime = 0;
    checked(cudaDriverGetVersion(&driver));
    checked(cudaRuntimeGetVersion(&runtime));
    const int sm = prop.multiProcessorCount;
    // At least four L2 capacities per array prevent a warm-L2 bandwidth result
    // from being labelled achieved streaming bandwidth. Bound allocation to 4 GiB.
    const std::size_t base = std::max<std::size_t>(1u << 24, prop.l2CacheSize / 2);
    const std::size_t maximum = smoke ? 8192 : 2 * base;
    if (maximum * 2 * sizeof(double) > (4ull << 30))
      throw std::runtime_error("probe memory budget exceeded");
    double *input = nullptr, *output = nullptr;
    checked(cudaMalloc(&input, maximum * sizeof(double)));
    checked(cudaMalloc(&output, maximum * sizeof(double)));
    std::vector<double> host(maximum);
    for (std::size_t i = 0; i < maximum; ++i) host[i] = 0.25 + (i % 1024) * 0.0001;
    checked(cudaMemcpy(input, host.data(), maximum * sizeof(double), cudaMemcpyHostToDevice));
    checked(cudaMemset(output, 0, maximum * sizeof(double)));
    host.clear();
    host.shrink_to_fit();

    std::vector<Case> cases;
    for (int blocks : {std::max(1, sm / 4), sm, sm * 2, sm * 4, sm * 8}) {
      for (int iterations : {512, 2048})
        cases.push_back({"fma-" + std::to_string(blocks) + "-" + std::to_string(iterations),
                         "train", "fma", blocks, 256, iterations, 0});
      for (std::size_t n : {base, maximum})
        cases.push_back({"copy-" + std::to_string(blocks) + "-" + std::to_string(n), "train",
                         "copy", blocks, 256, 0, n});
    }
    for (int blocks : {std::max(1, sm / 2), sm * 3, sm * 6}) {
      cases.push_back(
          {"held-fma-" + std::to_string(blocks), "holdout", "fma", blocks, 256, 1152, 0});
      cases.push_back({"held-copy-" + std::to_string(blocks), "holdout", "copy", blocks, 256, 0,
                       base + base / 2});
      for (int iterations : {1, 4, 16})
        cases.push_back({"held-mixed-" + std::to_string(blocks) + "-" + std::to_string(iterations),
                         "holdout", "mixed", blocks, 256, iterations, base + base / 4});
    }
    for (int launches : {256, 1024, 4096})
      cases.push_back(
          {"launch-" + std::to_string(launches), "train", "launch", 1, 128, launches, 0});
    for (int launches : {512, 2048})
      cases.push_back(
          {"held-launch-" + std::to_string(launches), "holdout", "launch", 1, 128, launches, 0});
    if (refined) {
      cases.clear();
      // Training grids include underfill and saturation. The held-out grids
      // introduce unseen fractional SM waves; no holdout was used to choose fits.
      for (int blocks : {std::max(1, sm / 4), std::max(1, sm / 2), sm, sm * 2, sm * 4, sm * 8}) {
        for (int iterations : {384, 1536})
          cases.push_back({"train-fma-" + std::to_string(blocks) + "-" + std::to_string(iterations),
                           "train", "fma", blocks, 256, iterations, 0});
        for (std::size_t n : {base, maximum})
          cases.push_back({"train-copy-" + std::to_string(blocks) + "-" + std::to_string(n),
                           "train", "copy", blocks, 256, 0, n});
        for (int iterations : {2, 8, 32, 64})
          cases.push_back(
              {"train-mixed-" + std::to_string(blocks) + "-" + std::to_string(iterations), "train",
               "mixed", blocks, 256, iterations, base});
      }
      for (int launches : {1, 4, 16, 64, 256, 1024, 4096})
        cases.push_back(
            {"train-launch-" + std::to_string(launches), "train", "launch", 1, 128, launches, 0});
      if (!training_only) {
        for (int blocks : {std::max(1, sm / 3), sm * 3 / 2, sm * 5}) {
          cases.push_back(
              {"fresh-fma-" + std::to_string(blocks), "holdout", "fma", blocks, 256, 896, 0});
          cases.push_back({"fresh-copy-" + std::to_string(blocks), "holdout", "copy", blocks, 256,
                           0, base + base / 3});
          for (int iterations : {3, 6, 12, 24, 48})
            cases.push_back(
                {"fresh-mixed-" + std::to_string(blocks) + "-" + std::to_string(iterations),
                 "holdout", "mixed", blocks, 256, iterations, base + base / 3});
        }
        for (int launches : {2, 8, 32, 128, 512, 2048})
          cases.push_back({"fresh-launch-" + std::to_string(launches), "holdout", "launch", 1, 128,
                           launches, 0});
      }
    }
    // Small, non-divisible extents exercise grid-stride tails under sanitizers.
    // They never enter a calibration fit or performance report.
    if (smoke)
      cases = {{"smoke-launch", "smoke", "launch", 1, 128, 3, 0},
               {"smoke-fma", "smoke", "fma", 3, 128, 17, 0},
               {"smoke-copy", "smoke", "copy", 3, 128, 0, 4099},
               {"smoke-mixed", "smoke", "mixed", 3, 128, 4, 8191}};
    std::mt19937 generator(1787);
    std::shuffle(cases.begin(), cases.end(), generator);
    cudaEvent_t start{}, stop{};
    checked(cudaEventCreate(&start));
    checked(cudaEventCreate(&stop));
    const Case warmup{"warmup", "warmup", "fma", smoke ? 2 : sm * 8, 256, smoke ? 32 : 2048, 0};
    const auto warm_begin = std::chrono::steady_clock::now();
    do {
      launch(warmup, input, output);
      checked(cudaDeviceSynchronize());
    } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - warm_begin).count() <
             (smoke ? 0.01 : 1.0));
    std::cout << std::setprecision(17);
    std::cout << "{\"schema\":\""
              << (refined ? "generativeqc.cuda-timing-probe.v2"
                          : "generativeqc.cuda-timing-probe.v1")
              << "\",\"device\":\"" << prop.name << "\",\"architecture\":\"sm_" << prop.major
              << prop.minor << "\",\"sm_count\":" << sm << ",\"l2_bytes\":" << prop.l2CacheSize
              << ",\"driver_version\":" << driver << ",\"runtime_version\":" << runtime
              << ",\"max_threads_per_sm\":" << prop.maxThreadsPerMultiProcessor
              << ",\"seed\":1787,\"samples\":" << samples << ",\"cases\":[\n";
    bool first_case = true;
    for (const auto& item : cases) {
      launch(item, input, output);
      checked(cudaDeviceSynchronize());
      const double error = validate(item, output);
      for (int i = 0; i < 3; ++i) launch(item, input, output);
      checked(cudaDeviceSynchronize());
      const Timing pilot = measure(item, input, output, 1, start, stop);
      const int launches =
          item.family == "launch"
              ? item.iterations
              : std::clamp(static_cast<int>(std::ceil(0.015 / pilot.wall)), 2, 256);
      const auto attr = attributes(item);
      const std::uint64_t threads = static_cast<std::uint64_t>(item.blocks) * item.threads;
      const std::uint64_t operations =
          item.family == "fma"     ? threads * (16ull * item.iterations + 16)
          : item.family == "mixed" ? 2ull * item.elements * item.iterations
                                   : 0;
      const std::uint64_t bytes = item.family == "fma"      ? 8 * threads
                                  : item.family == "launch" ? 0
                                                            : 16ull * item.elements;
      if (!first_case) std::cout << ",\n";
      first_case = false;
      std::cout << "{\"id\":\"" << item.id << "\",\"split\":\"" << item.split << "\",\"family\":\""
                << item.family << "\",\"grid_blocks\":" << item.blocks
                << ",\"block_threads\":" << item.threads << ",\"iterations\":" << item.iterations
                << ",\"elements\":" << item.elements << ",\"launch_count\":" << launches
                << ",\"operations_per_launch\":" << operations << ",\"bytes_per_launch\":" << bytes
                << ",\"registers_per_thread\":" << attr.numRegs
                << ",\"shared_bytes\":" << attr.sharedSizeBytes
                << ",\"local_bytes\":" << attr.localSizeBytes << ",\"max_absolute_error\":" << error
                << ",\"wall_seconds\":[";
      std::vector<double> event;
      for (int sample = 0; sample < samples; ++sample) {
        const auto timing = measure(item, input, output, launches, start, stop);
        if (sample) std::cout << ',';
        std::cout << timing.wall;
        event.push_back(timing.event);
      }
      std::cout << "],\"event_seconds\":[";
      for (int sample = 0; sample < samples; ++sample) {
        if (sample) std::cout << ',';
        std::cout << event[sample];
      }
      std::cout << "]}" << std::flush;
      std::cerr << item.id << " validated, " << launches << " launches/sample\n";
    }
    std::cout << "]}\n";
    checked(cudaEventDestroy(start));
    checked(cudaEventDestroy(stop));
    checked(cudaFree(output));
    checked(cudaFree(input));
  } catch (const std::exception& error) {
    std::cerr << "CUDA timing calibration: " << error.what() << '\n';
    return 1;
  }
}
