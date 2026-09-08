"""Standalone CUDA fixture driver for the exact generated DF primitive header."""


def emit_df_value_driver(architecture: str) -> str:
    """Check the allocated target, execute normalized primitives, and retain timings."""
    target = int(architecture.removeprefix("sm_"))
    return r"""#include "df_values.cuh"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace df = vibeqc::scf::generated_df;
struct Input {
  std::uint32_t centers_count;
  df::Angular angular[3];
  df::Vec3 centers[3];
  double exponents[3];
  double weight;
};
static_assert(sizeof(Input) == 144 && offsetof(Input, centers) == 40 &&
              offsetof(Input, exponents) == 112 && offsetof(Input, weight) == 136);

extern "C" __global__ void df_fixture_values(const Input* inputs, double* values, std::size_t count) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const Input& in = inputs[i];
  double value;
  if (in.centers_count == 2) {
    value = df::metric(in.exponents[0], in.centers[0], in.angular[0],
                       in.exponents[2], in.centers[2], in.angular[2]);
  } else if (in.centers_count == 3) {
    value = df::three_center(in.exponents[0], in.centers[0], in.angular[0],
        in.exponents[1], in.centers[1], in.angular[1],
        in.exponents[2], in.centers[2], in.angular[2]);
  } else { value = nan(""); }
  values[i] = in.weight * value;
}

void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
int main(int argc, char** argv) {
  try {
    if (argc != 4) throw std::runtime_error("input output threads required");
    const unsigned threads = static_cast<unsigned>(std::stoul(argv[3]));
    if (threads != 64 && threads != 128 && threads != 256) throw std::runtime_error("unsupported fixture schedule");
    int device, driver, runtime;
    check(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device));
    if (properties.major * 10 + properties.minor != EXPECTED_TARGET) throw std::runtime_error("compile/runtime target mismatch");
    check(cudaDriverGetVersion(&driver)); check(cudaRuntimeGetVersion(&runtime));
    std::ifstream stream(argv[1], std::ios::binary);
    char magic[8]; std::uint64_t count;
    stream.read(magic, 8); stream.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!stream || std::string(magic, 8) != "VQDF1421" || count > 10000000) throw std::runtime_error("invalid fixture header");
    std::vector<Input> inputs(count);
    stream.read(reinterpret_cast<char*>(inputs.data()), count * sizeof(Input));
    if (!stream || stream.peek() != std::char_traits<char>::eof()) throw std::runtime_error("invalid fixture dimensions");
    cudaFuncAttributes attributes{}; int blocks;
    check(cudaFuncGetAttributes(&attributes, df_fixture_values));
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, df_fixture_values, threads, 0));
    if (blocks < 1 || threads > static_cast<unsigned>(attributes.maxThreadsPerBlock)) throw std::runtime_error("illegal fixture launch resources");
    std::vector<double> output(count);
    std::vector<float> timings;
    if (count) {
      Input* gpu_inputs; double* gpu_output;
      check(cudaMalloc(&gpu_inputs, count * sizeof(Input)));
      check(cudaMalloc(&gpu_output, count * sizeof(double)));
      check(cudaMemcpy(gpu_inputs, inputs.data(), count * sizeof(Input), cudaMemcpyHostToDevice));
      cudaEvent_t start, stop; check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
      for (unsigned sample = 0; sample < 9; ++sample) {
        check(cudaEventRecord(start));
        df_fixture_values<<<static_cast<unsigned>((count + threads - 1) / threads), threads>>>(gpu_inputs, gpu_output, count);
        check(cudaGetLastError()); check(cudaEventRecord(stop)); check(cudaEventSynchronize(stop));
        float ms; check(cudaEventElapsedTime(&ms, start, stop));
        if (sample >= 2) timings.push_back(ms);
      }
      check(cudaMemcpy(output.data(), gpu_output, count * sizeof(double), cudaMemcpyDeviceToHost));
      check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
      check(cudaFree(gpu_inputs)); check(cudaFree(gpu_output));
    }
    std::ofstream result(argv[2], std::ios::binary);
    result.write(reinterpret_cast<const char*>(output.data()), count * sizeof(double));
    if (!result) throw std::runtime_error("cannot write complete fixture output");
    std::cout << "{\"device\":" << std::quoted(properties.name)
      << ",\"major\":" << properties.major << ",\"minor\":" << properties.minor
      << ",\"driver\":" << driver << ",\"runtime\":" << runtime
      << ",\"count\":" << count << ",\"threads\":" << threads
      << ",\"registers\":" << attributes.numRegs << ",\"local_bytes\":" << attributes.localSizeBytes
      << ",\"shared_bytes\":" << attributes.sharedSizeBytes << ",\"active_blocks_per_sm\":" << blocks
      << ",\"slurm_job_id\":" << std::quoted(std::getenv("SLURM_JOB_ID") ? std::getenv("SLURM_JOB_ID") : "")
      << ",\"milliseconds\":[";
    for (std::size_t i = 0; i < timings.size(); ++i) {
      if (i) std::cout << ','; std::cout << std::setprecision(9) << timings[i];
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
""".replace("EXPECTED_TARGET", str(target))
