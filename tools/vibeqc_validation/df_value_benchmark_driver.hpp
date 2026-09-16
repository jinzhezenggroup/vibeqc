#pragma once
/** Test-only bounded raw-value contraction benchmark. Independent libcint
 * values are serialized with sampled real shell blocks, outside CUDA code. */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "df_value_benchmark_api.hpp"

namespace vibeqc::df_value_benchmark {
inline void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
inline int run(int argc, char** argv, const std::vector<Candidate>& candidates) {
  try {
    if (argc != 2 || !std::getenv("SLURM_JOB_ID"))
      throw std::runtime_error("requires Slurm and workload");
    std::ifstream file(argv[1], std::ios::binary);
    char magic[8];
    file.read(magic, 8);
    if (std::string(magic, 8) != "VQDV4051")
      throw std::runtime_error("invalid value workload schema");
    std::uint32_t count = 0;
    file.read(reinterpret_cast<char*>(&count), 4);
    if (!count || count > 2000) throw std::runtime_error("invalid group count");
    cudaDeviceProp properties{};
    int device = 0;
    check(cudaGetDevice(&device));
    check(cudaGetDeviceProperties(&properties, device));
    std::cout << "{\"kind\":\"device\",\"architecture\":\"sm_"
              << 10 * properties.major + properties.minor << "\"}" << std::endl;
    std::cout << std::setprecision(17);
    for (unsigned group = 0; group < count; ++group) {
      // profile, angular[3], primitive lengths[3], sampled shell blocks,
      // complete real shell frequency, Cartesian component tasks, replicas.
      std::array<std::uint64_t, 11> meta{};
      file.read(reinterpret_cast<char*>(meta.data()), sizeof(meta));
      const auto primitives = meta[4] * meta[5] * meta[6], tasks = meta[9], replicas = meta[10];
      if (!tasks || !replicas || !primitives || tasks * primitives > 10000000 ||
          tasks * replicas > 10000000)
        throw std::runtime_error("invalid bounded workload extents");
      std::vector<Input> input(tasks * primitives);
      std::vector<double> expected(tasks), actual(tasks * replicas);
      file.read(reinterpret_cast<char*>(input.data()), input.size() * sizeof(Input));
      file.read(reinterpret_cast<char*>(expected.data()), expected.size() * sizeof(double));
      if (!file) throw std::runtime_error("truncated workload");
      Input* in{};
      double* out{};
      check(cudaMalloc(&in, input.size() * sizeof(Input)));
      check(cudaMalloc(&out, actual.size() * sizeof(double)));
      check(cudaMemcpy(in, input.data(), input.size() * sizeof(Input), cudaMemcpyHostToDevice));
      cudaEvent_t start{}, stop{};
      check(cudaEventCreate(&start));
      check(cudaEventCreate(&stop));
      for (const auto& candidate : candidates) {
        if (candidate.a != meta[1] || candidate.b != meta[2] || candidate.c != meta[3]) continue;
        check(candidate.launch(in, out, tasks, primitives, replicas));
        check(
            cudaMemcpy(actual.data(), out, actual.size() * sizeof(double), cudaMemcpyDeviceToHost));
        double error = 0;
        bool passed = true;
        for (std::size_t i = 0; i < actual.size(); ++i) {
          passed = passed && std::isfinite(actual[i]);
          error = std::max(error, std::abs(actual[i] - expected[i % tasks]));
          passed = passed && std::abs(actual[i] - expected[i % tasks]) <=
                                 8e-11 * (1 + std::abs(expected[i % tasks]));
        }
        std::array<float, 5> elapsed{};
        for (auto& time : elapsed) {
          check(cudaEventRecord(start));
          check(candidate.launch(in, out, tasks, primitives, replicas));
          check(cudaEventRecord(stop));
          check(cudaEventSynchronize(stop));
          check(cudaEventElapsedTime(&time, start, stop));
        }
        std::cout << "{\"profile\":\"" << meta[0] << "\",\"group\":" << group << ",\"candidate\":\""
                  << candidate.key << "\",\"maximum_error\":" << error
                  << ",\"numerical_passed\":" << (passed ? "true" : "false")
                  << ",\"sampled_shell_blocks\":" << meta[7] << ",\"shell_frequency\":" << meta[8]
                  << ",\"sampled_components\":" << tasks << ",\"replicas\":" << replicas
                  << ",\"measured_primitive_components\":" << tasks * replicas * primitives
                  << ",\"milliseconds\":[";
        for (unsigned i = 0; i < 5; ++i) std::cout << (i ? "," : "") << elapsed[i];
        std::cout << "]}" << std::endl;
      }
      check(cudaEventDestroy(start));
      check(cudaEventDestroy(stop));
      check(cudaFree(in));
      check(cudaFree(out));
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
}  // namespace vibeqc::df_value_benchmark
