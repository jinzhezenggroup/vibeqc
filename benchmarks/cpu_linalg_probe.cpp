#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "tensor/cpu_linalg.hpp"

int main(int argc, char** argv) try {
  const std::size_t n =
      argc > 1 ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10)) : 256;
  const int repeats = argc > 2 ? std::max(1, std::atoi(argv[2])) : 5;
  const std::string_view provider_name = argc > 3 ? argv[3] : "auto";
  const int provider_threads = argc > 4 ? std::max(1, std::atoi(argv[4])) : 1;
  if (!n || n > std::numeric_limits<std::size_t>::max() / sizeof(double) / n)
    throw std::invalid_argument("invalid probe matrix extent");

  using vibeqc::tensor::CpuLinalgProvider;
  using vibeqc::tensor::CpuLinalgThreadOwnership;
  CpuLinalgProvider provider = CpuLinalgProvider::automatic;
  if (provider_name == "scalar")
    provider = CpuLinalgProvider::scalar;
  else if (provider_name == "openblas")
    provider = CpuLinalgProvider::openblas;
  else if (provider_name != "auto")
    return 3;
  const auto ownership =
      provider_threads > 1 || (provider == CpuLinalgProvider::openblas &&
                               !vibeqc::tensor::cpu_openblas_local_thread_control_built())
          ? CpuLinalgThreadOwnership::provider_parallel
          : CpuLinalgThreadOwnership::task_parallel;

  std::vector<double> a(n * n), b(n * n), c(n * n);
  std::mt19937_64 rng(674);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  for (double& value : a) value = distribution(rng);
  for (double& value : b) value = distribution(rng);

  const auto plan = vibeqc::tensor::CpuLinalgPlan{provider, ownership, provider_threads};
  const auto diagnostic = vibeqc::tensor::cpu_linalg_diagnostic(plan);
  vibeqc::tensor::cpu_gemm('N', 'N', n, n, n, a.data(), b.data(), c.data(), 1.0, 0.0, plan);

  const auto start = std::chrono::steady_clock::now();
  for (int repeat = 0; repeat < repeats; ++repeat)
    vibeqc::tensor::cpu_gemm('N', 'N', n, n, n, a.data(), b.data(), c.data(), 1.0, 0.0, plan);
  const auto stop = std::chrono::steady_clock::now();
  const double seconds = std::chrono::duration<double>(stop - start).count() / repeats;
  const double gflops = (2.0 * static_cast<double>(n) * n * n) / seconds / 1.0e9;

  std::cout << "{\"n\":" << n << ",\"repeats\":" << repeats << ",\"provider\":\""
            << vibeqc::tensor::cpu_linalg_provider_name(diagnostic.provider)
            << "\",\"provider_threads\":" << diagnostic.provider_threads
            << ",\"thread_ownership\":\""
            << (diagnostic.thread_ownership == CpuLinalgThreadOwnership::task_parallel
                    ? "task_parallel"
                    : "provider_parallel")
            << "\",\"seconds\":" << seconds << ",\"gflops\":" << gflops << "}\n";
} catch (const std::exception& error) {
  std::cerr << "CPU linalg probe: " << error.what() << '\n';
  return 1;
}
