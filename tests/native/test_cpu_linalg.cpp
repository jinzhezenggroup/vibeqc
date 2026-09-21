#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "tensor/cpu_linalg.hpp"

namespace {
using vibeqc::tensor::CpuLinalgPlan;
using vibeqc::tensor::CpuLinalgProvider;
using vibeqc::tensor::CpuLinalgThreadOwnership;

bool close(double a, double b, double tolerance = 1.0e-12) {
  return std::abs(a - b) <= tolerance * std::max({1.0, std::abs(a), std::abs(b)});
}

bool check_zero_scaling(CpuLinalgProvider provider, CpuLinalgThreadOwnership ownership,
                        int threads) {
  const CpuLinalgPlan plan{provider, ownership, threads};
  const double poison = std::numeric_limits<double>::quiet_NaN();
  const std::array<double, 4> a{1, 2, 3, 4}, b{5, 6, 7, 8};
  for (char ta : {'N', 'T'})
    for (char tb : {'N', 'T'}) {
      std::array<double, 4> c{poison, poison, poison, poison};
      vibeqc::tensor::cpu_gemm(ta, tb, 2, 2, 2, a.data(), b.data(), c.data(), 1, 0, plan);
      for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j) {
          double expected = 0;
          for (std::size_t p = 0; p < 2; ++p)
            expected += a[ta == 'T' ? p * 2 + i : i * 2 + p] * b[tb == 'T' ? j * 2 + p : p * 2 + j];
          if (!std::isfinite(c[i * 2 + j]) || c[i * 2 + j] != expected) return false;
        }
    }
  std::array<double, 4> c{poison, poison, poison, poison}, nan{poison, poison, poison, poison};
  vibeqc::tensor::cpu_gemm('N', 'N', 2, 2, 0, nullptr, nullptr, c.data(), 1, 0, plan);
  if (!std::all_of(c.begin(), c.end(), [](double x) { return x == 0; })) return false;
  c.fill(2);
  vibeqc::tensor::cpu_gemm('N', 'N', 2, 2, 2, nan.data(), nan.data(), c.data(), 0, 3, plan);
  if (!std::all_of(c.begin(), c.end(), [](double x) { return x == 6; })) return false;
  c.fill(poison);
  vibeqc::tensor::cpu_gemm('N', 'N', 2, 2, 2, nan.data(), nan.data(), c.data(), 0, 0, plan);
  return std::all_of(c.begin(), c.end(), [](double x) { return x == 0; });
}

bool check_gemm(CpuLinalgProvider provider,
                CpuLinalgThreadOwnership ownership = CpuLinalgThreadOwnership::task_parallel,
                int threads = 1) {
  const std::array<double, 6> a{1, 2, 3, 4, 5, 6};
  const std::array<double, 6> b{7, 8, 9, 10, 11, 12};
  std::array<double, 4> c{1, 1, 1, 1};
  CpuLinalgPlan plan{provider, ownership, threads};
  vibeqc::tensor::cpu_gemm('N', 'N', 2, 2, 3, a.data(), b.data(), c.data(), 1.0, 2.0, plan);
  const std::array<double, 4> expected{60, 66, 141, 156};
  for (std::size_t i = 0; i < c.size(); ++i)
    if (!close(c[i], expected[i])) return false;

  std::array<double, 9> gram{};
  vibeqc::tensor::cpu_gemm('T', 'N', 3, 3, 2, a.data(), a.data(), gram.data(), 1.0, 0.0, plan);
  const std::array<double, 9> gram_expected{17, 22, 27, 22, 29, 36, 27, 36, 45};
  for (std::size_t i = 0; i < gram.size(); ++i)
    if (!close(gram[i], gram_expected[i])) return false;

  std::array<double, 1> zero_inner{2.0};
  vibeqc::tensor::cpu_gemm('N', 'N', 1, 1, 0, nullptr, nullptr, zero_inner.data(), 1.0, 3.0, plan);
  return close(zero_inner[0], 6.0) && check_zero_scaling(provider, ownership, threads);
}

bool check_cholesky(CpuLinalgProvider provider,
                    CpuLinalgThreadOwnership ownership = CpuLinalgThreadOwnership::task_parallel,
                    int threads = 1) {
  std::array<double, 9> a{4, 12, -16, 12, 37, -43, -16, -43, 98};
  const CpuLinalgPlan plan{provider, ownership, threads};
  if (vibeqc::tensor::cpu_cholesky_lower(a.data(), 3, plan) != 0) return false;
  const std::array<double, 6> expected{2, 6, 1, -8, 5, 3};
  const std::array<std::size_t, 6> where{0, 3, 4, 6, 7, 8};
  for (std::size_t i = 0; i < where.size(); ++i)
    if (!close(a[where[i]], expected[i])) return false;

  std::array<double, 4> bad{1, 2, 2, 1};
  return vibeqc::tensor::cpu_cholesky_lower(bad.data(), 2, plan) == 2;
}
bool check_eigen(CpuLinalgProvider provider,
                 CpuLinalgThreadOwnership ownership = CpuLinalgThreadOwnership::task_parallel,
                 int threads = 1) {
  const CpuLinalgPlan plan{provider, ownership, threads};
  auto result = vibeqc::tensor::cpu_symmetric_eigen({2.0, 1.0, 1.0, 2.0}, 2, plan);
  if (result.values.size() != 2 || result.vectors.size() != 4 || !close(result.values[0], 1.0) ||
      !close(result.values[1], 3.0))
    return false;
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      double reconstructed = 0.0;
      for (std::size_t k = 0; k < 2; ++k)
        reconstructed += result.vectors[i * 2 + k] * result.values[k] * result.vectors[j * 2 + k];
      if (!close(reconstructed, i == j ? 2.0 : 1.0, 2.0e-12)) return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  if (!check_gemm(CpuLinalgProvider::scalar) || !check_cholesky(CpuLinalgProvider::scalar) ||
      !check_eigen(CpuLinalgProvider::scalar)) {
    std::cerr << "scalar CPU linear algebra failed\n";
    return 1;
  }

  const auto automatic = vibeqc::tensor::cpu_linalg_diagnostic();
  if (automatic.thread_ownership != CpuLinalgThreadOwnership::task_parallel ||
      automatic.provider_threads != 1) {
    std::cerr << "default thread ownership is not task-parallel/single-thread\n";
    return 2;
  }
  if (automatic.cpu_target.empty() ||
      automatic.cpu_target != vibeqc::tensor::cpu_linalg_target_name()) {
    std::cerr << "CPU target identity is missing or inconsistent\n";
    return 7;
  }

  if (vibeqc::tensor::cpu_openblas_built()) {
    const bool local = vibeqc::tensor::cpu_openblas_local_thread_control_built();
    const bool global = vibeqc::tensor::cpu_openblas_global_thread_control_built();
    if (!local && automatic.provider != CpuLinalgProvider::scalar) {
      std::cerr << "OpenBLAS without local thread control must not auto-promote\n";
      return 3;
    }

    const auto ownership = local ? CpuLinalgThreadOwnership::task_parallel
                                 : CpuLinalgThreadOwnership::provider_parallel;
    if ((local || global) && !check_gemm(CpuLinalgProvider::openblas, ownership)) {
      std::cerr << "OpenBLAS GEMM failed\n";
      return 4;
    }
    if ((local || global) && vibeqc::tensor::cpu_openblas_lapack_built() &&
        (!check_cholesky(CpuLinalgProvider::openblas, ownership) ||
         !check_eigen(CpuLinalgProvider::openblas, ownership))) {
      std::cerr << "OpenBLAS LAPACK provider failed\n";
      return 5;
    }
  }

  try {
    const CpuLinalgPlan invalid{CpuLinalgProvider::automatic,
                                CpuLinalgThreadOwnership::task_parallel, 2};
    std::array<double, 1> one{1.0};
    vibeqc::tensor::cpu_gemm('N', 'N', 1, 1, 1, one.data(), one.data(), one.data(), 1, 0, invalid);
    std::cerr << "nested-provider thread contract was accepted\n";
    return 6;
  } catch (const std::invalid_argument&) {
  }

  std::cout << "CPU linear algebra provider tests passed ("
            << vibeqc::tensor::cpu_linalg_provider_name(automatic.provider) << ")\n";
}
