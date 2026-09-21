#ifndef VIBEQC_TENSOR_CPU_LINALG_HPP
#define VIBEQC_TENSOR_CPU_LINALG_HPP

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace vibeqc::tensor {

enum class CpuLinalgProvider : std::uint8_t { automatic, scalar, openblas };
enum class CpuLinalgThreadOwnership : std::uint8_t { task_parallel, provider_parallel };

struct CpuLinalgPlan {
  CpuLinalgProvider provider{CpuLinalgProvider::automatic};
  CpuLinalgThreadOwnership thread_ownership{CpuLinalgThreadOwnership::task_parallel};
  int provider_threads{1};
};

struct CpuLinalgDiagnostic {
  CpuLinalgProvider provider{CpuLinalgProvider::scalar};
  CpuLinalgThreadOwnership thread_ownership{CpuLinalgThreadOwnership::task_parallel};
  int provider_threads{1};
  bool external_provider{};
  bool lapack_available{};
  bool local_thread_control{};
  bool global_thread_control{};
  // VibeQC host-code build target. External BLAS may dispatch internally.
  std::string_view cpu_target{};
};

struct CpuSymmetricEigenResult {
  std::vector<double> values;
  std::vector<double> vectors;  // row-major matrix, eigenvectors in columns
};

[[nodiscard]] bool cpu_openblas_built() noexcept;
[[nodiscard]] bool cpu_openblas_lapack_built() noexcept;
[[nodiscard]] bool cpu_openblas_local_thread_control_built() noexcept;
[[nodiscard]] bool cpu_openblas_global_thread_control_built() noexcept;
[[nodiscard]] CpuLinalgProvider resolve_cpu_linalg_provider(const CpuLinalgPlan& plan,
                                                            bool require_lapack = false);
[[nodiscard]] CpuLinalgDiagnostic cpu_linalg_diagnostic(const CpuLinalgPlan& plan = {});
[[nodiscard]] std::string_view cpu_linalg_provider_name(CpuLinalgProvider provider) noexcept;
[[nodiscard]] std::string_view cpu_linalg_target_name() noexcept;

void cpu_gemm(char a_trans, char b_trans, std::size_t m, std::size_t n, std::size_t k,
              const double* a, const double* b, double* c, double alpha = 1.0, double beta = 0.0,
              const CpuLinalgPlan& plan = {});

/** In-place lower Cholesky factorization. Returns LAPACK-style info:
 * 0 on success, j>0 when the leading minor of order j is not positive definite.
 */
int cpu_cholesky_lower(double* matrix, std::size_t n, const CpuLinalgPlan& plan = {});
[[nodiscard]] CpuSymmetricEigenResult cpu_symmetric_eigen(std::vector<double> matrix, std::size_t n,
                                                          const CpuLinalgPlan& plan = {});

}  // namespace vibeqc::tensor

#endif
