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

/** Matrix-vector product with row-major A.
 * `trans == 'N'` consumes an m-by-n A and n-vector x, producing an m-vector y.
 * `trans == 'T'` consumes the same storage and an m-vector x, producing an n-vector y.
 */
void cpu_gemv(char trans, std::size_t m, std::size_t n, const double* a, const double* x, double* y,
              double alpha = 1.0, double beta = 0.0, const CpuLinalgPlan& plan = {});

/** Rank-1 update of a row-major m-by-n matrix: A := alpha * x * y^T + A. */
void cpu_ger(std::size_t m, std::size_t n, const double* x, const double* y, double* a,
             double alpha = 1.0, const CpuLinalgPlan& plan = {});

/** Symmetric matrix-matrix product with row-major storage.
 * `side == 'L'` computes C := alpha * A * B + beta * C with m-by-m symmetric A.
 * `side == 'R'` computes C := alpha * B * A + beta * C with n-by-n symmetric A.
 * Only the triangle selected by `uplo` is read from A.
 */
void cpu_symm(char side, char uplo, std::size_t m, std::size_t n, const double* a, const double* b,
              double* c, double alpha = 1.0, double beta = 0.0, const CpuLinalgPlan& plan = {});

/** Symmetric rank-k update of the selected triangle.
 * `trans == 'N'` consumes an n-by-k row-major A; `trans == 'T'` consumes k-by-n.
 * Only the triangle selected by `uplo` is read from or written to in C.
 */
void cpu_syrk(char uplo, char trans, std::size_t n, std::size_t k, const double* a, double* c,
              double alpha = 1.0, double beta = 0.0, const CpuLinalgPlan& plan = {});

/** Solve a triangular matrix equation in place using row-major storage.
 * `side == 'L'` computes B := alpha * op(A)^-1 * B and requires an m-by-m A.
 * `side == 'R'` computes B := alpha * B * op(A)^-1 and requires an n-by-n A.
 * `diag == 'U'` treats the diagonal as unit and does not read it.
 */
void cpu_trsm(char side, char uplo, char trans, char diag, std::size_t m, std::size_t n,
              const double* a, double* b, double alpha = 1.0, const CpuLinalgPlan& plan = {});

/** Multiply a general matrix by a triangular matrix in place using row-major storage.
 * `side == 'L'` computes B := alpha * op(A) * B and requires an m-by-m A.
 * `side == 'R'` computes B := alpha * B * op(A) and requires an n-by-n A.
 * `diag == 'U'` treats the diagonal as unit and does not read it.
 */
void cpu_trmm(char side, char uplo, char trans, char diag, std::size_t m, std::size_t n,
              const double* a, double* b, double alpha = 1.0, const CpuLinalgPlan& plan = {});

/** In-place lower Cholesky factorization. Returns LAPACK-style info:
 * 0 on success, j>0 when the leading minor of order j is not positive definite.
 */
int cpu_cholesky_lower(double* matrix, std::size_t n, const CpuLinalgPlan& plan = {});
[[nodiscard]] CpuSymmetricEigenResult cpu_symmetric_eigen(std::vector<double> matrix, std::size_t n,
                                                          const CpuLinalgPlan& plan = {});

}  // namespace vibeqc::tensor

#endif
