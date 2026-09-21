#include "tensor/cpu_linalg.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>

#ifndef VIBEQC_HAS_OPENBLAS
#define VIBEQC_HAS_OPENBLAS 0
#endif
#ifndef VIBEQC_OPENBLAS_SCIPY_PREFIX
#define VIBEQC_OPENBLAS_SCIPY_PREFIX 0
#endif
#ifndef VIBEQC_OPENBLAS_HAS_LAPACKE
#define VIBEQC_OPENBLAS_HAS_LAPACKE 0
#endif
#ifndef VIBEQC_OPENBLAS_HAS_LOCAL_THREADS
#define VIBEQC_OPENBLAS_HAS_LOCAL_THREADS 0
#endif
#ifndef VIBEQC_OPENBLAS_HAS_GLOBAL_THREADS
#define VIBEQC_OPENBLAS_HAS_GLOBAL_THREADS 0
#endif

#if VIBEQC_HAS_OPENBLAS
#include <cblas.h>
#if VIBEQC_OPENBLAS_HAS_LAPACKE
#include <lapacke.h>
#endif
#endif

namespace vibeqc::tensor {
namespace {

bool transpose(char value) {
  if (value == 'N' || value == 'n') return false;
  if (value == 'T' || value == 't') return true;
  throw std::invalid_argument("CPU GEMM transpose must be N or T");
}

bool syrk_transpose(char value) {
  if (value == 'N' || value == 'n') return false;
  if (value == 'T' || value == 't') return true;
  throw std::invalid_argument("CPU SYRK transpose must be N or T");
}

bool upper_triangle(char value) {
  if (value == 'U' || value == 'u') return true;
  if (value == 'L' || value == 'l') return false;
  throw std::invalid_argument("CPU SYRK triangle must be U or L");
}

bool trsm_left_side(char value) {
  if (value == 'L' || value == 'l') return true;
  if (value == 'R' || value == 'r') return false;
  throw std::invalid_argument("CPU TRSM side must be L or R");
}

bool trsm_upper_triangle(char value) {
  if (value == 'U' || value == 'u') return true;
  if (value == 'L' || value == 'l') return false;
  throw std::invalid_argument("CPU TRSM triangle must be U or L");
}

bool trsm_transpose(char value) {
  if (value == 'N' || value == 'n') return false;
  if (value == 'T' || value == 't') return true;
  throw std::invalid_argument("CPU TRSM transpose must be N or T");
}

bool trsm_unit_diagonal(char value) {
  if (value == 'U' || value == 'u') return true;
  if (value == 'N' || value == 'n') return false;
  throw std::invalid_argument("CPU TRSM diagonal must be U or N");
}

void validate_plan(const CpuLinalgPlan& plan) {
  if (plan.provider_threads < 1)
    throw std::invalid_argument("CPU linear algebra thread count must be positive");
  if (plan.thread_ownership == CpuLinalgThreadOwnership::task_parallel &&
      plan.provider_threads != 1)
    throw std::invalid_argument("task-parallel CPU linear algebra requires single-thread provider");
}

std::size_t checked_matrix_elements(std::size_t rows, std::size_t columns) {
  if (columns && rows > std::numeric_limits<std::size_t>::max() / sizeof(double) / columns)
    throw std::length_error("CPU matrix storage extent overflows");
  return rows * columns;
}

void scalar_gemm(bool ta, bool tb, std::size_t m, std::size_t n, std::size_t k, const double* a,
                 const double* b, double* c, double alpha, double beta) {
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double sum = 0.0;
      for (std::size_t p = 0; p < k; ++p) {
        const double av = ta ? a[p * m + i] : a[i * k + p];
        const double bv = tb ? b[j * k + p] : b[p * n + j];
        sum += av * bv;
      }
      c[i * n + j] = beta == 0.0 ? alpha * sum : alpha * sum + beta * c[i * n + j];
    }
  }
}

void scalar_syrk(bool upper, bool trans, std::size_t n, std::size_t k, const double* a, double* c,
                 double alpha, double beta) {
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t first_column = upper ? i : 0;
    const std::size_t last_column = upper ? n : i + 1;
    for (std::size_t j = first_column; j < last_column; ++j) {
      double sum = 0.0;
      for (std::size_t p = 0; p < k; ++p) {
        const double ai = trans ? a[p * n + i] : a[i * k + p];
        const double aj = trans ? a[p * n + j] : a[j * k + p];
        sum += ai * aj;
      }
      const std::size_t index = i * n + j;
      c[index] = beta == 0.0 ? alpha * sum : alpha * sum + beta * c[index];
    }
  }
}

void scalar_trsm(bool left, bool upper, bool trans, bool unit_diagonal, std::size_t m,
                 std::size_t n, const double* a, double* b, double alpha) {
  const std::size_t order = left ? m : n;
  const bool effective_upper = trans ? !upper : upper;
  const auto a_value = [&](std::size_t row, std::size_t column) {
    return trans ? a[column * order + row] : a[row * order + column];
  };

  if (alpha != 1.0)
    for (std::size_t index = 0; index < m * n; ++index) b[index] *= alpha;

  if (left) {
    for (std::size_t column = 0; column < n; ++column) {
      for (std::size_t step = 0; step < m; ++step) {
        const std::size_t row = effective_upper ? m - 1 - step : step;
        double value = b[row * n + column];
        if (effective_upper) {
          for (std::size_t p = row + 1; p < m; ++p) value -= a_value(row, p) * b[p * n + column];
        } else {
          for (std::size_t p = 0; p < row; ++p) value -= a_value(row, p) * b[p * n + column];
        }
        if (!unit_diagonal) value /= a_value(row, row);
        b[row * n + column] = value;
      }
    }
    return;
  }

  for (std::size_t row = 0; row < m; ++row) {
    for (std::size_t step = 0; step < n; ++step) {
      const std::size_t column = effective_upper ? step : n - 1 - step;
      double value = b[row * n + column];
      if (effective_upper) {
        for (std::size_t p = 0; p < column; ++p) value -= b[row * n + p] * a_value(p, column);
      } else {
        for (std::size_t p = column + 1; p < n; ++p) value -= b[row * n + p] * a_value(p, column);
      }
      if (!unit_diagonal) value /= a_value(column, column);
      b[row * n + column] = value;
    }
  }
}

int scalar_cholesky_lower(double* matrix, std::size_t n) {
  for (std::size_t j = 0; j < n; ++j) {
    double diagonal = matrix[j * n + j];
    for (std::size_t p = 0; p < j; ++p) {
      const double value = matrix[j * n + p];
      diagonal -= value * value;
    }
    if (!(diagonal > 0.0) || !std::isfinite(diagonal)) return static_cast<int>(j + 1);
    const double root = std::sqrt(diagonal);
    matrix[j * n + j] = root;
    for (std::size_t i = j + 1; i < n; ++i) {
      double value = matrix[i * n + j];
      for (std::size_t p = 0; p < j; ++p) value -= matrix[i * n + p] * matrix[j * n + p];
      matrix[i * n + j] = value / root;
    }
  }
  return 0;
}

CpuSymmetricEigenResult scalar_symmetric_eigen(std::vector<double> matrix, std::size_t n) {
  // Jacobi angle differences and doubled off-diagonals can overflow even
  // when every input and eigenvalue is representable. Normalize only extreme
  // scales; preserve established ordinary-range arithmetic and eigenvectors.
  double scale = 0.0, output_scale = 1.0;
  for (double value : matrix) scale = std::max(scale, std::abs(value));
  if (scale > std::sqrt(std::numeric_limits<double>::max()) ||
      (scale > 0.0 && scale < std::sqrt(std::numeric_limits<double>::min()))) {
    output_scale = scale;
    for (double& value : matrix) value /= scale;
  }
  std::vector<double> vectors(matrix.size(), 0.0);
  for (std::size_t item = 0; item < n; ++item) vectors[item * n + item] = 1.0;
  constexpr std::size_t maximum_sweeps = 100;
  bool converged = n == 1;
  for (std::size_t sweep = 0; sweep < maximum_sweeps && !converged; ++sweep) {
    double matrix_scale = 0.0;
    for (double value : matrix) matrix_scale = std::max(matrix_scale, std::abs(value));
    if (matrix_scale == 0.0) {
      converged = true;
      break;
    }
    const double tolerance = 1.0e-14 * matrix_scale;
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t q = p + 1; q < n; ++q) {
        const double apq = matrix[p * n + q];
        if (std::abs(apq) <= tolerance) continue;
        const double app = matrix[p * n + p], aqq = matrix[q * n + q];
        const double angle = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double cosine = std::cos(angle), sine = std::sin(angle);
        for (std::size_t k = 0; k < n; ++k) {
          if (k == p || k == q) continue;
          const double mkp = matrix[k * n + p], mkq = matrix[k * n + q];
          matrix[k * n + p] = matrix[p * n + k] = cosine * mkp - sine * mkq;
          matrix[k * n + q] = matrix[q * n + k] = sine * mkp + cosine * mkq;
        }
        matrix[p * n + p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
        matrix[q * n + q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
        matrix[p * n + q] = matrix[q * n + p] = 0.0;
        for (std::size_t row = 0; row < n; ++row) {
          const double vkp = vectors[row * n + p], vkq = vectors[row * n + q];
          vectors[row * n + p] = cosine * vkp - sine * vkq;
          vectors[row * n + q] = sine * vkp + cosine * vkq;
        }
      }
    }
    double largest_off_diagonal = 0.0;
    for (std::size_t row = 0; row < n; ++row)
      for (std::size_t column = row + 1; column < n; ++column)
        largest_off_diagonal = std::max(largest_off_diagonal, std::abs(matrix[row * n + column]));
    converged = largest_off_diagonal <= tolerance;
  }
  if (!converged) throw std::runtime_error("CPU symmetric eigensolver did not converge");
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](std::size_t a, std::size_t b) { return matrix[a * n + a] < matrix[b * n + b]; });
  CpuSymmetricEigenResult result;
  result.values.resize(n);
  result.vectors.resize(matrix.size());
  for (std::size_t column = 0; column < n; ++column) {
    const std::size_t source = order[column];
    result.values[column] = matrix[source * n + source] * output_scale;
    if (!std::isfinite(result.values[column]))
      throw std::overflow_error("CPU symmetric eigenvalue exceeds finite FP64 range");
    for (std::size_t row = 0; row < n; ++row)
      result.vectors[row * n + column] = vectors[row * n + source];
  }
  return result;
}

#if VIBEQC_HAS_OPENBLAS
[[maybe_unused]] void openblas_set_local_threads(int threads) {
#if VIBEQC_OPENBLAS_HAS_LOCAL_THREADS
#if VIBEQC_OPENBLAS_SCIPY_PREFIX
  (void)scipy_openblas_set_num_threads_local(threads);
#else
  (void)openblas_set_num_threads_local(threads);
#endif
#else
  (void)threads;
#endif
}

[[maybe_unused]] int openblas_set_local_threads_return_previous(int threads) {
#if VIBEQC_OPENBLAS_HAS_LOCAL_THREADS
#if VIBEQC_OPENBLAS_SCIPY_PREFIX
  return scipy_openblas_set_num_threads_local(threads);
#else
  return openblas_set_num_threads_local(threads);
#endif
#else
  (void)threads;
  return 0;
#endif
}

[[maybe_unused]] int openblas_get_global_threads() {
#if VIBEQC_OPENBLAS_HAS_GLOBAL_THREADS
#if VIBEQC_OPENBLAS_SCIPY_PREFIX
  return scipy_openblas_get_num_threads();
#else
  return openblas_get_num_threads();
#endif
#else
  return 0;
#endif
}

[[maybe_unused]] void openblas_set_global_threads(int threads) {
#if VIBEQC_OPENBLAS_HAS_GLOBAL_THREADS
#if VIBEQC_OPENBLAS_SCIPY_PREFIX
  scipy_openblas_set_num_threads(threads);
#else
  openblas_set_num_threads(threads);
#endif
#else
  (void)threads;
#endif
}

[[maybe_unused]] std::mutex& openblas_global_thread_mutex() {
  static std::mutex mutex;
  return mutex;
}

class OpenBlasThreadGuard {
 public:
  explicit OpenBlasThreadGuard(const CpuLinalgPlan& plan) {
#if VIBEQC_OPENBLAS_HAS_LOCAL_THREADS
    previous_ = openblas_set_local_threads_return_previous(plan.provider_threads);
    local_ = true;
#elif VIBEQC_OPENBLAS_HAS_GLOBAL_THREADS
    if (plan.thread_ownership != CpuLinalgThreadOwnership::provider_parallel)
      throw std::runtime_error(
          "OpenBLAS build lacks thread-local control; use provider-parallel ownership");
    global_lock_ = std::unique_lock<std::mutex>(openblas_global_thread_mutex());
    previous_ = openblas_get_global_threads();
    if (previous_ != plan.provider_threads) {
      openblas_set_global_threads(plan.provider_threads);
      global_changed_ = true;
    }
#else
    (void)plan;
    throw std::runtime_error("OpenBLAS provider lacks runtime thread control");
#endif
  }
  OpenBlasThreadGuard(const OpenBlasThreadGuard&) = delete;
  OpenBlasThreadGuard& operator=(const OpenBlasThreadGuard&) = delete;
  ~OpenBlasThreadGuard() {
    if (local_) openblas_set_local_threads(previous_);
    if (global_changed_) openblas_set_global_threads(previous_);
  }

 private:
  int previous_{1};
  bool local_{};
  bool global_changed_{};
  std::unique_lock<std::mutex> global_lock_;
};

void openblas_gemm(bool ta, bool tb, std::size_t m, std::size_t n, std::size_t k, const double* a,
                   const double* b, double* c, double alpha, double beta,
                   const CpuLinalgPlan& plan) {
  const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (m > limit || n > limit || k > limit)
    throw std::length_error("OpenBLAS GEMM dimensions exceed int range");
  OpenBlasThreadGuard guard(plan);
  const auto trans_a = ta ? CblasTrans : CblasNoTrans;
  const auto trans_b = tb ? CblasTrans : CblasNoTrans;
#if VIBEQC_OPENBLAS_SCIPY_PREFIX
  scipy_cblas_dgemm(CblasRowMajor, trans_a, trans_b, static_cast<int>(m), static_cast<int>(n),
                    static_cast<int>(k), alpha, a, static_cast<int>(ta ? m : k), b,
                    static_cast<int>(tb ? k : n), beta, c, static_cast<int>(n));
#else
  cblas_dgemm(CblasRowMajor, trans_a, trans_b, static_cast<int>(m), static_cast<int>(n),
              static_cast<int>(k), alpha, a, static_cast<int>(ta ? m : k), b,
              static_cast<int>(tb ? k : n), beta, c, static_cast<int>(n));
#endif
}

void openblas_syrk(bool upper, bool trans, std::size_t n, std::size_t k, const double* a, double* c,
                   double alpha, double beta, const CpuLinalgPlan& plan) {
  const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (n > limit || k > limit) throw std::length_error("OpenBLAS SYRK dimensions exceed int range");
  OpenBlasThreadGuard guard(plan);
  const auto triangle = upper ? CblasUpper : CblasLower;
  const auto transpose_a = trans ? CblasTrans : CblasNoTrans;
#if VIBEQC_OPENBLAS_SCIPY_PREFIX
  scipy_cblas_dsyrk(CblasRowMajor, triangle, transpose_a, static_cast<int>(n), static_cast<int>(k),
                    alpha, a, static_cast<int>(trans ? n : k), beta, c, static_cast<int>(n));
#else
  cblas_dsyrk(CblasRowMajor, triangle, transpose_a, static_cast<int>(n), static_cast<int>(k), alpha,
              a, static_cast<int>(trans ? n : k), beta, c, static_cast<int>(n));
#endif
}

void openblas_trsm(bool left, bool upper, bool trans, bool unit_diagonal, std::size_t m,
                   std::size_t n, const double* a, double* b, double alpha,
                   const CpuLinalgPlan& plan) {
  const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (m > limit || n > limit) throw std::length_error("OpenBLAS TRSM dimensions exceed int range");
  OpenBlasThreadGuard guard(plan);
  const auto side = left ? CblasLeft : CblasRight;
  const auto triangle = upper ? CblasUpper : CblasLower;
  const auto transpose_a = trans ? CblasTrans : CblasNoTrans;
  const auto diagonal = unit_diagonal ? CblasUnit : CblasNonUnit;
  const int order = static_cast<int>(left ? m : n);
#if VIBEQC_OPENBLAS_SCIPY_PREFIX
  scipy_cblas_dtrsm(CblasRowMajor, side, triangle, transpose_a, diagonal, static_cast<int>(m),
                    static_cast<int>(n), alpha, a, order, b, static_cast<int>(n));
#else
  cblas_dtrsm(CblasRowMajor, side, triangle, transpose_a, diagonal, static_cast<int>(m),
              static_cast<int>(n), alpha, a, order, b, static_cast<int>(n));
#endif
}

int openblas_cholesky_lower(double* matrix, std::size_t n, const CpuLinalgPlan& plan) {
#if VIBEQC_OPENBLAS_HAS_LAPACKE
  if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("OpenBLAS Cholesky dimension exceeds int range");
  OpenBlasThreadGuard guard(plan);
#if VIBEQC_OPENBLAS_SCIPY_PREFIX
  return static_cast<int>(scipy_LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', static_cast<int>(n), matrix,
                                               static_cast<int>(n)));
#else
  return static_cast<int>(
      LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', static_cast<int>(n), matrix, static_cast<int>(n)));
#endif
#else
  (void)matrix;
  (void)n;
  (void)plan;
  throw std::runtime_error("OpenBLAS provider was built without LAPACKE");
#endif
}

CpuSymmetricEigenResult openblas_symmetric_eigen(std::vector<double> matrix, std::size_t n,
                                                 const CpuLinalgPlan& plan) {
#if VIBEQC_OPENBLAS_HAS_LAPACKE
  if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("OpenBLAS eigensolver dimension exceeds int range");
  std::vector<double> values(n);
  OpenBlasThreadGuard guard(plan);
#if VIBEQC_OPENBLAS_SCIPY_PREFIX
  const int info =
      static_cast<int>(scipy_LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', static_cast<int>(n),
                                            matrix.data(), static_cast<int>(n), values.data()));
#else
  const int info =
      static_cast<int>(LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', static_cast<int>(n),
                                      matrix.data(), static_cast<int>(n), values.data()));
#endif
  if (info < 0) throw std::invalid_argument("OpenBLAS symmetric eigensolver rejected an argument");
  if (info > 0) throw std::runtime_error("OpenBLAS symmetric eigensolver did not converge");
  return {std::move(values), std::move(matrix)};
#else
  (void)matrix;
  (void)n;
  (void)plan;
  throw std::runtime_error("OpenBLAS provider was built without LAPACKE");
#endif
}
#endif

bool fits_openblas(std::size_t m, std::size_t n, std::size_t k) noexcept {
  const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  return m <= limit && n <= limit && k <= limit;
}

}  // namespace

bool cpu_openblas_built() noexcept { return VIBEQC_HAS_OPENBLAS != 0; }
bool cpu_openblas_lapack_built() noexcept {
  return VIBEQC_HAS_OPENBLAS != 0 && VIBEQC_OPENBLAS_HAS_LAPACKE != 0;
}
bool cpu_openblas_local_thread_control_built() noexcept {
  return VIBEQC_HAS_OPENBLAS != 0 && VIBEQC_OPENBLAS_HAS_LOCAL_THREADS != 0;
}
bool cpu_openblas_global_thread_control_built() noexcept {
  return VIBEQC_HAS_OPENBLAS != 0 && VIBEQC_OPENBLAS_HAS_GLOBAL_THREADS != 0;
}

CpuLinalgProvider resolve_cpu_linalg_provider(const CpuLinalgPlan& plan, bool require_lapack) {
  validate_plan(plan);
  if (plan.provider == CpuLinalgProvider::scalar) return CpuLinalgProvider::scalar;
  const bool thread_control = plan.thread_ownership == CpuLinalgThreadOwnership::task_parallel
                                  ? cpu_openblas_local_thread_control_built()
                                  : (cpu_openblas_local_thread_control_built() ||
                                     cpu_openblas_global_thread_control_built());
  const bool usable =
      cpu_openblas_built() && thread_control && (!require_lapack || cpu_openblas_lapack_built());
  if (plan.provider == CpuLinalgProvider::openblas) {
    if (!usable)
      throw std::runtime_error(
          require_lapack
              ? "requested OpenBLAS LAPACK provider is unavailable"
              : "requested OpenBLAS provider is unavailable for this thread-ownership mode");
    return CpuLinalgProvider::openblas;
  }
  return usable ? CpuLinalgProvider::openblas : CpuLinalgProvider::scalar;
}

std::string_view cpu_linalg_target_name() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX512F__) && defined(__FMA__)
  return "x86_64-avx512f-fma";
#elif defined(__AVX2__) && defined(__FMA__)
  return "x86_64-avx2-fma";
#elif defined(__AVX2__)
  return "x86_64-avx2";
#else
  return "x86_64-generic";
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
  return "aarch64-generic";
#else
  return "generic";
#endif
}

std::string_view cpu_linalg_provider_name(CpuLinalgProvider provider) noexcept {
  switch (provider) {
    case CpuLinalgProvider::automatic:
      return "automatic";
    case CpuLinalgProvider::scalar:
      return "scalar";
    case CpuLinalgProvider::openblas:
      return "openblas";
  }
  return "unknown";
}

CpuLinalgDiagnostic cpu_linalg_diagnostic(const CpuLinalgPlan& plan) {
  const auto provider = resolve_cpu_linalg_provider(plan);
  return {.provider = provider,
          .thread_ownership = plan.thread_ownership,
          .provider_threads = plan.provider_threads,
          .external_provider = provider == CpuLinalgProvider::openblas,
          .lapack_available = cpu_openblas_lapack_built(),
          .local_thread_control = cpu_openblas_local_thread_control_built(),
          .global_thread_control = cpu_openblas_global_thread_control_built(),
          .cpu_target = cpu_linalg_target_name()};
}

void cpu_gemm(char a_trans, char b_trans, std::size_t m, std::size_t n, std::size_t k,
              const double* a, const double* b, double* c, double alpha, double beta,
              const CpuLinalgPlan& plan) {
  const bool ta = transpose(a_trans);
  const bool tb = transpose(b_trans);
  validate_plan(plan);
  if (!m || !n) return;
  const auto elements = checked_matrix_elements(m, n);
  if (!c) throw std::invalid_argument("CPU GEMM received null storage");
  if (!k || alpha == 0.0) {
    if (beta == 0.0)
      std::fill(c, c + elements, 0.0);
    else if (beta != 1.0)
      for (std::size_t i = 0; i < elements; ++i) c[i] *= beta;
    return;
  }
  checked_matrix_elements(m, k);
  checked_matrix_elements(k, n);
  if (!a || !b) throw std::invalid_argument("CPU GEMM received null storage");

  CpuLinalgProvider provider = plan.provider;
  if (provider == CpuLinalgProvider::automatic) {
    provider =
        fits_openblas(m, n, k) ? resolve_cpu_linalg_provider(plan) : CpuLinalgProvider::scalar;
  } else {
    provider = resolve_cpu_linalg_provider(plan);
  }
#if VIBEQC_HAS_OPENBLAS
  if (provider == CpuLinalgProvider::openblas) {
    openblas_gemm(ta, tb, m, n, k, a, b, c, alpha, beta, plan);
    return;
  }
#endif
  scalar_gemm(ta, tb, m, n, k, a, b, c, alpha, beta);
}

void cpu_syrk(char uplo, char trans, std::size_t n, std::size_t k, const double* a, double* c,
              double alpha, double beta, const CpuLinalgPlan& plan) {
  const bool upper = upper_triangle(uplo);
  const bool transposed = syrk_transpose(trans);
  validate_plan(plan);
  if (!n) return;
  checked_matrix_elements(n, n);
  if (!c) throw std::invalid_argument("CPU SYRK received null storage");
  if (!k || alpha == 0.0) {
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t first_column = upper ? i : 0;
      const std::size_t last_column = upper ? n : i + 1;
      for (std::size_t j = first_column; j < last_column; ++j) {
        const std::size_t index = i * n + j;
        if (beta == 0.0)
          c[index] = 0.0;
        else if (beta != 1.0)
          c[index] *= beta;
      }
    }
    return;
  }
  checked_matrix_elements(transposed ? k : n, transposed ? n : k);
  if (!a) throw std::invalid_argument("CPU SYRK received null storage");

  CpuLinalgProvider provider = plan.provider;
  if (provider == CpuLinalgProvider::automatic) {
    provider =
        fits_openblas(n, n, k) ? resolve_cpu_linalg_provider(plan) : CpuLinalgProvider::scalar;
  } else {
    provider = resolve_cpu_linalg_provider(plan);
  }
#if VIBEQC_HAS_OPENBLAS
  if (provider == CpuLinalgProvider::openblas) {
    openblas_syrk(upper, transposed, n, k, a, c, alpha, beta, plan);
    return;
  }
#endif
  scalar_syrk(upper, transposed, n, k, a, c, alpha, beta);
}

void cpu_trsm(char side, char uplo, char trans, char diag, std::size_t m, std::size_t n,
              const double* a, double* b, double alpha, const CpuLinalgPlan& plan) {
  const bool left = trsm_left_side(side);
  const bool upper = trsm_upper_triangle(uplo);
  const bool transposed = trsm_transpose(trans);
  const bool unit_diagonal = trsm_unit_diagonal(diag);
  validate_plan(plan);
  if (!m || !n) return;
  const auto elements = checked_matrix_elements(m, n);
  if (!b) throw std::invalid_argument("CPU TRSM received null output storage");
  if (alpha == 0.0) {
    std::fill(b, b + elements, 0.0);
    return;
  }
  const std::size_t order = left ? m : n;
  checked_matrix_elements(order, order);
  if (!a) throw std::invalid_argument("CPU TRSM received null triangular storage");

  CpuLinalgProvider provider = plan.provider;
  if (provider == CpuLinalgProvider::automatic) {
    provider =
        fits_openblas(m, n, order) ? resolve_cpu_linalg_provider(plan) : CpuLinalgProvider::scalar;
  } else {
    provider = resolve_cpu_linalg_provider(plan);
  }
#if VIBEQC_HAS_OPENBLAS
  if (provider == CpuLinalgProvider::openblas) {
    openblas_trsm(left, upper, transposed, unit_diagonal, m, n, a, b, alpha, plan);
    return;
  }
#endif
  scalar_trsm(left, upper, transposed, unit_diagonal, m, n, a, b, alpha);
}

int cpu_cholesky_lower(double* matrix, std::size_t n, const CpuLinalgPlan& plan) {
  validate_plan(plan);
  if (!n) return 0;
  checked_matrix_elements(n, n);
  if (!matrix) throw std::invalid_argument("CPU Cholesky received null storage");
#if VIBEQC_HAS_OPENBLAS
  if (resolve_cpu_linalg_provider(plan, true) == CpuLinalgProvider::openblas)
    return openblas_cholesky_lower(matrix, n, plan);
#else
  (void)resolve_cpu_linalg_provider(plan, true);
#endif
  return scalar_cholesky_lower(matrix, n);
}

CpuSymmetricEigenResult cpu_symmetric_eigen(std::vector<double> matrix, std::size_t n,
                                            const CpuLinalgPlan& plan) {
  validate_plan(plan);
  if (!n || n > std::numeric_limits<std::size_t>::max() / n || matrix.size() != n * n)
    throw std::invalid_argument("CPU symmetric eigensolver dimensions are inconsistent");
  if (!std::all_of(matrix.begin(), matrix.end(), [](double x) { return std::isfinite(x); }))
    throw std::invalid_argument("CPU symmetric eigensolver requires finite input");
#if VIBEQC_HAS_OPENBLAS
  if (resolve_cpu_linalg_provider(plan, true) == CpuLinalgProvider::openblas)
    return openblas_symmetric_eigen(std::move(matrix), n, plan);
#else
  (void)resolve_cpu_linalg_provider(plan, true);
#endif
  return scalar_symmetric_eigen(std::move(matrix), n);
}

}  // namespace vibeqc::tensor
