#include "tensor/cpu_linalg.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
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

void validate_plan(const CpuLinalgPlan& plan) {
  if (plan.provider_threads < 1)
    throw std::invalid_argument("CPU linear algebra thread count must be positive");
  if (plan.thread_ownership == CpuLinalgThreadOwnership::task_parallel &&
      plan.provider_threads != 1)
    throw std::invalid_argument("task-parallel CPU linear algebra requires single-thread provider");
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
      c[i * n + j] = alpha * sum + beta * c[i * n + j];
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
    openblas_set_global_threads(plan.provider_threads);
    global_ = true;
#else
    (void)plan;
    throw std::runtime_error("OpenBLAS provider lacks runtime thread control");
#endif
  }
  OpenBlasThreadGuard(const OpenBlasThreadGuard&) = delete;
  OpenBlasThreadGuard& operator=(const OpenBlasThreadGuard&) = delete;
  ~OpenBlasThreadGuard() {
    if (local_) openblas_set_local_threads(previous_);
    if (global_) openblas_set_global_threads(previous_);
  }

 private:
  int previous_{1};
  bool local_{};
  bool global_{};
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
          .global_thread_control = cpu_openblas_global_thread_control_built()};
}

void cpu_gemm(char a_trans, char b_trans, std::size_t m, std::size_t n, std::size_t k,
              const double* a, const double* b, double* c, double alpha, double beta,
              const CpuLinalgPlan& plan) {
  const bool ta = transpose(a_trans);
  const bool tb = transpose(b_trans);
  validate_plan(plan);
  if (!m || !n) return;
  if (!c || (k && (!a || !b))) throw std::invalid_argument("CPU GEMM received null storage");
  if (!k) {
    for (std::size_t i = 0; i < m * n; ++i) c[i] *= beta;
    return;
  }

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

int cpu_cholesky_lower(double* matrix, std::size_t n, const CpuLinalgPlan& plan) {
  validate_plan(plan);
  if (!n) return 0;
  if (!matrix) throw std::invalid_argument("CPU Cholesky received null storage");
#if VIBEQC_HAS_OPENBLAS
  if (resolve_cpu_linalg_provider(plan, true) == CpuLinalgProvider::openblas)
    return openblas_cholesky_lower(matrix, n, plan);
#else
  (void)resolve_cpu_linalg_provider(plan, true);
#endif
  return scalar_cholesky_lower(matrix, n);
}

}  // namespace vibeqc::tensor
