"""Compiler-owned batched occupied projection and small metric layout lowering.

The runtime reuses the existing DF occupied adjoint; this module only emits
dense contractions. No runtime, oracle or device import is required.
"""

from __future__ import annotations

PROJECTION_EQUATIONS = ("qmn,ni->qmi", "mj,qmi->qji")
METRIC_EQUATIONS = ("pq,pij->qij", "pq,qij->pij")


def emit_occupied_response_helpers() -> str:
    """Return allocation-free C++ helpers for the established cuBLAS provider."""
    return r"""
/** Bounded auxiliary panel size. extra_raw=1 reserves a second raw panel
 * for source-major -> auxiliary-major conversion. All storage is already
 * owned; neither the emitter nor the caller may widen its allowance.
 */
inline std::size_t df_occupied_projection_tile(
    std::size_t n, std::size_t rank, std::size_t auxiliary,
    std::size_t capacity, std::size_t requested, bool extra_raw) {
  const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (!n || n > limit || rank > n || !auxiliary || auxiliary > limit || !requested)
    return 0;
  if (n > limit / n) return 0;
  const auto matrix = n * n;
  if (matrix > std::numeric_limits<std::size_t>::max() / 3) return 0;
  const auto per_auxiliary = (extra_raw ? 2 : 1) * matrix + n * rank;
  auto tile = capacity / per_auxiliary;
  if (tile > auxiliary) tile = auxiliary;
  if (tile > requested) tile = requested;
  return tile;
}

inline cublasStatus_t df_occupied_project_panel(
    cublasHandle_t blas, int n, int rank, int auxiliary, int begin, int count,
    const double* coefficients, const double* panels, double* temporary,
    double* projected) {
  if (n <= 0 || rank < 0 || rank > n || auxiliary <= 0 || begin < 0 ||
      begin > auxiliary || count < 0 || count > auxiliary - begin ||
      n > std::numeric_limits<int>::max() / n)
    return CUBLAS_STATUS_INVALID_VALUE;
  if (!count || !rank) return CUBLAS_STATUS_SUCCESS;
  if (!coefficients || !panels || !temporary || !projected)
    return CUBLAS_STATUS_INVALID_VALUE;
  const long long matrix = static_cast<long long>(n) * n;
  const long long nr = static_cast<long long>(n) * rank;
  const long long rr = static_cast<long long>(rank) * rank;
  const double one = 1, zero = 0;
  auto status = cublasDgemmStridedBatched(
      blas, CUBLAS_OP_N, CUBLAS_OP_N, n, rank, n, &one,
      panels, n, matrix, coefficients, n, 0, &zero, temporary, n, nr, count);
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  return cublasDgemmStridedBatched(
      blas, CUBLAS_OP_T, CUBLAS_OP_N, rank, rank, n, &one,
      coefficients, n, 0, temporary, n, nr, &zero,
      projected + static_cast<std::size_t>(begin) * rr, rank, rr, count);
}

/** Finish the exact fitted occupied projection from the final-K linear
 * factor U[mu,j,Q]=sum_nu B[Q,mu,nu] C[nu,j]. The packed K kernel stores Q
 * fastest, so one GEMM contracts C over mu into [i,j,Q] order. The caller
 * gathers that result into [Q,i,j]; treating U as Q-major GEMM batches reads
 * unrelated occupied/auxiliary elements once rank or auxiliary exceeds one.
 */
inline cublasStatus_t df_occupied_finish_projection(
    cublasHandle_t blas, int n, int rank, int auxiliary,
    const double* coefficients, const double* linear, double* pair_major) {
  if (n <= 0 || rank <= 0 || rank > n || auxiliary <= 0 ||
      auxiliary > std::numeric_limits<int>::max() / rank ||
      !coefficients || !linear || !pair_major)
    return CUBLAS_STATUS_INVALID_VALUE;
  const int ar = auxiliary * rank;
  const double one = 1, zero = 0;
  return cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, ar, rank, n, &one,
                     linear, ar, coefficients, n, &zero, pair_major, ar);
}

inline cublasStatus_t df_occupied_to_metric_eigenbasis(
    cublasHandle_t blas, int auxiliary, int rank_squared,
    const double* eigenvectors, const double* projected, double* eigenfactors) {
  if (auxiliary <= 0 || rank_squared <= 0 || !eigenvectors || !projected || !eigenfactors)
    return CUBLAS_STATUS_INVALID_VALUE;
  const double one = 1, zero = 0;
  return cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_T,
                     auxiliary, rank_squared, auxiliary, &one,
                     eigenvectors, auxiliary, projected, rank_squared,
                     &zero, eigenfactors, auxiliary);
}

/** Full-rank fitted B already contains one inverse root. Apply the immutable
 * plan's symmetric second root to S[aux,occ,occ] in one contraction. This
 * never constructs M^-1 or reconstructs a discarded metric direction.
 * Input and output occupy disjoint, already charged response intervals.
 */
inline cublasStatus_t df_occupied_apply_metric_root(
    cublasHandle_t blas, int auxiliary, int rank_squared,
    const double* inverse_root, const double* projected, double* fitted) {
  if (auxiliary <= 0 || rank_squared <= 0 || !inverse_root || !projected ||
      !fitted || projected == fitted)
    return CUBLAS_STATUS_INVALID_VALUE;
  const double one = 1, zero = 0;
  return cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                     rank_squared, auxiliary, auxiliary, &one,
                     projected, rank_squared, inverse_root, auxiliary,
                     &zero, fitted, rank_squared);
}

/** Diagonal-first coordinates for a symmetric occupied matrix. The first
 * rank entries retain the trace directly; each remaining entry denotes both
 * (i,j) and (j,i), with unit weight until the Gram contraction below.
 * The caller must prove symmetry from its physical packed AO source.
 */
#if defined(__CUDACC__)
__host__ __device__
#endif
inline std::size_t df_occupied_symmetric_pair(
    std::size_t rank, std::size_t i, std::size_t j) {
  if (i == j) return i;
  const auto hi = i > j ? i : j, lo = i > j ? j : i;
  return rank + hi * (hi - 1) / 2 + lo;
}

/** Add the symmetric occupied Gram into the lower auxiliary triangle.
 * Diagonal pairs contribute once and off-diagonal pairs twice, reproducing
 * sum_ij U[P,i,j] U[Q,i,j] without sqrt(2) rescaling. beta=1 preserves the
 * already accumulated Coulomb response. The caller mirrors the triangle
 * before any consumer reads the complete metric adjoint.
 */
inline cublasStatus_t df_occupied_symmetric_metric_gram(
    cublasHandle_t blas, int auxiliary, int rank, double coefficient,
    const double* factors, double* metric, bool* triangular = nullptr) {
  if (auxiliary <= 0 || rank <= 0 || !factors || !metric || factors == metric)
    return CUBLAS_STATUS_INVALID_VALUE;
  const auto pairs = static_cast<long long>(rank) * (rank + 1LL) / 2;
  if (pairs > std::numeric_limits<int>::max()) return CUBLAS_STATUS_INVALID_VALUE;
  const auto stride = static_cast<int>(pairs);
  const double one = 1, twice = 2 * coefficient;
  // Match the occupied-K provider contract: a provider without SYRK may
  // update both triangles with GEMM. The caller consumes/mirrors the lower
  // triangle and records the actual full-product work in that case.
  const auto product = [&](auto handle, int count, const double* scale,
                           const double* input) {
    if constexpr (requires {
        cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, auxiliary,
                    count, scale, input, stride, &one, metric, auxiliary);
    }) {
      if (triangular) *triangular = true;
      return cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, auxiliary,
                         count, scale, input, stride, &one, metric, auxiliary);
    } else {
      if (triangular) *triangular = false;
      return cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, auxiliary, auxiliary,
                         count, scale, input, stride, input, stride, &one,
                         metric, auxiliary);
    }
  };
  auto status = product(blas, rank, &coefficient, factors);
  if (status != CUBLAS_STATUS_SUCCESS || stride == rank) return status;
  return product(blas, stride - rank, &twice, factors + rank);
}

inline cublasStatus_t df_occupied_from_metric_eigenbasis(
    cublasHandle_t blas, int auxiliary, int rank_squared,
    const double* eigenvectors, const double* eigenfactors, double* projected) {
  if (auxiliary <= 0 || rank_squared <= 0 || !eigenvectors || !eigenfactors || !projected)
    return CUBLAS_STATUS_INVALID_VALUE;
  const double one = 1, zero = 0;
  return cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_T,
                     rank_squared, auxiliary, auxiliary, &one,
                     eigenfactors, auxiliary, eigenvectors, auxiliary,
                     &zero, projected, rank_squared);
}
"""
