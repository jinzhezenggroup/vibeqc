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
