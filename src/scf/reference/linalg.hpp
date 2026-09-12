#ifndef VIBEQC_SCF_REFERENCE_LINALG_HPP
#define VIBEQC_SCF_REFERENCE_LINALG_HPP

#include <cstddef>
#include <vector>

namespace vibeqc::scf::reference {

/** Owned row-major FP64 matrix used by the independent CPU reference.
 * Unless documented otherwise, callers supply finite n-by-n matrices and
 * validate dimensions before entering these small deterministic routines.
 */
using Matrix = std::vector<double>;

/** Row-major indexing shared by reference numerics and their consumers. */
inline std::size_t index(std::size_t row, std::size_t column, std::size_t n) {
  return row * n + column;
}

/** Eigenvalues in ascending order; the corresponding vectors are columns. */
struct EigenResult {
  std::vector<double> values;
  Matrix vectors;
};

/** Form a square identity matrix. */
Matrix identity(std::size_t n);
/** Ordinary row-major product; retain the reference accumulation order. */
Matrix multiply(const Matrix& a, const Matrix& b, std::size_t n);
/** Return the transpose of a square row-major matrix. */
Matrix transpose(const Matrix& a, std::size_t n);
/** Deterministic Jacobi oracle for real symmetric matrices.
 * Preserve the historical 1e-14 off-diagonal stopping rule and sweep bound;
 * this independent reference must not become an alias for generated kernels.
 */
EigenResult symmetric_eigen(Matrix matrix, std::size_t n);
/** Form S^(-1/2); reject overlap eigenvalues below the existing 1e-10 gate. */
Matrix symmetric_orthogonalizer(const Matrix& overlap, std::size_t n);
/** Solve F C = S C e using the caller's symmetric orthogonalizer X=S^(-1/2). */
EigenResult generalized_eigen(const Matrix& fock, const Matrix& orthogonalizer, std::size_t n);
/** Euclidean inner product of equally sized flattened matrices/vectors. */
double dot(const Matrix& a, const Matrix& b);
/** Pivoted dense solve. Return false at a pivot below 1e-14, leaving x unchanged. */
bool solve_linear(Matrix a, std::vector<double> b, std::vector<double>& x, std::size_t n);

}  // namespace vibeqc::scf::reference
#endif
