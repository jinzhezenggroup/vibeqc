#include "scf/solver/diis.hpp"

#include <vector>

#include "generated_scf_array_native.hpp"
#include "solver/diis_coefficients.hpp"

namespace vibeqc::scf::solver {
Diis::Diis(std::size_t capacity, bool normalize_metric)
    : history_(capacity), normalize_metric_(normalize_metric) {}

/** Actual retained numerical capacity; no temporary extrapolation work. */
std::size_t Diis::numeric_capacity() const noexcept { return history_.numeric_capacity_bytes(); }

void Diis::clear() { history_.clear(); }

Matrix Diis::update(const Matrix& fock, const Matrix& residual) {
  if (history_.capacity() < 2) return fock;
  history_.push(fock, residual);
  if (history_.size() < 2) return fock;

  for (;;) {
    const std::size_t m = history_.size();
    Matrix gram(m * m, 0.0);
    generated::diis_gram(gram.data(), m, history_.errors(), m, residual.size());

    ::vibeqc::solver::detail::DiisCoefficientPolicy policy;
    if (normalize_metric_) {
      policy.metric_scaling = ::vibeqc::solver::detail::DiisMetricScaling::MaximumDiagonal;
      policy.failure_retirement_floor = 2;
    }
    std::vector<double> coefficients;
    const auto action =
        ::vibeqc::solver::detail::solve_diis_coefficients(gram, m, policy, coefficients);
    if (action == ::vibeqc::solver::detail::DiisCoefficientAction::RetireOldest) {
      history_.retire_oldest();
      continue;
    }
    if (action == ::vibeqc::solver::detail::DiisCoefficientAction::RetainCurrent) return fock;

    Matrix extrapolated(fock.size());
    generated::diis_extrapolate(extrapolated.data(), history_.vectors(), coefficients.data(), m,
                                fock.size());
    return extrapolated;
  }
}

}  // namespace vibeqc::scf::solver
