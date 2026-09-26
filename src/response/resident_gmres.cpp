#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include "response/resident_krylov.hpp"

namespace vibeqc::response {
namespace {
using WorkspaceVector = runtime::TrackedVector<double>;

std::size_t checked_add(std::size_t first, std::size_t second) {
  if (second > std::numeric_limits<std::size_t>::max() - first)
    throw std::overflow_error("resident GMRES host workspace overflow");
  return first + second;
}

bool finite(std::span<const double> values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

double resident_norm(ResidentKrylovBackend& backend, std::size_t slot) {
  try {
    return backend.norm(slot);
  } catch (const std::invalid_argument&) {
    // stable_norm-based backends reject nonfinite vectors by exception.
    return std::numeric_limits<double>::infinity();
  } catch (const std::overflow_error&) {
    return std::numeric_limits<double>::infinity();
  }
}

double relative_residual(double residual, double rhs) {
  if (rhs > 0.0) return residual / rhs;
  return residual == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
}

bool solve_upper(std::span<const double> hessenberg, std::size_t stride,
                 std::span<const double> transformed_rhs, std::size_t count, double tolerance,
                 std::span<double> solution) {
  std::fill(solution.begin(), solution.end(), 0.0);
  for (std::size_t reverse = count; reverse > 0; --reverse) {
    const auto row = reverse - 1;
    double value = transformed_rhs[row];
    for (std::size_t column = row + 1; column < count; ++column)
      value -= hessenberg[row * stride + column] * solution[column];
    const double diagonal = hessenberg[row * stride + row];
    if (!std::isfinite(diagonal) || std::abs(diagonal) <= tolerance) return false;
    solution[row] = value / diagonal;
    if (!std::isfinite(solution[row])) return false;
  }
  return true;
}

struct Slots {
  explicit Slots(std::size_t restart)
      : basis(3),
        preconditioned(basis + restart + 1),
        work(preconditioned + restart),
        candidate(work + 1),
        candidate_image(candidate + 1),
        candidate_residual(candidate_image + 1),
        best_x(candidate_residual + 1),
        best_residual(best_x + 1),
        count(best_residual + 1) {}

  static constexpr std::size_t x = 0;
  static constexpr std::size_t rhs = 1;
  static constexpr std::size_t residual = 2;
  std::size_t basis, preconditioned, work, candidate, candidate_image, candidate_residual;
  std::size_t best_x, best_residual, count;
};

ResidentGmresResult make_result(const ResidentGmresWorkspace& workspace, std::size_t resident_bytes,
                                WorkspaceVector solution, GmresStatus status, double residual,
                                double rhs_norm, std::size_t iterations, std::size_t restarts,
                                std::size_t operator_actions) {
  GmresResult result;
  result.solution = std::move(solution);
  result.status = status;
  result.residual_norm = residual;
  result.relative_residual = relative_residual(residual, rhs_norm);
  result.iterations = iterations;
  result.restarts = restarts;
  result.operator_actions = operator_actions;
  result.preconditioner_actions = 0;
  result.workspace_bytes = checked_add(workspace.host_scalar_bytes, workspace.host_result_bytes);
  if (const auto counter = result.solution.get_allocator().counter()) {
    const auto measured = counter->snapshot();
    result.measured_workspace_peak_bytes = measured.peak_bytes;
    result.workspace_allocation_count = measured.allocation_count;
  }
  return {std::move(result), workspace, resident_bytes};
}
}  // namespace

ResidentGmresResult solve_gmres_resident(const GmresPlan& plan, ResidentKrylovBackend& backend,
                                         std::span<const double> rhs,
                                         std::span<const double> initial_guess) {
  const auto workspace = resident_gmres_workspace(plan);
  const auto host_bytes = checked_add(workspace.host_scalar_bytes, workspace.host_result_bytes);
  if (host_bytes > plan.options.max_workspace_bytes) {
    GmresResult refused;
    refused.status = GmresStatus::workspace_limit;
    refused.residual_norm = std::numeric_limits<double>::infinity();
    refused.relative_residual = std::numeric_limits<double>::infinity();
    refused.workspace_bytes = host_bytes;
    return {std::move(refused), workspace, 0};
  }

  const auto n = plan.dimension;
  const runtime::TrackedAllocator<double> allocator(std::make_shared<runtime::AllocationCounter>());
  if (rhs.size() != n || !finite(rhs) ||
      (!initial_guess.empty() && (initial_guess.size() != n || !finite(initial_guess)))) {
    WorkspaceVector solution(n, 0.0, allocator);
    return make_result(workspace, 0, std::move(solution), GmresStatus::nonfinite_input,
                       std::numeric_limits<double>::infinity(), 0.0, 0, 0, 0);
  }

  validate_resident_gmres_backend(plan, backend);
  const auto slots = Slots(plan.restart);
  if (slots.count != workspace.vector_slots)
    throw std::logic_error("resident GMRES slot layout does not match its workspace contract");

  double rhs_norm = 0.0;
  try {
    rhs_norm = stable_norm(rhs);
  } catch (const std::exception&) {
    WorkspaceVector solution(n, 0.0, allocator);
    if (!initial_guess.empty())
      std::copy(initial_guess.begin(), initial_guess.end(), solution.begin());
    return make_result(workspace, backend.owned_resident_bytes(), std::move(solution),
                       GmresStatus::nonfinite_input, std::numeric_limits<double>::infinity(),
                       0.0, 0, 0, 0);
  }
  const double target =
      std::max(plan.options.absolute_tolerance, plan.options.relative_tolerance * rhs_norm);
  if (initial_guess.empty() && rhs_norm <= target) {
    WorkspaceVector solution(n, 0.0, allocator);
    return make_result(workspace, backend.owned_resident_bytes(), std::move(solution),
                       GmresStatus::initial_residual, rhs_norm, rhs_norm, 0, 0, 0);
  }

  backend.upload(Slots::rhs, rhs);
  if (initial_guess.empty())
    backend.zero(Slots::x);
  else
    backend.upload(Slots::x, initial_guess);
  backend.copy(Slots::residual, Slots::rhs);

  std::size_t operator_actions = 0;
  if (!initial_guess.empty()) {
    backend.apply(slots.work, Slots::x);
    ++operator_actions;
    backend.axpy(Slots::residual, -1.0, slots.work);
  }
  double beta = resident_norm(backend, Slots::residual);
  if (!std::isfinite(beta)) {
    WorkspaceVector solution(n, 0.0, allocator);
    backend.download(Slots::x, solution);
    return make_result(workspace, backend.owned_resident_bytes(), std::move(solution),
                       GmresStatus::nonfinite_operator, std::numeric_limits<double>::infinity(),
                       rhs_norm, 0, 0, operator_actions);
  }
  if (beta <= target) {
    WorkspaceVector solution(n, 0.0, allocator);
    backend.download(Slots::x, solution);
    return make_result(workspace, backend.owned_resident_bytes(), std::move(solution),
                       GmresStatus::initial_residual, beta, rhs_norm, 0, 0, operator_actions);
  }

  const auto restart = plan.restart;
  WorkspaceVector hessenberg((restart + 1) * restart, 0.0, allocator);
  WorkspaceVector cosine(restart, 0.0, allocator), sine(restart, 0.0, allocator),
      transformed(restart + 1, 0.0, allocator), coefficients(restart, 0.0, allocator);

  backend.copy(slots.best_x, Slots::x);
  backend.copy(slots.best_residual, Slots::residual);
  std::size_t iterations = 0, restarts = 0, stagnation = 0;

  auto finish = [&](std::size_t solution_slot, GmresStatus status, double residual) {
    WorkspaceVector solution(n, 0.0, allocator);
    backend.download(solution_slot, solution);
    return make_result(workspace, backend.owned_resident_bytes(), std::move(solution), status,
                       residual, rhs_norm, iterations, restarts, operator_actions);
  };

  while (iterations < plan.options.max_iterations) {
    std::fill(hessenberg.begin(), hessenberg.end(), 0.0);
    std::fill(transformed.begin(), transformed.end(), 0.0);
    transformed[0] = beta;
    backend.copy(slots.basis, Slots::residual);
    backend.scale(slots.basis, 1.0 / beta);
    double best_norm = beta;
    bool completed_cycle = false;

    for (std::size_t column = 0; column < restart && iterations < plan.options.max_iterations;
         ++column) {
      const auto basis_column = slots.basis + column;
      const auto z_column = slots.preconditioned + column;
      backend.copy(z_column, basis_column);
      backend.apply(slots.work, z_column);
      ++operator_actions;

      bool nonfinite = false;
      for (unsigned pass = 0; pass < plan.options.reorthogonalize && !nonfinite; ++pass) {
        for (std::size_t row = 0; row <= column; ++row) {
          const double projection = backend.dot(slots.basis + row, slots.work);
          if (!std::isfinite(projection)) {
            nonfinite = true;
            break;
          }
          hessenberg[row * restart + column] += projection;
          backend.axpy(slots.work, -projection, slots.basis + row);
        }
      }
      if (nonfinite) return finish(slots.best_x, GmresStatus::nonfinite_operator, best_norm);

      const double next_norm = resident_norm(backend, slots.work);
      if (!std::isfinite(next_norm))
        return finish(slots.best_x, GmresStatus::nonfinite_operator, best_norm);
      hessenberg[(column + 1) * restart + column] = next_norm;
      const bool broke_down = next_norm <= plan.options.breakdown_tolerance;
      if (!broke_down) {
        backend.copy(slots.basis + column + 1, slots.work);
        backend.scale(slots.basis + column + 1, 1.0 / next_norm);
      }

      for (std::size_t row = 0; row < column; ++row) {
        const double upper = hessenberg[row * restart + column];
        const double lower = hessenberg[(row + 1) * restart + column];
        hessenberg[row * restart + column] = cosine[row] * upper + sine[row] * lower;
        hessenberg[(row + 1) * restart + column] = -sine[row] * upper + cosine[row] * lower;
      }
      const double upper = hessenberg[column * restart + column];
      const double lower = hessenberg[(column + 1) * restart + column];
      const double rotation_norm = std::hypot(upper, lower);
      if (rotation_norm > plan.options.breakdown_tolerance) {
        cosine[column] = upper / rotation_norm;
        sine[column] = lower / rotation_norm;
      } else {
        cosine[column] = 1.0;
        sine[column] = 0.0;
      }
      hessenberg[column * restart + column] = cosine[column] * upper + sine[column] * lower;
      hessenberg[(column + 1) * restart + column] = 0.0;
      const double transformed_upper = transformed[column];
      transformed[column] = cosine[column] * transformed_upper;
      transformed[column + 1] = -sine[column] * transformed_upper;
      ++iterations;

      const bool checkpoint = broke_down || column + 1 == restart ||
                              iterations == plan.options.max_iterations ||
                              iterations % plan.options.true_residual_every == 0;
      if (!checkpoint) continue;
      const std::size_t columns = column + 1;
      const bool solvable = solve_upper(hessenberg, restart, transformed, columns,
                                        plan.options.breakdown_tolerance, coefficients);
      backend.copy(slots.candidate, Slots::x);
      if (solvable)
        for (std::size_t vector = 0; vector < columns; ++vector)
          backend.axpy(slots.candidate, coefficients[vector], slots.preconditioned + vector);
      backend.apply(slots.candidate_image, slots.candidate);
      ++operator_actions;
      backend.copy(slots.candidate_residual, Slots::rhs);
      backend.axpy(slots.candidate_residual, -1.0, slots.candidate_image);
      const double candidate_norm = resident_norm(backend, slots.candidate_residual);
      if (!std::isfinite(candidate_norm))
        return finish(slots.best_x, GmresStatus::nonfinite_operator, best_norm);

      if (candidate_norm < best_norm) {
        best_norm = candidate_norm;
        backend.copy(slots.best_x, slots.candidate);
        backend.copy(slots.best_residual, slots.candidate_residual);
        stagnation = 0;
      } else if (candidate_norm >= best_norm * (1.0 - plan.options.stagnation_tolerance)) {
        ++stagnation;
      }
      if (candidate_norm <= target)
        return finish(slots.candidate, GmresStatus::converged, candidate_norm);
      if (broke_down || !solvable) return finish(slots.best_x, GmresStatus::breakdown, best_norm);
      if (stagnation >= plan.options.stagnation_window)
        return finish(slots.best_x, GmresStatus::stagnation, best_norm);
      if (column + 1 == restart || iterations == plan.options.max_iterations) {
        completed_cycle = true;
        break;
      }
    }

    backend.copy(Slots::x, slots.best_x);
    backend.copy(Slots::residual, slots.best_residual);
    beta = best_norm;
    if (iterations >= plan.options.max_iterations) break;
    if (!completed_cycle || beta == 0.0) break;
    ++restarts;
  }

  return finish(slots.best_x, GmresStatus::max_iterations, beta);
}

}  // namespace vibeqc::response
