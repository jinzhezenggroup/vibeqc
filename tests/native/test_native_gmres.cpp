#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "response/native_gmres.hpp"
#include "response/resident_krylov.hpp"
#include "response/solve.hpp"

namespace {
using vibeqc::response::GmresOptions;
using vibeqc::response::GmresStatus;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct DenseOperator {
  std::size_t dimension{};
  std::vector<double> values;
  std::size_t actions{};
  bool emit_nonfinite{};

  void operator()(std::span<const double> input, std::span<double> output) {
    ++actions;
    require(input.size() == dimension && output.size() == dimension,
            "operator received the wrong dimension");
    if (emit_nonfinite) {
      std::fill(output.begin(), output.end(), std::numeric_limits<double>::quiet_NaN());
      return;
    }
    for (std::size_t row = 0; row < dimension; ++row) {
      output[row] = 0.0;
      for (std::size_t column = 0; column < dimension; ++column)
        output[row] += values[row * dimension + column] * input[column];
    }
  }
};

double explicit_residual(const DenseOperator& matrix, std::span<const double> x,
                         std::span<const double> rhs) {
  std::vector<double> residual(rhs.begin(), rhs.end());
  for (std::size_t row = 0; row < matrix.dimension; ++row)
    for (std::size_t column = 0; column < matrix.dimension; ++column)
      residual[row] -= matrix.values[row * matrix.dimension + column] * x[column];
  return vibeqc::response::stable_norm(residual);
}


struct ContractResidentBackend final : vibeqc::response::ResidentKrylovBackend {
  ContractResidentBackend(std::size_t dimension, std::size_t vector_slots,
                          std::size_t resident_bytes)
      : n(dimension), slots(vector_slots), bytes(resident_bytes) {}

  std::size_t n{}, slots{}, bytes{};

  [[nodiscard]] std::size_t dimension() const noexcept override { return n; }
  [[nodiscard]] std::size_t vector_slots() const noexcept override { return slots; }
  [[nodiscard]] std::size_t owned_resident_bytes() const noexcept override { return bytes; }

  void upload(std::size_t, std::span<const double>) override {}
  void download(std::size_t, std::span<double>) override {}
  void zero(std::size_t) override {}
  void copy(std::size_t, std::size_t) override {}
  void scale(std::size_t, double) override {}
  void axpy(std::size_t, double, std::size_t) override {}
  [[nodiscard]] double dot(std::size_t, std::size_t) override { return 0.0; }
  [[nodiscard]] double norm(std::size_t) override { return 0.0; }
  void apply(std::size_t, std::size_t) override {}
};

void resident_krylov_contract() {
  GmresOptions options;
  options.restart = 7;
  options.max_iterations = 20;
  const auto plan = vibeqc::response::prepare_gmres(11, options);
  const auto workspace = vibeqc::response::resident_gmres_workspace(plan);
  require(workspace.vector_slots == 24, "resident GMRES vector-slot inventory is wrong");
  require(workspace.host_scalar_bytes == (7 * 7 + 5 * 7 + 1) * sizeof(double),
          "resident GMRES host-scalar workspace is wrong");

  ContractResidentBackend backend{11, workspace.vector_slots, 4096};
  const auto admitted = vibeqc::response::validate_resident_gmres_backend(plan, backend);
  require(admitted.vector_slots == workspace.vector_slots,
          "resident GMRES backend validation changed its workspace");

  bool rejected = false;
  try {
    ContractResidentBackend wrong_dimension{10, workspace.vector_slots, 4096};
    (void)vibeqc::response::validate_resident_gmres_backend(plan, wrong_dimension);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "resident GMRES accepted a backend from another dimension");

  rejected = false;
  try {
    ContractResidentBackend short_backend{11, workspace.vector_slots - 1, 4096};
    (void)vibeqc::response::validate_resident_gmres_backend(plan, short_backend);
  } catch (const std::length_error&) {
    rejected = true;
  }
  require(rejected, "resident GMRES accepted insufficient vector slots");

  rejected = false;
  try {
    ContractResidentBackend unowned{11, workspace.vector_slots, 0};
    (void)vibeqc::response::validate_resident_gmres_backend(plan, unowned);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "resident GMRES accepted unowned resident storage");
}

void exact_solve_and_true_residual() {
  DenseOperator matrix{2, {4.0, 1.0, 2.0, 3.0}};
  const std::array<double, 2> rhs{1.0, 2.0};
  GmresOptions options;
  options.relative_tolerance = 1e-13;
  options.restart = 2;
  options.max_iterations = 4;
  const auto plan = vibeqc::response::prepare_gmres(2, options);
  const auto result = vibeqc::response::solve_gmres(
      plan, [&](auto input, auto output) { matrix(input, output); }, rhs);
  require(result.status == GmresStatus::converged, "2x2 solve did not converge");
  require(std::abs(result.solution[0] - 0.1) < 1e-12 && std::abs(result.solution[1] - 0.6) < 1e-12,
          "2x2 solution is wrong");
  const auto residual = explicit_residual(matrix, result.solution, rhs);
  require(std::abs(result.residual_norm - residual) < 1e-15,
          "reported residual is not the recomputed true residual");
  require(result.relative_residual < 1e-12 && result.operator_actions == matrix.actions,
          "2x2 diagnostics are inconsistent");
}

void linear_response_problem_contract() {
  DenseOperator matrix{2, {4.0, 1.0, 2.0, 3.0}};
  const std::array<double, 2> rhs{1.0, 2.0};
  vibeqc::response::LinearResponseProblem problem(
      2, [&](auto input, auto output) { matrix(input, output); });
  auto options = GmresOptions{};
  options.relative_tolerance = 1e-13;
  options.restart = 2;
  options.max_iterations = 4;
  const auto plan = vibeqc::response::prepare_response(problem, options);
  const auto result = vibeqc::response::solve_response(plan, problem, rhs);
  require(result.converged(), "generic response problem did not converge");
  require(std::abs(result.solution[0] - 0.1) < 1e-12 && std::abs(result.solution[1] - 0.6) < 1e-12,
          "generic response problem returned the wrong solution");

  bool rejected = false;
  try {
    vibeqc::response::LinearResponseProblem invalid(0, [](auto, auto) {});
    (void)invalid;
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "zero-dimensional response problem was accepted");

  rejected = false;
  try {
    vibeqc::response::LinearResponseProblem invalid(2, {});
    (void)invalid;
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "response problem accepted an empty operator");

  rejected = false;
  try {
    const auto wrong_plan = vibeqc::response::prepare_gmres(1, options);
    (void)vibeqc::response::solve_response(wrong_plan, problem, rhs);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "response solve accepted a plan from another problem dimension");
}

void zero_rhs_is_transactional() {
  DenseOperator identity{2, {1.0, 0.0, 0.0, 1.0}};
  const std::array<double, 2> rhs{};
  const auto plan = vibeqc::response::prepare_gmres(2, {});
  const auto result = vibeqc::response::solve_gmres(
      plan, [&](auto input, auto output) { identity(input, output); }, rhs);
  require(result.status == GmresStatus::initial_residual && result.converged(),
          "zero RHS did not use the initial-residual path");
  require(result.iterations == 0 && result.operator_actions == 0 && identity.actions == 0,
          "zero RHS applied the operator");
  require(result.residual_norm == 0.0 && result.relative_residual == 0.0,
          "zero RHS residual convention is wrong");
}

void restarted_and_exhausted_paths() {
  DenseOperator diagonal{3, {1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 5.0}};
  const std::array<double, 3> rhs{1.0, -2.0, 3.0};
  GmresOptions restarted;
  restarted.relative_tolerance = 1e-11;
  restarted.restart = 1;
  restarted.max_iterations = 80;
  const auto converged = vibeqc::response::solve_gmres(
      vibeqc::response::prepare_gmres(3, restarted),
      [&](auto input, auto output) { diagonal(input, output); }, rhs);
  require(converged.status == GmresStatus::converged && converged.restarts > 0,
          "restart-one solve did not exercise a restart");
  require(explicit_residual(diagonal, converged.solution, rhs) < 1e-10,
          "restarted solution has a large true residual");

  DenseOperator limited{3, diagonal.values};
  GmresOptions exhausted = restarted;
  exhausted.relative_tolerance = 1e-15;
  exhausted.max_iterations = 1;
  const auto failed = vibeqc::response::solve_gmres(
      vibeqc::response::prepare_gmres(3, exhausted),
      [&](auto input, auto output) { limited(input, output); }, rhs);
  require(
      failed.status == GmresStatus::max_iterations && !failed.converged() && failed.iterations == 1,
      "iteration exhaustion claimed success or returned the wrong status");
  require(std::abs(failed.residual_norm - explicit_residual(limited, failed.solution, rhs)) < 1e-14,
          "exhausted solve did not publish its true residual");
}

void breakdown_and_nonfinite_paths() {
  DenseOperator zero{2, {0.0, 0.0, 0.0, 0.0}};
  const std::array<double, 2> rhs{1.0, 1.0};
  const auto broken = vibeqc::response::solve_gmres(
      vibeqc::response::prepare_gmres(2, {}), [&](auto input, auto output) { zero(input, output); },
      rhs);
  require(broken.status == GmresStatus::breakdown && !broken.converged(),
          "singular operator did not report breakdown");
  require(std::abs(broken.relative_residual - 1.0) < 1e-15,
          "breakdown residual diagnostic is wrong");

  DenseOperator identity{2, {1.0, 0.0, 0.0, 1.0}};
  const std::array<double, 2> bad_rhs{1.0, std::numeric_limits<double>::infinity()};
  const auto invalid = vibeqc::response::solve_gmres(
      vibeqc::response::prepare_gmres(2, {}),
      [&](auto input, auto output) { identity(input, output); }, bad_rhs);
  require(invalid.status == GmresStatus::nonfinite_input && identity.actions == 0,
          "nonfinite RHS was not rejected before the operator");

  DenseOperator bad_operator{2, {1.0, 0.0, 0.0, 1.0}};
  bad_operator.emit_nonfinite = true;
  const auto emitted = vibeqc::response::solve_gmres(
      vibeqc::response::prepare_gmres(2, {}),
      [&](auto input, auto output) { bad_operator(input, output); }, rhs);
  require(emitted.status == GmresStatus::nonfinite_operator && !emitted.converged(),
          "nonfinite operator output was not diagnosed");
}

void options_and_workspace_boundaries() {
  bool rejected = false;
  try {
    auto options = GmresOptions{};
    options.restart = 0;
    (void)vibeqc::response::prepare_gmres(2, options);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "zero restart was accepted");
  rejected = false;
  try {
    auto options = GmresOptions{};
    options.relative_tolerance = std::numeric_limits<double>::quiet_NaN();
    (void)vibeqc::response::prepare_gmres(2, options);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "nonfinite tolerance was accepted");
  rejected = false;
  try {
    auto options = GmresOptions{};
    options.restart = std::numeric_limits<std::size_t>::max();
    options.max_iterations = std::numeric_limits<std::size_t>::max();
    options.max_workspace_bytes = std::numeric_limits<std::size_t>::max();
    (void)vibeqc::response::prepare_gmres(std::numeric_limits<std::size_t>::max(), options);
  } catch (const std::overflow_error&) {
    rejected = true;
  }
  require(rejected, "GMRES workspace control overflow was accepted");

  GmresOptions probe_options;
  probe_options.restart = 2;
  probe_options.max_iterations = 4;
  const auto probe = vibeqc::response::prepare_gmres(2, probe_options);
  require(probe.workspace_bytes > 1, "workspace plan is empty");
  const std::array<double, 2> rhs{1.0, 2.0};
  DenseOperator exact_matrix{2, {2.0, 0.0, 0.0, 3.0}};
  auto exact_options = probe_options;
  exact_options.max_workspace_bytes = probe.workspace_bytes;
  const auto exact = vibeqc::response::solve_gmres(
      vibeqc::response::prepare_gmres(2, exact_options),
      [&](auto input, auto output) { exact_matrix(input, output); }, rhs);
  require(exact.converged(), "exact workspace budget was rejected");

  DenseOperator short_matrix{2, exact_matrix.values};
  auto short_options = exact_options;
  short_options.max_workspace_bytes = probe.workspace_bytes - 1;
  const auto short_result = vibeqc::response::solve_gmres(
      vibeqc::response::prepare_gmres(2, short_options),
      [&](auto input, auto output) { short_matrix(input, output); }, rhs);
  require(short_result.status == GmresStatus::workspace_limit &&
              short_result.operator_actions == 0 && short_matrix.actions == 0,
          "one-byte-short workspace was not rejected before the operator");
  require(short_result.workspace_bytes == probe.workspace_bytes,
          "workspace rejection lost the exact requirement");
  require(short_result.solution.empty(), "workspace refusal allocated a dimension-sized solution");
}

void measured_workspace_owns_returned_storage() {
  const std::array<double, 2> rhs{1.0, 2.0};
  const auto plan = vibeqc::response::prepare_gmres(2, {});
  std::shared_ptr<vibeqc::runtime::AllocationCounter> counter;
  {
    const auto result = vibeqc::response::solve_gmres(
        plan,
        [](auto in, auto out) {
          out[0] = 2.0 * in[0];
          out[1] = 3.0 * in[1];
        },
        rhs);
    counter = result.solution.get_allocator().counter();
    require(counter != nullptr && result.converged(), "successful response is unmeasured");
    const auto measured = counter->snapshot();
    require(measured.live_bytes == result.solution.capacity() * sizeof(double),
            "completed solve retained scratch or lost its output accounting");
    require(measured.peak_bytes == result.measured_workspace_peak_bytes &&
                measured.allocation_count == result.workspace_allocation_count &&
                result.measured_workspace_peak_bytes > measured.live_bytes &&
                result.measured_workspace_peak_bytes < plan.workspace_bytes,
            "response measurement is not actual allocation high-water usage");
  }
  require(counter->snapshot().live_bytes == 0, "returned response leaked its live allocation");

  const auto zero = vibeqc::response::solve_gmres(plan, [](auto, auto) {}, std::array<double, 2>{});
  require(zero.measured_workspace_peak_bytes == 3 * 2 * sizeof(double) &&
              zero.workspace_allocation_count == 3,
          "zero-RHS measurement includes unused Krylov workspace or substitutes its plan");

  auto refused_plan = plan;
  refused_plan.options.max_workspace_bytes = plan.workspace_bytes - 1;
  const auto refused = vibeqc::response::solve_gmres(refused_plan, [](auto, auto) {}, rhs);
  require(refused.measured_workspace_peak_bytes == 0 && refused.workspace_allocation_count == 0 &&
              !refused.solution.get_allocator().counter(),
          "workspace refusal allocated instrumentation or manufactured a measurement");

  bool threw = false;
  try {
    (void)vibeqc::response::solve_gmres(
        plan, [](auto, auto) { throw std::runtime_error("injected response operator failure"); },
        rhs);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  require(threw, "operator exception was swallowed by instrumentation");
  const auto retry = vibeqc::response::solve_gmres(
      plan, [](auto in, auto out) { std::copy(in.begin(), in.end(), out.begin()); }, rhs);
  require(retry.converged() && retry.measured_workspace_peak_bytes > 0,
          "failed response contaminated a later allocation domain");
}

void modified_plans_are_rejected_before_execution() {
  const std::array<double, 2> rhs{1.0, 2.0};
  const auto original = vibeqc::response::prepare_gmres(2, {});
  unsigned rejected = 0;
  std::size_t actions = 0;
  for (unsigned test = 0; test < 6; ++test) {
    auto plan = original;
    if (test == 0) {
      plan.workspace_bytes = 0;
      plan.options.max_workspace_bytes = 1;
    }
    if (test == 1) plan.restart = 0;
    if (test == 2) ++plan.dimension;
    if (test == 3) plan.options.relative_tolerance = std::numeric_limits<double>::infinity();
    if (test == 4) plan.options.restart = 0;
    if (test == 5) plan.options.max_iterations = 0;
    try {
      (void)vibeqc::response::solve_gmres(
          plan,
          [&](auto input, auto output) {
            ++actions;
            std::copy(input.begin(), input.end(), output.begin());
          },
          rhs);
      std::cerr << "Modified GMRES plan accepted: " << test << '\n';
    } catch (const std::invalid_argument&) {
      ++rejected;
    }
  }
  require(rejected == 6, "mutated plan bypassed validation or the declared workspace budget");
  require(actions == 0, "mutated plan reached its operator");
}

void response_batch_contract() {
  DenseOperator matrix{2, {2.0, 0.0, 0.0, 3.0}};
  vibeqc::response::LinearResponseProblem problem(
      2, [&](auto input, auto output) { matrix(input, output); });
  GmresOptions options;
  options.relative_tolerance = 1e-13;
  options.restart = 2;
  options.max_iterations = 4;

  const std::array<double, 2> rhs_a{2.0, 3.0};
  const std::array<double, 1> short_rhs{1.0};
  const std::array<double, 2> rhs_b{4.0, -6.0};
  const std::array<double, 2> diagonal{2.0, 3.0};
  const std::array<vibeqc::response::ResponseSolveRequest, 3> requests{
      vibeqc::response::ResponseSolveRequest{rhs_a},
      vibeqc::response::ResponseSolveRequest{short_rhs},
      vibeqc::response::ResponseSolveRequest{rhs_b, {}, diagonal},
  };

  const auto probe = vibeqc::response::prepare_response_batch(
      problem, options, requests.size(), std::numeric_limits<std::size_t>::max());
  const auto exact = vibeqc::response::prepare_response_batch(problem, options, requests.size(),
                                                              probe.peak_numeric_bytes);
  require(exact.admitted, "exact response batch numeric budget was rejected");
  require(exact.retained_solution_bytes == requests.size() * 2 * sizeof(double),
          "response batch retained-solution accounting is wrong");
  require(exact.peak_numeric_bytes ==
              exact.solve_plan.workspace_bytes + (requests.size() - 1) * 2 * sizeof(double),
          "response batch peak does not include prior live solutions");

  const auto batch = vibeqc::response::solve_response_batch(exact, problem, requests);
  require(batch.complete() && batch.responses.size() == requests.size(),
          "response batch did not complete every request");
  require(batch.responses[0].converged() &&
              batch.responses[1].status == GmresStatus::nonfinite_input &&
              batch.responses[2].converged(),
          "one malformed RHS contaminated an independent response request");
  require(std::abs(batch.responses[0].solution[0] - 1.0) < 1e-12 &&
              std::abs(batch.responses[0].solution[1] - 1.0) < 1e-12 &&
              std::abs(batch.responses[2].solution[0] - 2.0) < 1e-12 &&
              std::abs(batch.responses[2].solution[1] + 2.0) < 1e-12,
          "response batch returned a wrong solution");
  require(batch.planned_peak_numeric_bytes == exact.peak_numeric_bytes &&
              batch.retained_solution_bytes == exact.retained_solution_bytes,
          "response batch result lost its numeric-payload accounting");
  require(matrix.actions == batch.responses[0].operator_actions +
                                batch.responses[1].operator_actions +
                                batch.responses[2].operator_actions,
          "response batch operator-action diagnostics are inconsistent");

  DenseOperator blocked_matrix{2, matrix.values};
  vibeqc::response::LinearResponseProblem blocked_problem(
      2, [&](auto input, auto output) { blocked_matrix(input, output); });
  const auto blocked_plan = vibeqc::response::prepare_response_batch(
      blocked_problem, options, requests.size(), probe.peak_numeric_bytes - 1);
  require(!blocked_plan.admitted, "one-byte-short response batch budget was admitted");
  const auto blocked =
      vibeqc::response::solve_response_batch(blocked_plan, blocked_problem, requests);
  require(blocked.status == vibeqc::response::ResponseBatchStatus::workspace_limit &&
              blocked.responses.empty() && blocked_matrix.actions == 0,
          "response batch budget refusal invoked an operator or allocated response results");

  const auto empty_plan = vibeqc::response::prepare_response_batch(problem, options, 0, 0);
  const std::span<const vibeqc::response::ResponseSolveRequest> empty_requests;
  const auto empty = vibeqc::response::solve_response_batch(empty_plan, problem, empty_requests);
  require(empty_plan.admitted && empty_plan.peak_numeric_bytes == 0 && empty.complete() &&
              empty.responses.empty(),
          "empty response batch did not remain a zero-work request");

  bool rejected = false;
  auto modified = exact;
  --modified.peak_numeric_bytes;
  const auto actions_before = matrix.actions;
  try {
    (void)vibeqc::response::solve_response_batch(modified, problem, requests);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected && matrix.actions == actions_before,
          "mutated response batch plan reached its operator");
}

void stable_norm_extremes() {
  const std::array<double, 2> tiny{1e-200, 0.0};
  const std::array<double, 2> large{1e200, 0.0};
  require(vibeqc::response::stable_norm(tiny) == 1e-200, "stable norm underflowed a finite vector");
  require(vibeqc::response::stable_norm(large) == 1e200, "stable norm overflowed a finite vector");
  bool rejected = false;
  try {
    const std::array<double, 2> impossible{std::numeric_limits<double>::max(),
                                           std::numeric_limits<double>::max()};
    (void)vibeqc::response::stable_norm(impossible);
  } catch (const std::overflow_error&) {
    rejected = true;
  }
  require(rejected, "unrepresentable finite norm was accepted");
}
}  // namespace

int main() {
  try {
    modified_plans_are_rejected_before_execution();
    resident_krylov_contract();
    exact_solve_and_true_residual();
    linear_response_problem_contract();
    zero_rhs_is_transactional();
    restarted_and_exhausted_paths();
    breakdown_and_nonfinite_paths();
    options_and_workspace_boundaries();
    response_batch_contract();
    stable_norm_extremes();
    measured_workspace_owns_returned_storage();
    std::cout << "Native GMRES contracts passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
