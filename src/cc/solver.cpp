#include "cc/solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "generated_rccsd_cpu.hpp"
#include "solver/iteration_control.hpp"

namespace vibeqc::cc {
namespace {

std::size_t checked_add(std::size_t a, std::size_t b) { return generated::checked_add(a, b); }
std::size_t checked_mul(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("RCCSD size overflow");
  return a * b;
}
std::size_t bytes(std::size_t elements) { return checked_mul(elements, sizeof(double)); }

generated::Inputs inputs(const Problem& p, const double* t1, const double* t2) {
  return {p.foo.data(),
          p.fov.data(),
          p.fvv.data(),
          p.ovov.data(),
          p.ovvo.data(),
          p.oovv.data(),
          p.ovvv.data(),
          p.ovoo.data(),
          p.oooo.data(),
          p.vvvv.data(),
          p.d1.data(),
          p.d2.data(),
          t1,
          t2};
}

double max_abs(const double* p, std::size_t n) {
  double result = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(p[i])) throw std::runtime_error("nonfinite RCCSD physical residual");
    result = std::max(result, std::abs(p[i]));
  }
  return result;
}

bool solve_linear(std::vector<double> matrix, std::vector<double> rhs, std::vector<double>& x) {
  const auto n = rhs.size();
  for (std::size_t col = 0; col < n; ++col) {
    auto pivot = col;
    for (std::size_t row = col + 1; row < n; ++row)
      if (std::abs(matrix[row * n + col]) > std::abs(matrix[pivot * n + col])) pivot = row;
    const double divisor = matrix[pivot * n + col];
    if (!std::isfinite(divisor) || std::abs(divisor) < 1e-14) return false;
    if (pivot != col) {
      for (std::size_t j = 0; j < n; ++j) std::swap(matrix[col * n + j], matrix[pivot * n + j]);
      std::swap(rhs[col], rhs[pivot]);
    }
    for (std::size_t j = col; j < n; ++j) matrix[col * n + j] /= divisor;
    rhs[col] /= divisor;
    for (std::size_t row = 0; row < n; ++row) {
      if (row == col) continue;
      const double factor = matrix[row * n + col];
      for (std::size_t j = col; j < n; ++j) matrix[row * n + j] -= factor * matrix[col * n + j];
      rhs[row] -= factor * rhs[col];
    }
  }
  x = std::move(rhs);
  return std::all_of(x.begin(), x.end(), [](double y) { return std::isfinite(y); });
}

struct Diis {
  unsigned capacity{}, restarts{};
  std::size_t elements{};
  std::vector<std::vector<double>> vectors, errors;

  std::vector<double> update(std::vector<double> vector, std::vector<double> error) {
    if (!capacity) return vector;
    vectors.push_back(vector);
    errors.push_back(std::move(error));
    if (vectors.size() > capacity) {
      vectors.erase(vectors.begin());
      errors.erase(errors.begin());
    }
    while (vectors.size() > 1) {
      const auto n = vectors.size();
      std::vector<double> gram(n * n);
      double scale = 0.0;
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
          gram[i * n + j] =
              std::inner_product(errors[i].begin(), errors[i].end(), errors[j].begin(), 0.0);
          scale = std::max(scale, std::abs(gram[i * n + j]));
        }
      if (scale == 0.0) return vector;
      std::vector<double> system((n + 1) * (n + 1)), rhs(n + 1), solution;
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) system[i * (n + 1) + j] = gram[i * n + j] / scale;
      for (std::size_t i = 0; i < n; ++i) system[i * (n + 1) + n] = system[n * (n + 1) + i] = -1.0;
      rhs[n] = -1.0;
      if (solve_linear(system, rhs, solution)) {
        solution.resize(n);
        if (std::all_of(solution.begin(), solution.end(),
                        [](double c) { return std::isfinite(c) && std::abs(c) <= 1e6; })) {
          std::vector<double> result(elements);
          for (std::size_t row = 0; row < n; ++row)
            for (std::size_t i = 0; i < elements; ++i) result[i] += solution[row] * vectors[row][i];
          return result;
        }
      }
      vectors.erase(vectors.begin());
      errors.erase(errors.begin());
      ++restarts;
    }
    return vector;
  }
};

}  // namespace

void validate_problem(const Problem& p) {
  if (!p.nocc || !p.nvir)
    throw std::invalid_argument("RCCSD requires occupied and virtual orbitals");
  const auto o = p.nocc, v = p.nvir;
  auto expect = [](const std::vector<double>& x, std::size_t n, const char* name) {
    if (x.size() != n ||
        !std::all_of(x.begin(), x.end(), [](double y) { return std::isfinite(y); }))
      throw std::invalid_argument(std::string("invalid RCCSD input ") + name);
  };
  const auto ov = checked_mul(o, v), oo = checked_mul(o, o), vv = checked_mul(v, v);
  const auto oovv = checked_mul(oo, vv);
  expect(p.foo, oo, "foo");
  expect(p.fov, ov, "fov");
  expect(p.fvv, vv, "fvv");
  expect(p.ovov, oovv, "ovov");
  expect(p.ovvo, oovv, "ovvo");
  expect(p.oovv, oovv, "oovv");
  expect(p.ovvv, checked_mul(o, checked_mul(vv, v)), "ovvv");
  expect(p.ovoo, checked_mul(ov, oo), "ovoo");
  expect(p.oooo, checked_mul(oo, oo), "oooo");
  expect(p.vvvv, checked_mul(vv, vv), "vvvv");
  expect(p.d1, ov, "d1");
  expect(p.d2, oovv, "d2");
  expect(p.initial_t1, ov, "initial_t1");
  expect(p.initial_t2, oovv, "initial_t2");
  if (!std::isfinite(p.reference_energy))
    throw std::invalid_argument("nonfinite RCCSD reference energy");
}

void validate_options(const SolverOptions& o) {
  if (!o.max_iterations || o.diis_size == 1 || o.diis_size > 20 || !o.max_bytes)
    throw std::invalid_argument("invalid RCCSD iteration/history/budget option");
  if (!(o.energy_tolerance > 0.0 && o.energy_tolerance <= 1e-8) ||
      !(o.residual_tolerance > 0.0 && o.residual_tolerance <= 1e-9) ||
      !(o.denominator_threshold > 0.0) || !(o.damping >= 0.0 && o.damping < 1.0) ||
      !(o.level_shift >= 0.0) || !std::isfinite(o.energy_tolerance) ||
      !std::isfinite(o.residual_tolerance) || !std::isfinite(o.denominator_threshold) ||
      !std::isfinite(o.damping) || !std::isfinite(o.level_shift))
    throw std::invalid_argument("invalid RCCSD numeric option");
}

std::size_t problem_host_bytes(const Problem& p) {
  std::size_t result = 0;
  const std::vector<double>* values[] = {&p.foo,  &p.fov,  &p.fvv,        &p.ovov,      &p.ovvo,
                                         &p.oovv, &p.ovvv, &p.ovoo,       &p.oooo,      &p.vvvv,
                                         &p.d1,   &p.d2,   &p.initial_t1, &p.initial_t2};
  for (const auto* value : values) result = checked_add(result, bytes(value->capacity()));
  return result;
}

SolverResult solve_cpu(const Problem& p, const SolverOptions& options) {
  validate_problem(p);
  validate_options(options);
  const auto n1 = checked_mul(p.nocc, p.nvir);
  const auto n2 = checked_mul(checked_mul(p.nocc, p.nocc), checked_mul(p.nvir, p.nvir));
  const auto elements = checked_add(n1, n2);
  const auto iteration_elements = generated::iteration_arena_elements(p.nocc, p.nvir);
  const auto replay_elements = generated::replay_arena_elements(p.nocc, p.nvir);
  std::size_t capacity = checked_add(p.reference_retained_bytes, problem_host_bytes(p));
  capacity = checked_add(capacity, bytes(iteration_elements));
  capacity = checked_add(capacity, bytes(replay_elements));
  // Current, trial, error and a copied history vector coexist before trimming.
  // DIIS additionally retains Gram/original augmented arrays while solve_linear
  // owns its by-value matrix/RHS copies. These are numeric storage, not overhead.
  capacity = checked_add(capacity, bytes(checked_mul(4 + 2 * options.diis_size, elements)));
  if (options.diis_size) {
    const std::size_t h = options.diis_size, n = h + 1;
    const auto scratch = checked_add(
        checked_mul(h, h), checked_add(checked_mul(2, checked_mul(n, n)), checked_mul(2, n)));
    capacity = checked_add(capacity, bytes(scratch));
  }
  if (capacity > options.max_bytes)
    throw std::length_error("RCCSD CPU solve exceeds correlation memory budget");

  std::vector<double> iteration_arena(iteration_elements), replay_arena(replay_elements);
  std::vector<double> current;
  current.reserve(elements);
  current.insert(current.end(), p.initial_t1.begin(), p.initial_t1.end());
  current.insert(current.end(), p.initial_t2.begin(), p.initial_t2.end());
  Diis diis{options.diis_size, 0, elements, {}, {}};
  double previous = std::numeric_limits<double>::quiet_NaN();
  SolverResult result;
  result.diagnostic.numeric_capacity_bytes = std::max(p.provider_peak_bytes, capacity);
  result.reason = "maximum RCCSD iterations reached";

  const unsigned iteration_budget = options.max_iterations == std::numeric_limits<unsigned>::max()
                                        ? options.max_iterations
                                        : options.max_iterations + 1;
  vibeqc::solver::run_bounded_iterations(iteration_budget, [&](unsigned ordinal) {
    const unsigned iteration = ordinal - 1;
    try {
      auto in = inputs(p, current.data(), current.data() + n1);
      const auto out = generated::run_iteration_cpu(p.nocc, p.nvir, in, iteration_arena.data(),
                                                    iteration_arena.size());
      const double r1 = max_abs(out.r1, n1), r2 = max_abs(out.r2, n2);
      const double delta = std::isfinite(previous) ? std::abs(out.energy - previous)
                                                   : std::numeric_limits<double>::infinity();
      result.correlation_energy = out.energy;
      result.total_energy = p.reference_energy + out.energy;
      result.diagnostic.iterations = iteration + 1;
      result.diagnostic.energy_change = delta;
      result.diagnostic.r1_max = r1;
      result.diagnostic.r2_max = r2;
      if (std::isfinite(previous) && delta <= options.energy_tolerance &&
          std::max(r1, r2) <= options.residual_tolerance) {
        const auto replay =
            generated::run_replay_cpu(p.nocc, p.nvir, in, replay_arena.data(), replay_arena.size());
        result.diagnostic.replay_r1_max = max_abs(replay.r1, n1);
        result.diagnostic.replay_r2_max = max_abs(replay.r2, n2);
        if (std::max(result.diagnostic.replay_r1_max, result.diagnostic.replay_r2_max) <=
                options.residual_tolerance &&
            std::abs(replay.energy - out.energy) <= options.energy_tolerance) {
          result.status = SolveStatus::Converged;
          result.reason = "energy change and expanded physical R1/R2 passed";
          return false;
        }
      }
      if (iteration == options.max_iterations) return false;
      std::vector<double> trial(elements);
      const double jacobi = 1.0 - options.damping;
      for (std::size_t k = 0; k < n1; ++k)
        trial[k] = current[k] + jacobi * (out.next_t1[k] - current[k]);
      for (std::size_t k = 0; k < n2; ++k)
        trial[n1 + k] = current[n1 + k] + jacobi * (out.next_t2[k] - current[n1 + k]);
      auto trial_in = inputs(p, trial.data(), trial.data() + n1);
      const auto trial_out = generated::run_iteration_cpu(
          p.nocc, p.nvir, trial_in, iteration_arena.data(), iteration_arena.size());
      std::vector<double> error;
      error.reserve(elements);
      error.insert(error.end(), trial_out.r1, trial_out.r1 + n1);
      error.insert(error.end(), trial_out.r2, trial_out.r2 + n2);
      current = diis.update(std::move(trial), std::move(error));
      previous = out.energy;
      return true;
    } catch (const std::runtime_error& error) {
      result.status = SolveStatus::NumericalFailure;
      result.reason = error.what();
      return false;
    }
  });
  result.diagnostic.diis_restarts = diis.restarts;
  result.t1.assign(current.begin(), current.begin() + static_cast<std::ptrdiff_t>(n1));
  result.t2.assign(current.begin() + static_cast<std::ptrdiff_t>(n1), current.end());
  return result;
}

#if !VIBEQC_HAS_CUDA
SolverResult solve_cuda(const Problem&, const SolverOptions&, int) {
  throw std::runtime_error("CUDA RCCSD is not compiled");
}
#endif

}  // namespace vibeqc::cc
