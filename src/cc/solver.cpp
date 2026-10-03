#include "cc/solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "generated_df_ccsd_core_cpu.hpp"
#include "generated_rccsd_cpu.hpp"
#include "solver/diis.hpp"
#include "solver/iteration_control.hpp"

namespace generativeqc::cc {
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

}  // namespace

void validate_problem(const Problem& p, bool allow_df_virtual) {
  if (p.naux && !allow_df_virtual)
    throw std::invalid_argument("DF virtual inputs require a factorized execution owner");
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
  if (p.naux) {
    expect(p.ovvv, 0, "ovvv must be empty for DF");
    expect(p.vvvv, 0, "vvvv must be empty for DF");
    expect(p.df_bov, checked_mul(p.naux, ov), "df_bov");
    expect(p.df_bvv, checked_mul(p.naux, vv), "df_bvv");
    for (std::size_t q = 0; q < p.naux; ++q)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < a; ++b)
          if (std::abs(p.df_bvv[q * vv + a * v + b] - p.df_bvv[q * vv + b * v + a]) > 1e-10)
            throw std::invalid_argument("DF B_vv must preserve symmetric spatial-MO pairs");
  } else {
    expect(p.df_bov, 0, "df_bov requires naux");
    expect(p.df_bvv, 0, "df_bvv requires naux");
    expect(p.ovvv, checked_mul(o, checked_mul(vv, v)), "ovvv");
    expect(p.vvvv, checked_mul(vv, vv), "vvvv");
  }
  expect(p.ovoo, checked_mul(ov, oo), "ovoo");
  expect(p.oooo, checked_mul(oo, oo), "oooo");
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
  const std::vector<double>* values[] = {
      &p.foo,  &p.fov,  &p.fvv, &p.ovov, &p.ovvo,       &p.oovv,       &p.ovvv,   &p.ovoo,
      &p.oooo, &p.vvvv, &p.d1,  &p.d2,   &p.initial_t1, &p.initial_t2, &p.df_bov, &p.df_bvv};
  for (const auto* value : values) result = checked_add(result, bytes(value->capacity()));
  return result;
}

SolverResult solve_cpu(const Problem& p, const SolverOptions& options) {
  validate_problem(p, true);
  validate_options(options);
  const auto n1 = checked_mul(p.nocc, p.nvir);
  const auto n2 = checked_mul(checked_mul(p.nocc, p.nocc), checked_mul(p.nvir, p.nvir));
  const auto elements = checked_add(n1, n2);
  const auto iteration_elements = p.naux
                                      ? generated::dfcore::iteration_arena_elements(p.nocc, p.nvir)
                                      : generated::iteration_arena_elements(p.nocc, p.nvir);
  const auto replay_elements = p.naux ? generated::dfcore::replay_arena_elements(p.nocc, p.nvir)
                                      : generated::replay_arena_elements(p.nocc, p.nvir);
  const auto virtual_elements =
      p.naux ? generated::df::virtual_cpu_arena_elements(p.nocc, p.nvir) : 0;
  std::size_t capacity = checked_add(p.reference_retained_bytes, problem_host_bytes(p));
  capacity = checked_add(capacity, bytes(iteration_elements));
  capacity = checked_add(capacity, bytes(replay_elements));
  capacity = checked_add(capacity, bytes(virtual_elements));
  if (p.naux) capacity = checked_add(capacity, bytes(elements));
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
  std::vector<double> virtual_arena(virtual_elements), virtual_sum(p.naux ? elements : 0);
  std::vector<double> current;
  current.reserve(elements);
  current.insert(current.end(), p.initial_t1.begin(), p.initial_t1.end());
  current.insert(current.end(), p.initial_t2.begin(), p.initial_t2.end());
  generativeqc::solver::Diis diis(options.diis_size, elements);
  double previous = std::numeric_limits<double>::quiet_NaN();
  SolverResult result;
  result.diagnostic.numeric_capacity_bytes = std::max(p.provider_peak_bytes, capacity);
  result.reason = "maximum RCCSD iterations reached";
  const auto solve_started = std::chrono::steady_clock::now();

  // The same accumulation owner serves current/trial/replay amplitudes. Never
  // reuse a correction after DIIS changes T, or count only a single Q slice.
  auto df_inputs = [&](const generated::Inputs& in) {
    std::fill(virtual_sum.begin(), virtual_sum.end(), 0.0);
    generated::df::Inputs factors{};
    factors.t1 = in.t1;
    factors.t2 = in.t2;
    for (std::size_t q = 0; q < p.naux; ++q) {
      factors.bov = p.df_bov.data() + q * n1;
      factors.bvv = p.df_bvv.data() + q * p.nvir * p.nvir;
      const auto out = generated::df::run_virtual_cpu(p.nocc, p.nvir, factors, virtual_arena.data(),
                                                      virtual_arena.size());
      for (std::size_t k = 0; k < n1; ++k) virtual_sum[k] += out.singles[k];
      for (std::size_t k = 0; k < n2; ++k) virtual_sum[n1 + k] += out.doubles[k];
      ++result.diagnostic.df_auxiliary_slices;
      result.diagnostic.df_virtual_operations += generated::df::virtual_cpu_operation_count;
      result.diagnostic.df_accumulation_calls += 2;
    }
    return generated::dfcore::Inputs{in.foo,
                                     in.fov,
                                     in.fvv,
                                     in.ovov,
                                     in.ovvo,
                                     in.oovv,
                                     in.ovoo,
                                     in.oooo,
                                     in.d1,
                                     in.d2,
                                     in.t1,
                                     in.t2,
                                     virtual_sum.data(),
                                     virtual_sum.data() + n1};
  };
  auto run_iteration = [&](const generated::Inputs& in) -> generated::IterationOutputs {
    if (!p.naux)
      return generated::run_iteration_cpu(p.nocc, p.nvir, in, iteration_arena.data(),
                                          iteration_arena.size());
    const auto core = df_inputs(in);
    const auto out = generated::dfcore::run_iteration_cpu(
        p.nocc, p.nvir, core, iteration_arena.data(), iteration_arena.size());
    return {out.energy, out.r1, out.r2, out.next_t1, out.next_t2};
  };
  auto run_replay = [&](const generated::Inputs& in) -> generated::ReplayOutputs {
    if (!p.naux)
      return generated::run_replay_cpu(p.nocc, p.nvir, in, replay_arena.data(),
                                       replay_arena.size());
    const auto core = df_inputs(in);
    const auto out = generated::dfcore::run_replay_cpu(p.nocc, p.nvir, core, replay_arena.data(),
                                                       replay_arena.size());
    return {out.energy, out.r1, out.r2};
  };

  const unsigned iteration_budget = options.max_iterations == std::numeric_limits<unsigned>::max()
                                        ? options.max_iterations
                                        : options.max_iterations + 1;
  generativeqc::solver::run_bounded_iterations(iteration_budget, [&](unsigned ordinal) {
    const unsigned iteration = ordinal - 1;
    try {
      auto in = inputs(p, current.data(), current.data() + n1);
      const auto iteration_started = std::chrono::steady_clock::now();
      const auto out = run_iteration(in);
      result.diagnostic.iteration_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - iteration_started)
              .count();
      ++result.diagnostic.iteration_graph_calls;
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
        const auto replay_started = std::chrono::steady_clock::now();
        const auto replay = run_replay(in);
        result.diagnostic.replay_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - replay_started)
                .count();
        ++result.diagnostic.replay_graph_calls;
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
      const auto update_started = std::chrono::steady_clock::now();
      std::vector<double> trial(elements);
      const double jacobi = 1.0 - options.damping;
      for (std::size_t k = 0; k < n1; ++k)
        trial[k] = current[k] + jacobi * (out.next_t1[k] - current[k]);
      for (std::size_t k = 0; k < n2; ++k)
        trial[n1 + k] = current[n1 + k] + jacobi * (out.next_t2[k] - current[n1 + k]);
      result.diagnostic.update_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - update_started).count();
      ++result.diagnostic.update_calls;
      auto trial_in = inputs(p, trial.data(), trial.data() + n1);
      const auto trial_started = std::chrono::steady_clock::now();
      const auto trial_out = run_iteration(trial_in);
      result.diagnostic.iteration_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - trial_started).count();
      ++result.diagnostic.iteration_graph_calls;
      std::vector<double> error;
      error.reserve(elements);
      error.insert(error.end(), trial_out.r1, trial_out.r1 + n1);
      error.insert(error.end(), trial_out.r2, trial_out.r2 + n2);
      const auto diis_started = std::chrono::steady_clock::now();
      current = diis.update(std::move(trial), std::move(error));
      result.diagnostic.diis_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - diis_started).count();
      previous = out.energy;
      return true;
    } catch (const std::runtime_error& error) {
      result.status = SolveStatus::NumericalFailure;
      result.reason = error.what();
      return false;
    }
  });
  result.diagnostic.diis_restarts = diis.restarts();
  result.diagnostic.tensor_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_started).count();
  result.t1.assign(current.begin(), current.begin() + static_cast<std::ptrdiff_t>(n1));
  result.t2.assign(current.begin() + static_cast<std::ptrdiff_t>(n1), current.end());
  return result;
}

#if !GENERATIVEQC_HAS_CUDA
SolverResult solve_cuda(const Problem&, const SolverOptions&, int) {
  throw std::runtime_error("CUDA RCCSD is not compiled");
}
#endif

}  // namespace generativeqc::cc
