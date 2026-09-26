#include "cc/lambda_response.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>

#include "generated_rccsd_cpu.hpp"

namespace vibeqc::cc {
namespace {

std::size_t checked_add(std::size_t a, std::size_t b) { return generated::checked_add(a, b); }

std::size_t checked_mul(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("RCCSD Lambda size overflow");
  return a * b;
}

std::size_t bytes(std::size_t elements) { return checked_mul(elements, sizeof(double)); }

generated::Inputs inputs(const Problem& p, const SolverResult& cc) {
  return {p.foo.data(),  p.fov.data(),  p.fvv.data(),  p.ovov.data(), p.ovvo.data(),
          p.oovv.data(), p.ovvv.data(), p.ovoo.data(), p.oooo.data(), p.vvvv.data(),
          p.d1.data(),   p.d2.data(),   cc.t1.data(),  cc.t2.data()};
}

double max_abs(std::span<const double> values) {
  double result = 0.0;
  for (double value : values) {
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite RCCSD Lambda residual");
    result = std::max(result, std::abs(value));
  }
  return result;
}

struct AmplitudeLayout {
  std::size_t o{}, v{}, n1{}, n2{};
  std::vector<std::size_t> representatives;
  std::vector<std::size_t> partners;
  std::vector<double> sqrt_weights;

  explicit AmplitudeLayout(std::size_t occupied, std::size_t virtuals)
      : o(occupied),
        v(virtuals),
        n1(checked_mul(o, v)),
        n2(checked_mul(checked_mul(o, o), checked_mul(v, v))) {}

  [[nodiscard]] std::size_t pair_count() const { return checked_add(n2, n1) / 2; }
  [[nodiscard]] std::size_t dimension() const { return checked_add(n1, pair_count()); }
  [[nodiscard]] std::size_t storage_bytes() const {
    return checked_add(checked_mul(checked_mul(2, pair_count()), sizeof(std::size_t)),
                       bytes(dimension()));
  }

  void initialize() {
    sqrt_weights.assign(dimension(), 1.0);
    representatives.resize(pair_count());
    partners.resize(pair_count());
    std::size_t position = 0;
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t j = 0; j < o; ++j)
        for (std::size_t a = 0; a < v; ++a)
          for (std::size_t b = 0; b < v; ++b) {
            const auto flat = ((i * o + j) * v + a) * v + b;
            const auto mate = ((j * o + i) * v + b) * v + a;
            if (flat > mate) continue;
            representatives[position] = flat;
            partners[position] = mate;
            sqrt_weights[n1 + position] = flat == mate ? 1.0 : std::sqrt(2.0);
            ++position;
          }
  }

  void validate_dense(std::span<const double> one, std::span<const double> two) const {
    if (one.size() != n1 || two.size() != n2)
      throw std::invalid_argument("RCCSD Lambda amplitude shape mismatch");
    for (double value : one)
      if (!std::isfinite(value)) throw std::invalid_argument("nonfinite RCCSD Lambda singles");
    for (std::size_t k = 0; k < representatives.size(); ++k) {
      const double first = two[representatives[k]], second = two[partners[k]];
      if (!std::isfinite(first) || !std::isfinite(second) ||
          std::abs(first - second) > 1e-10 * (1.0 + std::max(std::abs(first), std::abs(second))))
        throw std::invalid_argument("RCCSD doubles violate simultaneous pair symmetry");
    }
  }

  void pack_weighted(std::span<const double> one, std::span<const double> two,
                     std::span<double> output) const {
    validate_dense(one, two);
    if (output.size() != dimension())
      throw std::invalid_argument("RCCSD Lambda packed output shape mismatch");
    std::copy(one.begin(), one.end(), output.begin());
    for (std::size_t k = 0; k < representatives.size(); ++k)
      output[n1 + k] = sqrt_weights[n1 + k] * two[representatives[k]];
  }

  void unpack_weighted(std::span<const double> packed, std::span<double> one,
                       std::span<double> two) const {
    if (packed.size() != dimension() || one.size() != n1 || two.size() != n2)
      throw std::invalid_argument("RCCSD Lambda unpack shape mismatch");
    std::fill(two.begin(), two.end(), 0.0);
    std::copy(packed.begin(), packed.begin() + static_cast<std::ptrdiff_t>(n1), one.begin());
    for (std::size_t k = 0; k < representatives.size(); ++k) {
      const double value = packed[n1 + k] / sqrt_weights[n1 + k];
      two[representatives[k]] = value;
      two[partners[k]] = value;
    }
  }

  void publish(std::span<const double> packed, std::vector<double>& one,
               std::vector<double>& two) const {
    one.resize(n1);
    two.resize(n2);
    unpack_weighted(packed, one, two);
  }
};

std::size_t vector_capacity_bytes(const std::vector<double>& values) {
  return bytes(values.capacity());
}

}  // namespace

void validate_lambda_options(const LambdaOptions& options) {
  if (!std::isfinite(options.cc_tolerance) || options.cc_tolerance <= 0.0 ||
      options.cc_tolerance > 1e-9 || !std::isfinite(options.lambda_tolerance) ||
      options.lambda_tolerance <= 0.0 || options.lambda_tolerance > 1e-9 || !options.max_bytes)
    throw std::invalid_argument("invalid RCCSD Lambda tolerance/budget");
  (void)response::prepare_gmres(1, options.gmres);
}

std::size_t lambda_cpu_numeric_capacity(const Problem& p, const SolverResult& cc,
                                        const LambdaOptions& options, bool with_energy_source) {
  validate_lambda_options(options);
  AmplitudeLayout layout(p.nocc, p.nvir);
  const auto replay_elements = generated::replay_arena_elements(p.nocc, p.nvir);
  const auto shared_rhs_elements = generated::lambda_rhs_arena_elements(p.nocc, p.nvir);
  const auto shared_jt_elements = generated::lambda_transpose_arena_elements(p.nocc, p.nvir);
  const auto independent_rhs_elements =
      generated::lambda_independent_rhs_arena_elements(p.nocc, p.nvir);
  const auto independent_jt_elements =
      generated::lambda_independent_transpose_arena_elements(p.nocc, p.nvir);
  const auto arena_elements = std::max({replay_elements, shared_rhs_elements, shared_jt_elements,
                                        independent_rhs_elements, independent_jt_elements});
  const auto gmres_plan = response::prepare_gmres(layout.dimension(), options.gmres);

  std::size_t capacity = checked_add(p.reference_retained_bytes, problem_host_bytes(p));
  capacity = checked_add(capacity, vector_capacity_bytes(cc.t1));
  capacity = checked_add(capacity, vector_capacity_bytes(cc.t2));
  capacity = checked_add(capacity, bytes(arena_elements));
  // Dense seeds, packed RHS/audit and published dense Lambda can coexist.
  capacity = checked_add(capacity, bytes(checked_mul(3, layout.n1 + layout.n2)));
  capacity = checked_add(capacity, bytes(checked_mul(3, layout.dimension())));
  capacity = checked_add(capacity, layout.storage_bytes());
  capacity = checked_add(capacity, gmres_plan.workspace_bytes);
  if (with_energy_source) capacity = checked_add(capacity, bytes(layout.dimension()));
  return capacity;
}

static LambdaResult solve_lambda_cpu_impl(const Problem& p, const SolverResult& cc,
                                          std::span<const double> t1_source,
                                          std::span<const double> t2_source,
                                          const LambdaOptions& options) {
  validate_problem(p);
  validate_lambda_options(options);
  if (!cc.converged()) throw std::invalid_argument("RCCSD Lambda requires a converged CC result");
  AmplitudeLayout layout(p.nocc, p.nvir);
  if (cc.t1.size() != layout.n1 || cc.t2.size() != layout.n2)
    throw std::invalid_argument("RCCSD Lambda CC result shape mismatch");
  if (!std::isfinite(cc.correlation_energy))
    throw std::invalid_argument("nonfinite RCCSD Lambda primal energy");
  const bool with_source = !t1_source.empty() || !t2_source.empty();
  if (with_source && (t1_source.size() != layout.n1 || t2_source.size() != layout.n2))
    throw std::invalid_argument("RCCSD Lambda energy-source shape mismatch");
  for (const auto source : {t1_source, t2_source})
    for (const double value : source)
      if (!std::isfinite(value))
        throw std::invalid_argument("nonfinite RCCSD Lambda energy source");
  const auto capacity = lambda_cpu_numeric_capacity(p, cc, options, with_source);
  if (capacity > options.max_bytes)
    throw std::length_error("RCCSD Lambda exceeds simultaneous host budget");

  const auto arena_elements =
      std::max({generated::replay_arena_elements(p.nocc, p.nvir),
                generated::lambda_rhs_arena_elements(p.nocc, p.nvir),
                generated::lambda_transpose_arena_elements(p.nocc, p.nvir),
                generated::lambda_independent_rhs_arena_elements(p.nocc, p.nvir),
                generated::lambda_independent_transpose_arena_elements(p.nocc, p.nvir)});
  const auto gmres_plan = response::prepare_gmres(layout.dimension(), options.gmres);
  const auto in = inputs(p, cc);

  layout.initialize();
  layout.validate_dense(cc.t1, cc.t2);

  std::vector<double> arena(arena_elements);
  const auto replay = generated::run_replay_cpu(p.nocc, p.nvir, in, arena.data(), arena.size());
  const double replay_r1 = max_abs({replay.r1, layout.n1});
  const double replay_r2 = max_abs({replay.r2, layout.n2});
  if (std::max(replay_r1, replay_r2) > options.cc_tolerance ||
      std::abs(replay.energy - cc.correlation_energy) > 1e-10)
    throw std::runtime_error("RCCSD Lambda fresh primal replay gate failed");

  const double energy_seed = -1.0;
  auto rhs_outputs =
      generated::run_lambda_rhs_cpu(p.nocc, p.nvir, in, &energy_seed, arena.data(), arena.size());
  std::vector<double> rhs(layout.dimension());
  layout.pack_weighted({rhs_outputs.t1, layout.n1}, {rhs_outputs.t2, layout.n2}, rhs);

  std::vector<double> packed_source;
  if (with_source) {
    // Project directly into the one retained packed vector. Separate dense
    // source copies are unnecessary and would add another simultaneous owner.
    packed_source.resize(layout.dimension());
    std::copy(t1_source.begin(), t1_source.end(), packed_source.begin());
    for (std::size_t k = 0; k < layout.representatives.size(); ++k) {
      const auto first = layout.representatives[k];
      const auto second = layout.partners[k];
      const double projected =
          first == second ? t2_source[first] : 0.5 * (t2_source[first] + t2_source[second]);
      packed_source[layout.n1 + k] = layout.sqrt_weights[layout.n1 + k] * projected;
      if (!std::isfinite(packed_source[layout.n1 + k]))
        throw std::invalid_argument("nonfinite projected RCCSD Lambda energy source");
    }
    for (std::size_t index = 0; index < rhs.size(); ++index) rhs[index] -= packed_source[index];
  }

  std::vector<double> dense_one(layout.n1), dense_two(layout.n2);
  auto apply = [&](std::span<const double> input, std::span<double> output) {
    layout.unpack_weighted(input, dense_one, dense_two);
    const auto action = generated::run_lambda_transpose_cpu(
        p.nocc, p.nvir, in, dense_one.data(), dense_two.data(), arena.data(), arena.size());
    layout.pack_weighted({action.t1, layout.n1}, {action.t2, layout.n2}, output);
  };

  auto solved = response::solve_gmres(gmres_plan, apply, rhs);
  if (!solved.converged()) throw std::runtime_error("RCCSD Lambda GMRES did not converge");

  layout.unpack_weighted(solved.solution, dense_one, dense_two);
  const auto independent_action = generated::run_lambda_independent_transpose_cpu(
      p.nocc, p.nvir, in, dense_one.data(), dense_two.data(), arena.data(), arena.size());
  std::vector<double> independent(layout.dimension());
  layout.pack_weighted({independent_action.t1, layout.n1}, {independent_action.t2, layout.n2},
                       independent);

  const auto independent_rhs_output = generated::run_lambda_independent_rhs_cpu(
      p.nocc, p.nvir, in, &energy_seed, arena.data(), arena.size());
  std::vector<double> independent_rhs(layout.dimension());
  layout.pack_weighted({independent_rhs_output.t1, layout.n1},
                       {independent_rhs_output.t2, layout.n2}, independent_rhs);
  if (!packed_source.empty())
    for (std::size_t index = 0; index < independent_rhs.size(); ++index)
      independent_rhs[index] -= packed_source[index];
  for (std::size_t index = 0; index < independent.size(); ++index)
    independent[index] -= independent_rhs[index];

  const double independent_norm = response::stable_norm(independent);
  double independent_max = 0.0;
  for (std::size_t index = 0; index < independent.size(); ++index)
    independent_max =
        std::max(independent_max, std::abs(independent[index] / layout.sqrt_weights[index]));
  if (std::max({solved.residual_norm, independent_norm, independent_max}) >
      options.lambda_tolerance)
    throw std::runtime_error("RCCSD Lambda independent physical residual gate failed");

  LambdaResult result;
  layout.publish(solved.solution, result.lambda1, result.lambda2);
  result.reason = "shared GMRES and expanded physical Lambda residual passed";
  result.diagnostic.cc_r1_max = replay_r1;
  result.diagnostic.cc_r2_max = replay_r2;
  result.diagnostic.lambda_residual_norm = solved.residual_norm;
  result.diagnostic.independent_residual_norm = independent_norm;
  result.diagnostic.independent_residual_max = independent_max;
  result.diagnostic.iterations = solved.iterations;
  result.diagnostic.operator_actions = solved.operator_actions;
  result.diagnostic.numeric_capacity_bytes = capacity;
  result.diagnostic.shared_program_hash = generated::lambda_transpose_program_hash;
  result.diagnostic.independent_program_hash = generated::lambda_independent_transpose_program_hash;
  return result;
}

LambdaResult solve_lambda_cpu(const Problem& problem, const SolverResult& cc_result,
                              const LambdaOptions& options) {
  return solve_lambda_cpu_impl(problem, cc_result, {}, {}, options);
}

LambdaResult solve_lambda_cpu_with_energy_source(const Problem& problem,
                                                 const SolverResult& cc_result,
                                                 std::span<const double> t1_source,
                                                 std::span<const double> t2_source,
                                                 const LambdaOptions& options) {
  return solve_lambda_cpu_impl(problem, cc_result, t1_source, t2_source, options);
}

}  // namespace vibeqc::cc
