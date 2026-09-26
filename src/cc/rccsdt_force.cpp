#include "cc/rccsdt_force.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cc/triples_response.hpp"
#include "generated_rccsd_cpu.hpp"
#include "posthf/capacity.hpp"
#include "posthf/mp2_derivative.hpp"
#include "posthf/mp2_gradient.hpp"
#include "posthf/native_provider.hpp"
#include "posthf/raw_source.hpp"
#include "scf/types.hpp"

namespace vibeqc::cc {
namespace {
constexpr double kStationarityTolerance = 1e-8;
constexpr double kOrbitalResidualTolerance = 1e-10;
constexpr double kCanonicalFockTolerance = 1e-9;
constexpr double kMinimumSameSpaceGap = 1e-10;
constexpr double kMinimumOrbitalCurvature = 1e-8;

std::size_t checked_add(std::size_t a, std::size_t b) { return posthf::checked_add(a, b); }
std::size_t checked_mul(std::size_t a, std::size_t b) { return posthf::checked_mul(a, b); }
std::size_t bytes(std::size_t n) { return checked_mul(n, sizeof(double)); }
std::size_t square(std::size_t n) { return checked_mul(n, n); }
std::size_t fourth(std::size_t n) { return square(square(n)); }

bool finite(std::span<const double> values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

std::vector<std::size_t> range(std::size_t n) {
  std::vector<std::size_t> result(n);
  std::iota(result.begin(), result.end(), std::size_t{0});
  return result;
}

generated::Inputs cc_inputs(const Problem& p, const SolverResult& cc) {
  return {p.foo.data(),  p.fov.data(),  p.fvv.data(),  p.ovov.data(), p.ovvo.data(),
          p.oovv.data(), p.ovvv.data(), p.ovoo.data(), p.oooo.data(), p.vvvv.data(),
          p.d1.data(),   p.d2.data(),   cc.t1.data(),  cc.t2.data()};
}

struct ParameterWeights {
  std::vector<double> foo, fov, fvv, ovov, ovvo, oovv, ovvv, ovoo, oooo, vvvv;
};

std::size_t parameter_elements(std::size_t o, std::size_t v) {
  auto total = checked_add(square(o), checked_add(checked_mul(o, v), square(v)));
  total = checked_add(total, checked_mul(square(o), square(v)));  // ovov
  total = checked_add(total, checked_mul(square(o), square(v)));  // ovvo
  total = checked_add(total, checked_mul(square(o), square(v)));  // oovv
  total = checked_add(total, checked_mul(o, checked_mul(v, square(v))));
  total = checked_add(total, checked_mul(checked_mul(o, v), square(o)));
  total = checked_add(total, fourth(o));
  total = checked_add(total, fourth(v));
  return total;
}

ParameterWeights parameter_vjp(const Problem& p, const SolverResult& cc, const LambdaResult& lambda,
                               std::size_t max_bytes) {
  const auto o = p.nocc, v = p.nvir;
  const auto arena_elements = std::max({generated::parameter_foo_arena_elements(o, v),
                                        generated::parameter_fov_arena_elements(o, v),
                                        generated::parameter_fvv_arena_elements(o, v),
                                        generated::parameter_ovov_arena_elements(o, v),
                                        generated::parameter_ovvo_arena_elements(o, v),
                                        generated::parameter_oovv_arena_elements(o, v),
                                        generated::parameter_ovvv_arena_elements(o, v),
                                        generated::parameter_ovoo_arena_elements(o, v),
                                        generated::parameter_oooo_arena_elements(o, v),
                                        generated::parameter_vvvv_arena_elements(o, v)});
  const auto required = checked_add(bytes(arena_elements), bytes(parameter_elements(o, v)));
  if (required > max_bytes)
    throw std::length_error("RCCSD(T) fixed-orbital response exceeds host budget");
  std::vector<double> arena(arena_elements);
  const auto in = cc_inputs(p, cc);
  const double energy_seed = 1.0;
  auto copy = [](generated::ParameterOutput output, std::size_t size) {
    return std::vector<double>(output.values, output.values + size);
  };
  ParameterWeights out;
  out.foo =
      copy(generated::run_parameter_foo_cpu(o, v, in, &energy_seed, lambda.lambda1.data(),
                                            lambda.lambda2.data(), arena.data(), arena.size()),
           square(o));
  out.fov =
      copy(generated::run_parameter_fov_cpu(o, v, in, &energy_seed, lambda.lambda1.data(),
                                            lambda.lambda2.data(), arena.data(), arena.size()),
           checked_mul(o, v));
  out.fvv =
      copy(generated::run_parameter_fvv_cpu(o, v, in, &energy_seed, lambda.lambda1.data(),
                                            lambda.lambda2.data(), arena.data(), arena.size()),
           square(v));
  out.ovov =
      copy(generated::run_parameter_ovov_cpu(o, v, in, &energy_seed, lambda.lambda1.data(),
                                             lambda.lambda2.data(), arena.data(), arena.size()),
           checked_mul(square(o), square(v)));
  out.ovvo =
      copy(generated::run_parameter_ovvo_cpu(o, v, in, &energy_seed, lambda.lambda1.data(),
                                             lambda.lambda2.data(), arena.data(), arena.size()),
           checked_mul(square(o), square(v)));
  out.oovv =
      copy(generated::run_parameter_oovv_cpu(o, v, in, &energy_seed, lambda.lambda1.data(),
                                             lambda.lambda2.data(), arena.data(), arena.size()),
           checked_mul(square(o), square(v)));
  out.ovvv =
      copy(generated::run_parameter_ovvv_cpu(o, v, in, &energy_seed, lambda.lambda1.data(),
                                             lambda.lambda2.data(), arena.data(), arena.size()),
           checked_mul(o, checked_mul(v, square(v))));
  out.ovoo =
      copy(generated::run_parameter_ovoo_cpu(o, v, in, &energy_seed, lambda.lambda1.data(),
                                             lambda.lambda2.data(), arena.data(), arena.size()),
           checked_mul(checked_mul(o, v), square(o)));
  out.oooo =
      copy(generated::run_parameter_oooo_cpu(o, v, in, &energy_seed, lambda.lambda1.data(),
                                             lambda.lambda2.data(), arena.data(), arena.size()),
           fourth(o));
  out.vvvv =
      copy(generated::run_parameter_vvvv_cpu(o, v, in, &energy_seed, lambda.lambda1.data(),
                                             lambda.lambda2.data(), arena.data(), arena.size()),
           fourth(v));
  return out;
}

void add_projected_triples(ParameterWeights& target, const TriplesResponseResult& triples,
                           std::size_t o, std::size_t v) {
  if (triples.fov.size() != target.fov.size() || triples.ovov.size() != target.ovov.size() ||
      triples.ovvv.size() != target.ovvv.size() || triples.ovoo.size() != target.ovoo.size())
    throw std::invalid_argument("RCCSD(T) direct triples parameter-source shape mismatch");
  for (std::size_t index = 0; index < target.fov.size(); ++index)
    target.fov[index] += triples.fov[index];
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t a = 0; a < v; ++a)
      for (std::size_t j = 0; j < o; ++j)
        for (std::size_t b = 0; b < v; ++b) {
          const auto first = ((i * v + a) * o + j) * v + b;
          const auto second = ((j * v + b) * o + i) * v + a;
          target.ovov[first] += 0.5 * (triples.ovov[first] + triples.ovov[second]);
        }
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t a = 0; a < v; ++a)
      for (std::size_t b = 0; b < v; ++b)
        for (std::size_t c = 0; c < v; ++c) {
          const auto first = ((i * v + a) * v + b) * v + c;
          const auto second = ((i * v + a) * v + c) * v + b;
          target.ovvv[first] += 0.5 * (triples.ovvv[first] + triples.ovvv[second]);
        }
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t a = 0; a < v; ++a)
      for (std::size_t j = 0; j < o; ++j)
        for (std::size_t k = 0; k < o; ++k) {
          const auto first = ((i * v + a) * o + j) * o + k;
          const auto second = ((i * v + a) * o + k) * o + j;
          target.ovoo[first] += 0.5 * (triples.ovoo[first] + triples.ovoo[second]);
        }
}

struct RawHamiltonian {
  std::vector<double> h, g, density, rotation;
};

RawHamiltonian raw_hamiltonian(const core::System& system, const scf::PhysicalReference& ref,
                               std::size_t max_bytes) {
  const auto n = ref.nbf;
  const auto n2 = square(n), n4 = fourth(n);
  const auto required = bytes(checked_add(checked_mul(3, n2), n4));
  if (required > max_bytes) throw std::length_error("RCCSD(T) raw Hamiltonian exceeds host budget");
  if (ref.coefficients.size() != n2 || ref.hcore.size() != n2)
    throw std::invalid_argument("RCCSD(T) reference one-electron shape mismatch");
  RawHamiltonian out;
  out.h.assign(n2, 0.0);
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t mu = 0; mu < n; ++mu)
        for (std::size_t nu = 0; nu < n; ++nu)
          out.h[p * n + q] +=
              ref.coefficients[mu * n + p] * ref.hcore[mu * n + nu] * ref.coefficients[nu * n + q];
  posthf::RawSource source(system);
  posthf::NativeBlockProvider provider(source, ref, max_bytes, 2);
  const auto all = range(n);
  out.g = provider.get({all, all, all, all}, false, 0);
  if (out.g.size() != n4) throw std::runtime_error("RCCSD(T) full MO ERI shape mismatch");
  out.density.assign(n2, 0.0);
  for (std::size_t i = 0; i < ref.nocc; ++i) out.density[i * n + i] = 2.0;
  out.rotation.assign(n2, 0.0);
  for (std::size_t p = 0; p < n; ++p) out.rotation[p * n + p] = 1.0;
  if (!finite(out.h) || !finite(out.g))
    throw std::runtime_error("nonfinite RCCSD(T) raw Hamiltonian");
  return out;
}

struct ResponseWeights {
  std::vector<double> hcore, eri, overlap, rotation_gradient, stationarity, orbital_rhs;
};

#if VIBEQC_HAS_CUDA
CudaParameterResponseView cuda_parameter_view(const ParameterWeights& bar) {
  return {std::span<const double>{bar.foo},  std::span<const double>{bar.fov},
          std::span<const double>{bar.fvv},  std::span<const double>{bar.ovov},
          std::span<const double>{bar.ovvo}, std::span<const double>{bar.oovv},
          std::span<const double>{bar.ovvv}, std::span<const double>{bar.ovoo},
          std::span<const double>{bar.oooo}, std::span<const double>{bar.vvvv}};
}

ResponseWeights detach_cuda_response(CudaHamiltonianResponseResult result) {
  return {std::move(result.hcore),        std::move(result.eri),
          std::move(result.overlap),      std::move(result.rotation_gradient),
          std::move(result.stationarity), std::move(result.orbital_rhs)};
}
#endif

ResponseWeights copy_hamiltonian_outputs(const generated::HamiltonianOutputs& output, std::size_t n,
                                         std::size_t o, std::size_t v) {
  const auto n2 = square(n), n4 = fourth(n), ov = checked_mul(o, v);
  return {{output.hcore, output.hcore + n2},
          {output.eri, output.eri + n4},
          {output.overlap, output.overlap + n2},
          {output.rotation_gradient, output.rotation_gradient + n2},
          {output.stationarity, output.stationarity + n2},
          {output.orbital_rhs, output.orbital_rhs + ov}};
}

ResponseWeights hamiltonian_pullback(const ParameterWeights& bar, double reference_seed,
                                     const RawHamiltonian& raw, std::size_t o, std::size_t v,
                                     std::size_t max_bytes) {
  const auto n = checked_add(o, v);
  const auto arena_elements = generated::hamiltonian_weights_arena_elements(o, v);
  if (checked_add(bytes(arena_elements),
                  bytes(checked_add(checked_add(fourth(n), checked_mul(4, square(n))),
                                    checked_mul(o, v)))) > max_bytes)
    throw std::length_error("RCCSD(T) Hamiltonian response exceeds host budget");
  std::vector<double> arena(arena_elements);
  generated::HamiltonianWeightInputs inputs{};
  inputs.bar_foo = bar.foo.data();
  inputs.bar_fov = bar.fov.data();
  inputs.bar_fvv = bar.fvv.data();
  inputs.bar_ovov = bar.ovov.data();
  inputs.bar_ovvo = bar.ovvo.data();
  inputs.bar_oovv = bar.oovv.data();
  inputs.bar_ovvv = bar.ovvv.data();
  inputs.bar_ovoo = bar.ovoo.data();
  inputs.bar_oooo = bar.oooo.data();
  inputs.bar_vvvv = bar.vvvv.data();
  inputs.bar_reference_electronic_energy = &reference_seed;
  inputs.density = raw.density.data();
  inputs.g = raw.g.data();
  inputs.h = raw.h.data();
  inputs.rotation = raw.rotation.data();
  const auto output =
      generated::run_hamiltonian_weights_cpu(o, v, inputs, arena.data(), arena.size());
  return copy_hamiltonian_outputs(output, n, o, v);
}

ResponseWeights fock_pullback(std::span<const double> bar_fock, const RawHamiltonian& raw,
                              std::size_t o, std::size_t v, std::size_t max_bytes) {
  const auto n = checked_add(o, v);
  if (bar_fock.size() != square(n))
    throw std::invalid_argument("RCCSD(T) Fock response shape mismatch");
  const auto arena_elements = generated::fock_weights_arena_elements(o, v);
  if (bytes(
          checked_add(arena_elements, checked_add(checked_add(fourth(n), checked_mul(4, square(n))),
                                                  checked_mul(o, v)))) > max_bytes)
    throw std::length_error("RCCSD(T) Fock response exceeds host budget");
  std::vector<double> arena(arena_elements);
  generated::FockWeightInputs inputs{};
  inputs.bar_fock = bar_fock.data();
  inputs.density = raw.density.data();
  inputs.g = raw.g.data();
  inputs.h = raw.h.data();
  inputs.rotation = raw.rotation.data();
  const auto output = generated::run_fock_weights_cpu(o, v, inputs, arena.data(), arena.size());
  return copy_hamiltonian_outputs(output, n, o, v);
}

void add_in_place(ResponseWeights& target, const ResponseWeights& source) {
  auto add = [](std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("RCCSD(T) response shape mismatch");
    for (std::size_t i = 0; i < a.size(); ++i) a[i] += b[i];
  };
  add(target.hcore, source.hcore);
  add(target.eri, source.eri);
  add(target.overlap, source.overlap);
  add(target.rotation_gradient, source.rotation_gradient);
  add(target.stationarity, source.stationarity);
  add(target.orbital_rhs, source.orbital_rhs);
}

ParameterWeights zero_parameters(std::size_t o, std::size_t v) {
  ParameterWeights z;
  z.foo.assign(square(o), 0.0);
  z.fov.assign(checked_mul(o, v), 0.0);
  z.fvv.assign(square(v), 0.0);
  z.ovov.assign(checked_mul(square(o), square(v)), 0.0);
  z.ovvo.assign(checked_mul(square(o), square(v)), 0.0);
  z.oovv.assign(checked_mul(square(o), square(v)), 0.0);
  z.ovvv.assign(checked_mul(o, checked_mul(v, square(v))), 0.0);
  z.ovoo.assign(checked_mul(checked_mul(o, v), square(o)), 0.0);
  z.oooo.assign(fourth(o), 0.0);
  z.vvvv.assign(fourth(v), 0.0);
  return z;
}

double max_abs(std::span<const double> values) {
  double result = 0.0;
  for (const double value : values) {
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite RCCSD(T) response weight");
    result = std::max(result, std::abs(value));
  }
  return result;
}

double minimum_symmetric_eigenvalue(std::vector<double> matrix, std::size_t n) {
  if (matrix.size() != square(n)) throw std::invalid_argument("RCCSD(T) response matrix shape");
  if (!n) return std::numeric_limits<double>::infinity();
  const auto max_sweeps = checked_mul(std::size_t{100}, square(n));
  for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
    std::size_t p = 0, q = 0;
    double largest = 0.0;
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = i + 1; j < n; ++j)
        if (std::abs(matrix[i * n + j]) > largest) {
          largest = std::abs(matrix[i * n + j]);
          p = i;
          q = j;
        }
    if (largest < 1e-13) break;
    const double app = matrix[p * n + p], aqq = matrix[q * n + q], apq = matrix[p * n + q];
    const double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
    const double c = std::cos(phi), s = std::sin(phi);
    for (std::size_t k = 0; k < n; ++k) {
      if (k == p || k == q) continue;
      const double akp = matrix[k * n + p], akq = matrix[k * n + q];
      matrix[k * n + p] = matrix[p * n + k] = c * akp - s * akq;
      matrix[k * n + q] = matrix[q * n + k] = s * akp + c * akq;
    }
    matrix[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
    matrix[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
    matrix[p * n + q] = matrix[q * n + p] = 0.0;
  }
  double result = matrix[0];
  for (std::size_t i = 1; i < n; ++i) result = std::min(result, matrix[i * n + i]);
  return result;
}

}  // namespace

RccsdtForcePlan plan_rccsdt_force_cpu(const core::System& system,
                                      const scf::PhysicalReference& reference, const Problem& p,
                                      const SolverResult& cc, std::size_t max_bytes) {
  const auto o = p.nocc, v = p.nvir, n = checked_add(o, v);
  if (!o || !v || n > 12 || reference.nbf != n || reference.nocc != o ||
      molecule::ao_count(system) != n || !max_bytes)
    throw std::invalid_argument("invalid RCCSD(T) force resource dimensions");
  const auto n2 = square(n), n4 = fourth(n), ov = checked_mul(o, v),
             amplitudes = checked_add(ov, square(ov));
  auto sum = [](std::initializer_list<std::size_t> values) {
    std::size_t result = 0;
    for (const auto value : values) result = checked_add(result, value);
    return result;
  };
  std::size_t reference_bytes = 0;
  for (const auto* values :
       {&reference.overlap, &reference.hcore, &reference.fock, &reference.coefficients,
        &reference.orbital_energies, &reference.density, &reference.weighted_density})
    reference_bytes = checked_add(reference_bytes, bytes(values->capacity()));
  RccsdtForcePlan plan;
  // Lambda's existing bound includes this borrowed subset. Subtract it only
  // when composing that stage, so the actual reference is charged once.
  const auto lambda_borrowed = sum({p.reference_retained_bytes, problem_host_bytes(p),
                                    bytes(cc.t1.capacity()), bytes(cc.t2.capacity())});
  plan.retained_input_bytes =
      sum({std::max(reference_bytes, p.reference_retained_bytes), problem_host_bytes(p),
           bytes(cc.t1.capacity()), bytes(cc.t2.capacity()), bytes(n),
           posthf::source_capacity(system)});
  const auto triples_retained =
      bytes(sum({checked_mul(o, checked_mul(v, square(v))), checked_mul(ov, square(o)),
                 checked_mul(2, square(ov)), checked_mul(2, ov), n}));
  const auto pages = std::min<std::size_t>(TriplesResponseOptions{}.batch_capacity,
                                           checked_mul(v, checked_mul(v + 1, v + 2)) / 6);
  plan.triples_phase_bytes =
      sum({plan.retained_input_bytes, bytes(n), triples_retained,
           bytes(generated::triples_response_arena_elements(o, v, pages)),
           checked_mul(pages, 3 * sizeof(std::int64_t) + 2 * sizeof(double))});
  LambdaOptions lambda_options;
  lambda_options.max_bytes = max_bytes;
  lambda_options.gmres.max_workspace_bytes = max_bytes;
  const auto lambda_capacity = lambda_cpu_numeric_capacity(p, cc, lambda_options, true);
  if (lambda_capacity < lambda_borrowed) throw std::logic_error("Lambda capacity underflow");
  plan.lambda_phase_bytes =
      sum({plan.retained_input_bytes, triples_retained, lambda_capacity - lambda_borrowed});
  const auto parameter_retained = bytes(parameter_elements(o, v));
  const auto parameter_arena = bytes(std::max({generated::parameter_foo_arena_elements(o, v),
                                               generated::parameter_fov_arena_elements(o, v),
                                               generated::parameter_fvv_arena_elements(o, v),
                                               generated::parameter_ovov_arena_elements(o, v),
                                               generated::parameter_ovvo_arena_elements(o, v),
                                               generated::parameter_oovv_arena_elements(o, v),
                                               generated::parameter_ovvv_arena_elements(o, v),
                                               generated::parameter_ovoo_arena_elements(o, v),
                                               generated::parameter_oooo_arena_elements(o, v),
                                               generated::parameter_vvvv_arena_elements(o, v)}));
  const auto before_raw =
      sum({plan.retained_input_bytes, triples_retained, bytes(amplitudes), parameter_retained});
  plan.parameter_phase_bytes = checked_add(before_raw, parameter_arena);
  std::size_t shell = 0;
  for (const auto& basis_shell : system.shells) {
    const auto l = static_cast<std::size_t>(basis_shell.angular_momentum);
    shell = std::max(shell, system.basis_representation == VIBEQC_BASIS_SPHERICAL
                                ? 2 * l + 1
                                : (l + 1) * (l + 2) / 2);
  }
  const auto tile = std::min<std::size_t>(2, shell);
  // The generated MO provider plan supplies its source/recurrence, coefficient,
  // full output and cyclic transform bounds. Its borrowed reference is already
  // in retained_input_bytes, so request only its additional buffers here.
  const auto provider = posthf::numeric_block_plan(n, 0, posthf::source_capacity(system),
                                                   {n, n, n, n}, {tile, tile, tile, tile}, false);
  plan.raw_phase_bytes = sum({before_raw, bytes(checked_mul(3, n2)), provider.host_bytes,
                              checked_mul(checked_mul(13, n), sizeof(std::size_t))});
  const auto raw_retained = bytes(sum({n4, checked_mul(3, n2)}));
  const auto response_retained = bytes(sum({n4, checked_mul(4, n2), ov}));
  const auto hamiltonian_arena = bytes(generated::hamiltonian_weights_arena_elements(o, v));
  const auto fock_arena = bytes(generated::fock_weights_arena_elements(o, v));
  const auto core = checked_add(before_raw, raw_retained);
  // Correlation, denominator and canonicalization outputs remain live. The
  // generated orbital action owns its arena throughout the remaining phases.
  const auto response_base = sum(
      {core, checked_mul(3, response_retained), bytes(generated::orbital_jvp_arena_elements(o, v)),
       bytes(sum({checked_mul(2, n2), square(ov), checked_mul(2, ov)}))});
  response::GmresOptions z_options;
  z_options.restart = 30;
  z_options.max_workspace_bytes = max_bytes;
  const auto gmres = response::prepare_gmres(ov, z_options);
  plan.response_phase_bytes =
      std::max({sum({core, response_retained, hamiltonian_arena}),
                sum({core, checked_mul(3, response_retained), bytes(n2), fock_arena}),
                checked_add(response_base, bytes(square(ov))),  // eigenvalue-check matrix copy
                checked_add(response_base, gmres.workspace_bytes),
                sum({response_base, bytes(checked_mul(2, ov)), checked_mul(2, parameter_retained),
                     checked_mul(2, response_retained), hamiltonian_arena})});
  // After the final pullback, only the temporary zero-parameter pack dies.
  // Moving total weights into the derivative consumer does not free their data.
  const auto derivative_live = sum({response_base, bytes(checked_mul(2, ov)), parameter_retained,
                                    checked_mul(2, response_retained)});
  const auto coordinates = checked_mul(3, system.atoms.size());
  const auto derivative_staging =
      bytes(sum({checked_mul(2, n2), checked_mul(shell, checked_mul(n, n2)),
                 checked_mul(square(shell), n2), checked_mul(checked_mul(shell, square(shell)), n),
                 fourth(shell), checked_mul(2, coordinates)}));
  plan.derivative_phase_bytes =
      sum({derivative_live, derivative_staging,
           checked_mul(system.shells.size() + 1, sizeof(std::size_t)),
           posthf::source_capacity(system), posthf::source_scratch_bytes});
  plan.peak_bytes =
      std::max({plan.triples_phase_bytes, plan.lambda_phase_bytes, plan.parameter_phase_bytes,
                plan.raw_phase_bytes, plan.response_phase_bytes, plan.derivative_phase_bytes});
  if (plan.peak_bytes > max_bytes)
    throw std::length_error("RCCSD(T) complete force exceeds simultaneous host budget");
  return plan;
}

static RccsdtForceResult rccsdt_force_impl(
    const core::System& system, const scf::PhysicalReference& reference, const Problem& problem,
    const SolverResult& cc_result, std::span<const double> eps_o, std::span<const double> eps_v,
    std::size_t max_bytes, bool cuda_derivative, int device_id, std::size_t derivative_stage_budget,
    double denominator_threshold) {
  validate_problem(problem);
#if !VIBEQC_HAS_CUDA
  if (cuda_derivative) throw std::runtime_error("RCCSD(T) CUDA force is unavailable in this build");
#endif
  if (!cc_result.converged())
    throw std::invalid_argument("RCCSD(T) force requires converged RCCSD amplitudes");
  if (!max_bytes || reference.nbf > 12 || reference.nbf != problem.nocc + problem.nvir ||
      reference.nocc != problem.nocc || eps_o.size() != problem.nocc ||
      eps_v.size() != problem.nvir || !std::isfinite(denominator_threshold) ||
      denominator_threshold <= 0.0 ||
      (cuda_derivative && (device_id < 0 || !derivative_stage_budget)))
    throw std::invalid_argument("RCCSD(T) force is outside the qualified conventional domain");
  const auto o = problem.nocc, v = problem.nvir, n = reference.nbf;
  if (reference.orbital_energies.size() != n || !finite(reference.orbital_energies))
    throw std::invalid_argument("RCCSD(T) force requires finite canonical orbital energies");
  const auto resources = plan_rccsdt_force_cpu(system, reference, problem, cc_result, max_bytes);

  TriplesResponseOptions triples_options;
  triples_options.denominator_threshold = denominator_threshold;
  triples_options.max_bytes = max_bytes;
  const auto triples =
      triples_response_cpu(problem, cc_result, std::vector<double>(eps_o.begin(), eps_o.end()),
                           std::vector<double>(eps_v.begin(), eps_v.end()), triples_options);

  LambdaOptions lambda_options;
  lambda_options.max_bytes = max_bytes;
  lambda_options.cc_tolerance = 1e-9;
  lambda_options.lambda_tolerance = 1e-9;
  lambda_options.gmres.absolute_tolerance = 1e-12;
  lambda_options.gmres.relative_tolerance = 0.0;
  lambda_options.gmres.max_workspace_bytes = max_bytes;
  LambdaResult corrected;
  ParameterWeights parameters;
#if VIBEQC_HAS_CUDA
  if (cuda_derivative) {
    auto fixed_orbital = solve_lambda_parameter_response_cuda_with_energy_source(
        problem, cc_result, triples.t1, triples.t2, device_id, lambda_options);
    corrected = std::move(fixed_orbital.lambda);
    parameters = {std::move(fixed_orbital.foo),  std::move(fixed_orbital.fov),
                  std::move(fixed_orbital.fvv),  std::move(fixed_orbital.ovov),
                  std::move(fixed_orbital.ovvo), std::move(fixed_orbital.oovv),
                  std::move(fixed_orbital.ovvv), std::move(fixed_orbital.ovoo),
                  std::move(fixed_orbital.oooo), std::move(fixed_orbital.vvvv)};
  } else
#endif
  {
    corrected = solve_lambda_cpu_with_energy_source(problem, cc_result, triples.t1, triples.t2,
                                                    lambda_options);
    parameters = parameter_vjp(problem, cc_result, corrected, max_bytes);
  }
  if (cuda_derivative && !corrected.diagnostic.cuda_actions)
    throw std::runtime_error("RCCSD(T) CUDA force lost CUDA Lambda action ownership");

  add_projected_triples(parameters, triples, o, v);
  const auto raw = raw_hamiltonian(system, reference, max_bytes);
#if VIBEQC_HAS_CUDA
  std::unique_ptr<CudaHamiltonianResponseOwner> cuda_response;
  if (cuda_derivative)
    cuda_response = std::make_unique<CudaHamiltonianResponseOwner>(
        o, v, CudaRawHamiltonianView{raw.density, raw.g, raw.h, raw.rotation}, device_id,
        max_bytes);
#endif
  auto hamiltonian_dispatch = [&](const ParameterWeights& bar,
                                  double reference_seed) -> ResponseWeights {
#if VIBEQC_HAS_CUDA
    if (cuda_response)
      return detach_cuda_response(
          cuda_response->hamiltonian(cuda_parameter_view(bar), reference_seed));
#endif
    return hamiltonian_pullback(bar, reference_seed, raw, o, v, max_bytes);
  };
  auto fock_dispatch = [&](std::span<const double> bar_fock) -> ResponseWeights {
#if VIBEQC_HAS_CUDA
    if (cuda_response) return detach_cuda_response(cuda_response->fock(bar_fock));
#endif
    return fock_pullback(bar_fock, raw, o, v, max_bytes);
  };
  auto correlation = hamiltonian_dispatch(parameters, 0.0);

  std::vector<double> bar_fock(square(n), 0.0);
  for (std::size_t i = 0; i < o; ++i) bar_fock[i * n + i] = triples.eps_o[i];
  for (std::size_t a = 0; a < v; ++a) bar_fock[(o + a) * n + o + a] = triples.eps_v[a];
  const auto denominator = fock_dispatch(bar_fock);
  add_in_place(correlation, denominator);

  double minimum_same_space_gap = std::numeric_limits<double>::infinity();
  std::fill(bar_fock.begin(), bar_fock.end(), 0.0);
  for (const auto& bounds :
       {std::pair<std::size_t, std::size_t>{0, o}, std::pair<std::size_t, std::size_t>{o, n}}) {
    for (std::size_t p = bounds.first; p < bounds.second; ++p)
      for (std::size_t q = p + 1; q < bounds.second; ++q) {
        const double gap = reference.orbital_energies[p] - reference.orbital_energies[q];
        minimum_same_space_gap = std::min(minimum_same_space_gap, std::abs(gap));
        if (std::abs(gap) <= kMinimumSameSpaceGap)
          throw std::runtime_error("degenerate RCCSD(T) canonical occupied/virtual subspace");
        const double value = -correlation.stationarity[p * n + q] / (2.0 * gap);
        bar_fock[p * n + q] = value;
        bar_fock[q * n + p] = value;
      }
  }
  const auto canonicalization = fock_dispatch(bar_fock);
  add_in_place(correlation, canonicalization);
  double same_space_stationarity = 0.0;
  for (const auto& bounds :
       {std::pair<std::size_t, std::size_t>{0, o}, std::pair<std::size_t, std::size_t>{o, n}})
    for (std::size_t p = bounds.first; p < bounds.second; ++p)
      for (std::size_t q = bounds.first; q < bounds.second; ++q)
        same_space_stationarity =
            std::max(same_space_stationarity, std::abs(correlation.stationarity[p * n + q]));
  if (same_space_stationarity > kStationarityTolerance)
    throw std::runtime_error("RCCSD(T) same-space canonicalization response failed");

  const auto orbital_arena_elements = generated::orbital_jvp_arena_elements(o, v);
  if (!cuda_derivative && bytes(orbital_arena_elements) > max_bytes)
    throw std::length_error("RCCSD(T) orbital-response action exceeds host budget");
  std::vector<double> orbital_arena;
  if (!cuda_derivative) orbital_arena.resize(orbital_arena_elements);
  std::vector<double> d_rotation(square(n), 0.0);
  generated::OrbitalJvpInputs orbital_inputs{};
  orbital_inputs.d_rotation = d_rotation.data();
  orbital_inputs.density = raw.density.data();
  orbital_inputs.g = raw.g.data();
  orbital_inputs.h = raw.h.data();
  orbital_inputs.rotation = raw.rotation.data();
  auto apply = [&](std::span<const double> input, std::span<double> output) {
    if (input.size() != checked_mul(o, v) || output.size() != input.size())
      throw std::invalid_argument("RCCSD(T) orbital-response vector shape mismatch");
    std::fill(d_rotation.begin(), d_rotation.end(), 0.0);
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t a = 0; a < v; ++a) {
        const double value = input[i * v + a];
        d_rotation[i * n + o + a] = value;
        d_rotation[(o + a) * n + i] = -value;
      }
#if VIBEQC_HAS_CUDA
    if (cuda_response) {
      const auto jvp = cuda_response->orbital_jvp(d_rotation);
      for (std::size_t index = 0; index < output.size(); ++index) output[index] = -jvp[index];
      return;
    }
#endif
    const auto jvp = generated::run_orbital_jvp_cpu(o, v, orbital_inputs, orbital_arena.data(),
                                                    orbital_arena.size());
    for (std::size_t index = 0; index < output.size(); ++index) output[index] = -jvp.d_fov[index];
  };

  const auto dimension = checked_mul(o, v);
  std::vector<double> response_matrix(square(dimension), 0.0), basis(dimension, 0.0),
      action(dimension);
  for (std::size_t column = 0; column < dimension; ++column) {
    std::fill(basis.begin(), basis.end(), 0.0);
    basis[column] = 1.0;
    apply(basis, action);
    for (std::size_t row = 0; row < dimension; ++row)
      response_matrix[row * dimension + column] = action[row];
  }
  double asymmetry = 0.0;
  for (std::size_t i = 0; i < dimension; ++i)
    for (std::size_t j = i + 1; j < dimension; ++j)
      asymmetry = std::max(asymmetry, std::abs(response_matrix[i * dimension + j] -
                                               response_matrix[j * dimension + i]));
  if (asymmetry > 1e-10)
    throw std::runtime_error("generated RCCSD(T) RHF response is not symmetric");
  const double minimum_curvature = minimum_symmetric_eigenvalue(response_matrix, dimension);
  if (!(minimum_curvature > kMinimumOrbitalCurvature))
    throw std::runtime_error("RCCSD(T) RHF orbital response is unstable or near-singular");

  response::GmresOptions z_options;
  z_options.relative_tolerance = 0.0;
  z_options.absolute_tolerance = 1e-12;
  z_options.restart = 30;
  z_options.max_iterations = 200;
  z_options.max_workspace_bytes = max_bytes;
  const auto z_plan = response::prepare_gmres(dimension, z_options);
  auto z = response::solve_gmres(z_plan, apply, correlation.orbital_rhs);
  if (!z.converged())
    throw std::runtime_error("RCCSD(T) physical orbital response did not converge");
  std::vector<double> independent(dimension, 0.0);
  for (std::size_t row = 0; row < dimension; ++row) {
    for (std::size_t column = 0; column < dimension; ++column)
      independent[row] += response_matrix[row * dimension + column] * z.solution[column];
    independent[row] -= correlation.orbital_rhs[row];
  }
  const double independent_residual = response::stable_norm(independent);
  if (std::max(z.residual_norm, independent_residual) > kOrbitalResidualTolerance)
    throw std::runtime_error("independent RCCSD(T) physical Z-vector residual failed");

  auto orbital_parameters = zero_parameters(o, v);
  for (std::size_t index = 0; index < dimension; ++index)
    orbital_parameters.fov[index] = -z.solution[index];
  const auto orbital = hamiltonian_dispatch(orbital_parameters, 0.0);
  auto total = hamiltonian_dispatch(zero_parameters(o, v), 1.0);
  add_in_place(total, correlation);
  add_in_place(total, orbital);
  const double stationarity = max_abs(total.stationarity);
  if (stationarity > kStationarityTolerance)
    throw std::runtime_error("complete HF+RCCSD(T)+Z orbital stationarity failed");

  mp2::LagrangianWeights weights;
  weights.orbitals = n;
  weights.occupied = o;
  weights.one_electron = std::move(total.hcore);
  weights.two_electron = std::move(total.eri);
  weights.overlap = std::move(total.overlap);
  weights.stationarity_residual = stationarity;
  auto gradient = cuda_derivative
                      ? mp2::conventional_derivative_cuda(system, reference, weights, device_id,
                                                          derivative_stage_budget)
                      : mp2::conventional_derivative_cpu(system, reference, weights);
  for (double& value : gradient) value = -value;
  if (!finite(gradient)) throw std::runtime_error("nonfinite RCCSD(T) analytic force");

  RccsdtForceResult result;
  result.forces = std::move(gradient);
  result.lambda = corrected.diagnostic;
  result.orbital_response = std::move(z);
  result.independent_orbital_residual = independent_residual;
  result.orbital_stationarity = stationarity;
  result.minimum_orbital_curvature = minimum_curvature;
  result.minimum_same_space_gap = minimum_same_space_gap;
  result.triples_response_pages = triples.pages;
  result.numeric_capacity_bytes = resources.peak_bytes;
#if VIBEQC_HAS_CUDA
  if (cuda_response) {
    result.response_owned_device_bytes = cuda_response->owned_device_bytes();
    result.response_h2d_bytes = cuda_response->h2d_bytes();
    result.response_d2h_bytes = cuda_response->d2h_bytes();
    result.response_synchronizations = cuda_response->synchronizations();
    result.cuda_response_actions = true;
  }
#endif
  result.response_operator_hash = generated::orbital_jvp_program_hash;
  return result;
}

RccsdtForceResult rccsdt_force_cpu(const core::System& system,
                                   const scf::PhysicalReference& reference, const Problem& problem,
                                   const SolverResult& cc_result, std::span<const double> eps_o,
                                   std::span<const double> eps_v, std::size_t max_bytes,
                                   double denominator_threshold) {
  return rccsdt_force_impl(system, reference, problem, cc_result, eps_o, eps_v, max_bytes, false, 0,
                           0, denominator_threshold);
}

RccsdtForceResult rccsdt_force_cuda(const core::System& system,
                                    const scf::PhysicalReference& reference, const Problem& problem,
                                    const SolverResult& cc_result, std::span<const double> eps_o,
                                    std::span<const double> eps_v, std::size_t max_bytes,
                                    int device_id, std::size_t derivative_stage_budget,
                                    double denominator_threshold) {
  return rccsdt_force_impl(system, reference, problem, cc_result, eps_o, eps_v, max_bytes, true,
                           device_id, derivative_stage_budget, denominator_threshold);
}

}  // namespace vibeqc::cc
