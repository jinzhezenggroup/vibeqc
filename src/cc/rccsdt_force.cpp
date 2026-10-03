#include "cc/rccsdt_force.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cc/triples_response.hpp"
#include "cc/triples_response_internal.hpp"
#include "generated_rccsd_cpu.hpp"
#include "posthf/capacity.hpp"
#include "posthf/mp2_derivative.hpp"
#include "posthf/mp2_gradient.hpp"
#include "posthf/native_provider.hpp"
#include "posthf/raw_source.hpp"
#include "runtime/df_progress_trace.hpp"
#include "scf/types.hpp"
#include "tensor/cpu_linalg.hpp"

namespace generativeqc::cc {
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

void add_triples_parameter_sources(ParameterWeights& target, const TriplesResponseResult& triples) {
  if (triples.fov.size() != target.fov.size() || triples.ovov.size() != target.ovov.size() ||
      triples.ovvv.size() != target.ovvv.size() || triples.ovoo.size() != target.ovoo.size())
    throw std::invalid_argument("RCCSD(T) direct triples parameter-source shape mismatch");
  auto add = [](std::vector<double>& destination, const std::vector<double>& source) {
    for (std::size_t index = 0; index < destination.size(); ++index)
      destination[index] += source[index];
  };
  add(target.fov, triples.fov);
  add(target.ovov, triples.ovov);
  add(target.ovvv, triples.ovvv);
  add(target.ovoo, triples.ovoo);
}

// A borrowed interaction source must describe the same nuclear Hamiltonian.
// Compare metadata before memory admission or reads, without allocating a second
// identity vector or accepting equal AO counts as proof of source equivalence.
bool force_source_matches_system(const core::System& a, const core::System& b) noexcept {
  if (a.basis_representation != b.basis_representation || a.charge != b.charge ||
      a.multiplicity != b.multiplicity || a.electron_count != b.electron_count ||
      a.atoms.size() != b.atoms.size() || a.shells.size() != b.shells.size() ||
      a.ecp_terms != b.ecp_terms)
    return false;
  for (std::size_t i = 0; i < a.atoms.size(); ++i) {
    const auto& left = a.atoms[i];
    const auto& right = b.atoms[i];
    if (left.atomic_number != right.atomic_number || left.ecp_core != right.ecp_core ||
        left.position != right.position)
      return false;
  }
  for (std::size_t i = 0; i < a.shells.size(); ++i) {
    const auto& left = a.shells[i];
    const auto& right = b.shells[i];
    if (left.atom_index != right.atom_index || left.angular_momentum != right.angular_momentum ||
        left.primitives.size() != right.primitives.size())
      return false;
    for (std::size_t j = 0; j < left.primitives.size(); ++j)
      if (left.primitives[j].exponent != right.primitives[j].exponent ||
          left.primitives[j].coefficient != right.primitives[j].coefficient)
        return false;
  }
  return true;
}

struct RawHamiltonian {
  std::vector<double> h, g, density, rotation;
};

RawHamiltonian raw_hamiltonian(const integrals::ElectronInteractionSource& source,
                               const hf::PhysicalReference& ref, std::size_t max_bytes, bool cuda,
                               int device_id, std::size_t provider_budget, unsigned axis_tile,
                               posthf::ProviderWork& work) {
  const auto n = ref.nbf;
  if (source.nbf() != n || !source.supports(integrals::ElectronInteractionOperator::eri))
    throw std::invalid_argument("RCCSD(T) raw Hamiltonian source/reference mismatch");
  const auto n2 = square(n), n4 = fourth(n);
  const auto retained_required = bytes(checked_add(checked_mul(3, n2), n4));
  const auto transform_required = bytes(checked_mul(2, n2));
  const auto required = std::max(retained_required, transform_required);
  if (required > max_bytes) throw std::length_error("RCCSD(T) raw Hamiltonian exceeds host budget");
  if (ref.coefficients.size() != n2 || ref.hcore.size() != n2)
    throw std::invalid_argument("RCCSD(T) reference one-electron shape mismatch");
  RawHamiltonian out;
  out.h.assign(n2, 0.0);
  {
    std::vector<double> workspace(n2);
    tensor::cpu_congruence('T', n, ref.coefficients.data(), ref.hcore.data(), out.h.data(),
                           workspace.data());
  }
  posthf::NativeBlockProvider provider(source, ref, provider_budget, axis_tile,
                                       posthf::AOTileDomain::Basis);
  const auto all = range(n);
  out.g = provider.get({all, all, all, all}, cuda, device_id, nullptr, &work);
  if (out.g.size() != n4) throw std::runtime_error("RCCSD(T) full MO ERI shape mismatch");
  out.density.assign(n2, 0.0);
  for (std::size_t i = 0; i < ref.nocc; ++i) out.density[i * n + i] = 2.0;
  out.rotation.assign(n2, 0.0);
  for (std::size_t p = 0; p < n; ++p) out.rotation[p * n + p] = 1.0;
  if (!finite(out.h) || !finite(out.g))
    throw std::runtime_error("nonfinite RCCSD(T) raw Hamiltonian");
  return out;
}

struct SmallResponseWeights {
  std::vector<double> hcore, overlap, rotation_gradient, stationarity, orbital_rhs;
};

struct ControlWeights {
  std::vector<double> stationarity, orbital_rhs;
};

#if GENERATIVEQC_HAS_CUDA
CudaParameterResponseView cuda_parameter_view(const ParameterWeights& bar) {
  return {std::span<const double>{bar.foo},  std::span<const double>{bar.fov},
          std::span<const double>{bar.fvv},  std::span<const double>{bar.ovov},
          std::span<const double>{bar.ovvo}, std::span<const double>{bar.oovv},
          std::span<const double>{bar.ovvv}, std::span<const double>{bar.ovoo},
          std::span<const double>{bar.oooo}, std::span<const double>{bar.vvvv}};
}

SmallResponseWeights detach_cuda_small(CudaHamiltonianSmallResponseResult result) {
  return {std::move(result.hcore), std::move(result.overlap), std::move(result.rotation_gradient),
          std::move(result.stationarity), std::move(result.orbital_rhs)};
}
#endif

SmallResponseWeights copy_small_hamiltonian_outputs(
    const generated::HamiltonianSmallOutputs& output, std::size_t n, std::size_t o, std::size_t v) {
  const auto n2 = square(n), ov = checked_mul(o, v);
  return {{output.hcore, output.hcore + n2},
          {output.overlap, output.overlap + n2},
          {output.rotation_gradient, output.rotation_gradient + n2},
          {output.stationarity, output.stationarity + n2},
          {output.orbital_rhs, output.orbital_rhs + ov}};
}

generated::HamiltonianWeightInputs hamiltonian_weight_inputs(const ParameterWeights& bar,
                                                             const double* reference_seed,
                                                             const RawHamiltonian& raw) {
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
  inputs.bar_reference_electronic_energy = reference_seed;
  inputs.density = raw.density.data();
  inputs.g = raw.g.data();
  inputs.h = raw.h.data();
  inputs.rotation = raw.rotation.data();
  return inputs;
}

SmallResponseWeights hamiltonian_small_pullback(const ParameterWeights& bar, double reference_seed,
                                                const RawHamiltonian& raw, std::size_t o,
                                                std::size_t v, std::size_t max_bytes) {
  const auto n = checked_add(o, v);
  const auto arena_elements = generated::hamiltonian_small_weights_arena_elements(o, v);
  const auto retained_elements = checked_add(checked_mul(4, square(n)), checked_mul(o, v));
  if (checked_add(bytes(arena_elements), bytes(retained_elements)) > max_bytes)
    throw std::length_error("RCCSD(T) small Hamiltonian response exceeds host budget");
  std::vector<double> arena(arena_elements);
  const auto inputs = hamiltonian_weight_inputs(bar, &reference_seed, raw);
  const auto output =
      generated::run_hamiltonian_small_weights_cpu(o, v, inputs, arena.data(), arena.size());
  return copy_small_hamiltonian_outputs(output, n, o, v);
}

std::vector<double> hamiltonian_eri_pullback(const ParameterWeights& bar, double reference_seed,
                                             const RawHamiltonian& raw, std::size_t o,
                                             std::size_t v, std::size_t max_bytes) {
  const auto n4 = fourth(checked_add(o, v));
  const auto arena_elements = generated::hamiltonian_eri_weights_arena_elements(o, v);
  if (checked_add(bytes(arena_elements), bytes(n4)) > max_bytes)
    throw std::length_error("RCCSD(T) ERI response exceeds host budget");
  std::vector<double> arena(arena_elements);
  const auto inputs = hamiltonian_weight_inputs(bar, &reference_seed, raw);
  const auto output =
      generated::run_hamiltonian_eri_weights_cpu(o, v, inputs, arena.data(), arena.size());
  return {output.eri, output.eri + n4};
}

ControlWeights hamiltonian_control_pullback(const ParameterWeights& bar, double reference_seed,
                                            const RawHamiltonian& raw, std::size_t o, std::size_t v,
                                            std::size_t max_bytes) {
  const auto n = checked_add(o, v), ov = checked_mul(o, v);
  const auto arena_elements = generated::hamiltonian_control_arena_elements(o, v);
  if (checked_add(bytes(arena_elements), bytes(checked_add(square(n), ov))) > max_bytes)
    throw std::length_error("RCCSD(T) Hamiltonian control response exceeds host budget");
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
      generated::run_hamiltonian_control_cpu(o, v, inputs, arena.data(), arena.size());
  return {{output.stationarity, output.stationarity + square(n)},
          {output.orbital_rhs, output.orbital_rhs + ov}};
}

void add_same_space_fock_seed(ParameterWeights& target, std::span<const double> bar_fock,
                              std::size_t o, std::size_t v) {
  const auto n = checked_add(o, v);
  if (bar_fock.size() != square(n))
    throw std::invalid_argument("RCCSD(T) Fock seed shape mismatch");
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t a = 0; a < v; ++a)
      if (bar_fock[i * n + o + a] != 0.0 || bar_fock[(o + a) * n + i] != 0.0)
        throw std::logic_error("RCCSD(T) control folding accepts same-space Fock seeds only");
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j) target.foo[i * o + j] += bar_fock[i * n + j];
  for (std::size_t a = 0; a < v; ++a)
    for (std::size_t b = 0; b < v; ++b) target.fvv[a * v + b] += bar_fock[(o + a) * n + o + b];
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

static RccsdtForcePlan plan_relaxed_rccsd_force_cpu(const core::System& system,
                                                    const hf::PhysicalReference& reference,
                                                    const Problem& p, const SolverResult& cc,
                                                    std::size_t max_bytes, bool include_triples,
                                                    std::size_t source_bytes,
                                                    bool cuda_transform = false) {
  const auto o = p.nocc, v = p.nvir, n = checked_add(o, v);
  if (!o || !v || n > (include_triples ? kRccsdtForceMaxAOs : 12) || reference.nbf != n ||
      reference.nocc != o || molecule::ao_count(system) != n || !max_bytes)
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
           bytes(cc.t1.capacity()), bytes(cc.t2.capacity()), bytes(n), source_bytes});
  const auto triples_retained =
      include_triples
          ? bytes(sum({checked_mul(o, checked_mul(v, square(v))), checked_mul(ov, square(o)),
                       checked_mul(2, square(ov)), checked_mul(2, ov), n}))
          : 0;
  plan.triples_phase_bytes =
      include_triples ? sum({plan.retained_input_bytes, bytes(n),
                             detail::triples_response_layout(
                                 o, v, TriplesResponseOptions{}.batch_capacity, cuda_transform)
                                 .numeric_bytes()})
                      : 0;
  LambdaOptions lambda_options;
  lambda_options.max_bytes = max_bytes;
  lambda_options.gmres.max_workspace_bytes = max_bytes;
  const auto lambda_capacity = lambda_cpu_numeric_capacity(p, cc, lambda_options, include_triples);
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
    shell = std::max(shell, system.basis_representation == GENERATIVEQC_BASIS_SPHERICAL
                                ? 2 * l + 1
                                : (l + 1) * (l + 2) / 2);
  }
  const std::size_t tile = 1;
  plan.raw_provider_axis_tile = static_cast<unsigned>(tile);
  // The generated MO provider plan supplies its source/recurrence, coefficient,
  // full output and cyclic transform bounds. Its borrowed reference is already
  // in retained_input_bytes, so request only its additional buffers here.
  // The source is borrowed and already charged in retained_input_bytes.
  // Ask the generated provider plan only for its additional staging/output buffers.
  const auto provider =
      posthf::numeric_block_plan(n, 0, 0, {n, n, n, n}, {tile, tile, tile, tile}, cuda_transform);
  // The nested provider owns only this phase's buffers. Add its borrowed
  // reference/source for its local admission without charging them twice in
  // the complete endpoint peak below.
  plan.raw_provider_budget_bytes = sum({provider.host_bytes, provider.device_bytes,
                                        bytes(checked_add(checked_mul(5, n2), n)), source_bytes});
  const auto rank2_transform_phase = sum({before_raw, bytes(checked_mul(2, n2))});
  const auto provider_phase =
      sum({before_raw, bytes(checked_mul(3, n2)), provider.host_bytes, provider.device_bytes,
           checked_mul(checked_mul(13, n), sizeof(std::size_t))});
  plan.raw_phase_bytes = std::max(rank2_transform_phase, provider_phase);
  const auto raw_retained = bytes(sum({n4, checked_mul(3, n2)}));
  const auto small_response_retained = bytes(sum({checked_mul(4, n2), ov}));
  const auto eri_response_retained = bytes(n4);
  const auto hamiltonian_small_arena =
      bytes(generated::hamiltonian_small_weights_arena_elements(o, v));
  const auto hamiltonian_eri_arena = bytes(generated::hamiltonian_eri_weights_arena_elements(o, v));
  const auto control_arena = bytes(generated::hamiltonian_control_arena_elements(o, v));
  const auto core = checked_add(before_raw, raw_retained);
  // Retain the conservative split-response envelope while introducing the
  // control-only CPU arena. The control owner is smaller than this bound and
  // is released before final small/ERI publication; no admission cap is relaxed.
  const auto response_base =
      sum({core, checked_mul(2, small_response_retained),
           bytes(generated::orbital_jvp_arena_elements(o, v)),
           bytes(sum({checked_mul(3, n2), square(ov), checked_mul(2, ov)}))});
  response::GmresOptions z_options;
  z_options.restart = 30;
  z_options.max_workspace_bytes = max_bytes;
  const auto gmres = response::prepare_gmres(ov, z_options);
  // This remains an upper bound after early control-workspace release.
  // Moving final weights into the derivative consumer does not free their data.
  const auto final_response_base = checked_add(response_base, bytes(checked_mul(2, ov)));
  plan.response_phase_bytes =
      std::max({sum({core, checked_mul(2, small_response_retained), hamiltonian_small_arena}),
                sum({core, checked_mul(2, small_response_retained), bytes(checked_mul(2, n2)),
                     control_arena}),
                checked_add(response_base, bytes(square(ov))),  // eigenvalue-check matrix copy
                checked_add(response_base, gmres.workspace_bytes),
                sum({final_response_base, small_response_retained, hamiltonian_small_arena}),
                sum({final_response_base, small_response_retained, eri_response_retained,
                     hamiltonian_eri_arena})});
  const auto derivative_live =
      sum({final_response_base, small_response_retained, eri_response_retained});
  const auto coordinates = checked_mul(3, system.atoms.size());
  const auto derivative_staging =
      bytes(sum({checked_mul(3, n2), checked_mul(shell, checked_mul(n, n2)),
                 checked_mul(square(shell), n2), checked_mul(checked_mul(shell, square(shell)), n),
                 fourth(shell), checked_mul(2, coordinates)}));
  plan.derivative_phase_bytes =
      sum({derivative_live, derivative_staging,
           checked_mul(system.shells.size() + 1, sizeof(std::size_t)),
           posthf::source_capacity(system), posthf::source_scratch_bytes});
  plan.peak_bytes =
      std::max({plan.lambda_phase_bytes, plan.parameter_phase_bytes, plan.raw_phase_bytes,
                plan.response_phase_bytes, plan.derivative_phase_bytes, plan.triples_phase_bytes});
  plan.minimum_peak_bytes = plan.peak_bytes;
  if (plan.peak_bytes > max_bytes)
    throw std::length_error("RCCSD(T) complete force exceeds simultaneous host budget");
  // Admit wider source tiles against the complete endpoint, not merely the
  // provider's local allowance. A roomy selected peak is not a minimum budget:
  // tight callers can shrink to width one before any numerical source work.
  for (std::size_t width = n; width > tile; --width) {
    const auto wider = posthf::numeric_block_plan(n, 0, 0, {n, n, n, n},
                                                  {width, width, width, width}, cuda_transform);
    if (cuda_transform && wider.stage_elements > INT32_MAX) continue;
    const auto phase =
        sum({before_raw, bytes(checked_mul(3, n2)), wider.host_bytes, wider.device_bytes,
             checked_mul(checked_mul(13, n), sizeof(std::size_t))});
    if (phase > max_bytes) continue;
    plan.raw_provider_axis_tile = static_cast<unsigned>(width);
    plan.raw_provider_budget_bytes = sum({wider.host_bytes, wider.device_bytes,
                                          bytes(checked_add(checked_mul(5, n2), n)), source_bytes});
    plan.raw_phase_bytes = std::max(rank2_transform_phase, phase);
    plan.peak_bytes = std::max(plan.minimum_peak_bytes, plan.raw_phase_bytes);
    break;
  }
  return plan;
}

RccsdtForcePlan plan_rccsd_force_cpu(const core::System& system,
                                     const integrals::ElectronInteractionSource& source,
                                     const hf::PhysicalReference& reference, const Problem& p,
                                     const SolverResult& cc, std::size_t max_bytes) {
  if (source.nbf() != reference.nbf ||
      !source.supports(integrals::ElectronInteractionOperator::eri) ||
      !force_source_matches_system(system, source.orbital()))
    throw std::invalid_argument("RCCSD force interaction source/reference mismatch");
  return plan_relaxed_rccsd_force_cpu(system, reference, p, cc, max_bytes, false,
                                      source.retained_numeric_bytes());
}

RccsdtForcePlan plan_rccsd_force_cpu(const core::System& system,
                                     const hf::PhysicalReference& reference, const Problem& p,
                                     const SolverResult& cc, std::size_t max_bytes) {
  return plan_relaxed_rccsd_force_cpu(system, reference, p, cc, max_bytes, false,
                                      posthf::source_capacity(system));
}

RccsdtForcePlan plan_rccsdt_force_cpu(const core::System& system,
                                      const integrals::ElectronInteractionSource& source,
                                      const hf::PhysicalReference& reference, const Problem& p,
                                      const SolverResult& cc, std::size_t max_bytes) {
  if (source.nbf() != reference.nbf ||
      !source.supports(integrals::ElectronInteractionOperator::eri) ||
      !force_source_matches_system(system, source.orbital()))
    throw std::invalid_argument("RCCSD(T) force interaction source/reference mismatch");
  return plan_relaxed_rccsd_force_cpu(system, reference, p, cc, max_bytes, true,
                                      source.retained_numeric_bytes());
}

RccsdtForcePlan plan_rccsdt_force_cpu(const core::System& system,
                                      const hf::PhysicalReference& reference, const Problem& p,
                                      const SolverResult& cc, std::size_t max_bytes) {
  return plan_relaxed_rccsd_force_cpu(system, reference, p, cc, max_bytes, true,
                                      posthf::source_capacity(system));
}

static RccsdtForceResult relaxed_rccsd_force_impl(
    const core::System& system, const integrals::ElectronInteractionSource& source,
    const hf::PhysicalReference& reference, const Problem& problem, const SolverResult& cc_result,
    std::span<const double> eps_o, std::span<const double> eps_v, std::size_t max_bytes,
    bool include_triples, bool cuda_derivative, int device_id, std::size_t derivative_stage_budget,
    double denominator_threshold) {
  validate_problem(problem);
#if !GENERATIVEQC_HAS_CUDA
  if (cuda_derivative) throw std::runtime_error("RCCSD(T) CUDA force is unavailable in this build");
#endif
  if (!cc_result.converged())
    throw std::invalid_argument("RCCSD(T) force requires converged RCCSD amplitudes");
  if (!max_bytes || reference.nbf > (include_triples ? kRccsdtForceMaxAOs : 12) ||
      reference.nbf != problem.nocc + problem.nvir || reference.nocc != problem.nocc ||
      eps_o.size() != problem.nocc || eps_v.size() != problem.nvir ||
      !std::isfinite(denominator_threshold) || denominator_threshold <= 0.0 ||
      (cuda_derivative && (device_id < 0 || !derivative_stage_budget)))
    throw std::invalid_argument("RCCSD(T) force is outside the qualified conventional domain");
  const auto o = problem.nocc, v = problem.nvir, n = reference.nbf;
  if (reference.orbital_energies.size() != n || !finite(reference.orbital_energies))
    throw std::invalid_argument("RCCSD(T) force requires finite canonical orbital energies");
  if (source.nbf() != reference.nbf ||
      !source.supports(integrals::ElectronInteractionOperator::eri) ||
      !force_source_matches_system(system, source.orbital()))
    throw std::invalid_argument("RCCSD(T) force interaction source/reference mismatch");
  const auto resources = plan_relaxed_rccsd_force_cpu(
      system, reference, problem, cc_result, max_bytes, include_triples,
      source.retained_numeric_bytes(), cuda_derivative);

  using Clock = std::chrono::steady_clock;
  const auto triples_started = Clock::now();

  std::optional<TriplesResponseResult> triples;
  if (include_triples) {
    TriplesResponseOptions triples_options;
    triples_options.denominator_threshold = denominator_threshold;
    triples_options.max_bytes = max_bytes - resources.retained_input_bytes - bytes(n);
#if GENERATIVEQC_HAS_CUDA
    if (cuda_derivative)
      triples.emplace(triples_response_cuda(
          problem, cc_result, std::vector<double>(eps_o.begin(), eps_o.end()),
          std::vector<double>(eps_v.begin(), eps_v.end()), device_id, triples_options));
    else
#endif
      triples.emplace(
          triples_response_cpu(problem, cc_result, std::vector<double>(eps_o.begin(), eps_o.end()),
                               std::vector<double>(eps_v.begin(), eps_v.end()), triples_options));
  }

  LambdaOptions lambda_options;
  const auto lambda_started = Clock::now();
  lambda_options.max_bytes = max_bytes;
  lambda_options.cc_tolerance = 1e-9;
  lambda_options.lambda_tolerance = 1e-9;
  lambda_options.gmres.absolute_tolerance = 1e-12;
  lambda_options.gmres.relative_tolerance = 0.0;
  lambda_options.gmres.max_workspace_bytes = max_bytes;
  LambdaResult corrected;
  ParameterWeights parameters;
#if GENERATIVEQC_HAS_CUDA
  if (cuda_derivative) {
    auto fixed_orbital =
        include_triples
            ? solve_lambda_parameter_response_cuda_with_energy_source(
                  problem, cc_result, triples->t1, triples->t2, device_id, lambda_options)
            : solve_lambda_parameter_response_cuda(problem, cc_result, device_id, lambda_options);
    corrected = std::move(fixed_orbital.lambda);
    parameters = {std::move(fixed_orbital.foo),  std::move(fixed_orbital.fov),
                  std::move(fixed_orbital.fvv),  std::move(fixed_orbital.ovov),
                  std::move(fixed_orbital.ovvo), std::move(fixed_orbital.oovv),
                  std::move(fixed_orbital.ovvv), std::move(fixed_orbital.ovoo),
                  std::move(fixed_orbital.oooo), std::move(fixed_orbital.vvvv)};
  } else
#endif
  {
    corrected = include_triples ? solve_lambda_cpu_with_energy_source(
                                      problem, cc_result, triples->t1, triples->t2, lambda_options)
                                : solve_lambda_cpu(problem, cc_result, lambda_options);
    parameters = parameter_vjp(problem, cc_result, corrected, max_bytes);
  }
  if (cuda_derivative && !corrected.diagnostic.cuda_actions)
    throw std::runtime_error("RCCSD(T) CUDA force lost CUDA Lambda action ownership");

  if (triples) add_triples_parameter_sources(parameters, *triples);
  const auto raw_started = Clock::now();
  posthf::ProviderWork raw_work;
  auto raw = raw_hamiltonian(source, reference, max_bytes, cuda_derivative,
                             cuda_derivative ? device_id : 0, resources.raw_provider_budget_bytes,
                             resources.raw_provider_axis_tile, raw_work);
  const auto orbital_started = Clock::now();
#if GENERATIVEQC_HAS_CUDA
  std::unique_ptr<CudaHamiltonianResponseOwner> cuda_response;
  if (cuda_derivative)
    cuda_response = std::make_unique<CudaHamiltonianResponseOwner>(
        o, v, CudaRawHamiltonianView{raw.density, raw.g, raw.h, raw.rotation}, device_id,
        max_bytes);
#endif
  auto hamiltonian_dispatch = [&](const ParameterWeights& bar,
                                  double reference_seed) -> SmallResponseWeights {
#if GENERATIVEQC_HAS_CUDA
    if (cuda_response)
      return detach_cuda_small(
          cuda_response->hamiltonian_small(cuda_parameter_view(bar), reference_seed));
#endif
    return hamiltonian_small_pullback(bar, reference_seed, raw, o, v, max_bytes);
  };
  auto eri_dispatch = [&](const ParameterWeights& bar, double reference_seed) {
#if GENERATIVEQC_HAS_CUDA
    if (cuda_response)
      return cuda_response->hamiltonian_eri(cuda_parameter_view(bar), reference_seed);
#endif
    return hamiltonian_eri_pullback(bar, reference_seed, raw, o, v, max_bytes);
  };
  auto control_dispatch = [&](const ParameterWeights& bar,
                              double reference_seed) -> ControlWeights {
#if GENERATIVEQC_HAS_CUDA
    if (cuda_response) {
      auto full = detach_cuda_small(
          cuda_response->hamiltonian_small(cuda_parameter_view(bar), reference_seed));
      return {std::move(full.stationarity), std::move(full.orbital_rhs)};
    }
#endif
    return hamiltonian_control_pullback(bar, reference_seed, raw, o, v, max_bytes);
  };

  // Canonical-orbital denominator sources are same-space Fock cotangents.
  // Fold them into the already-declared foo/fvv parameter seeds so every
  // intermediate orbital-control query can use the pruned Hamiltonian VJP.
  std::vector<double> bar_fock(square(n), 0.0);
  if (triples) {
    for (std::size_t i = 0; i < o; ++i) bar_fock[i * n + i] = triples->eps_o[i];
    for (std::size_t a = 0; a < v; ++a) bar_fock[(o + a) * n + o + a] = triples->eps_v[a];
    add_same_space_fock_seed(parameters, bar_fock, o, v);
    std::fill(bar_fock.begin(), bar_fock.end(), 0.0);
  }
  auto correlation = control_dispatch(parameters, 0.0);

  double minimum_same_space_gap = std::numeric_limits<double>::infinity();
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
  add_same_space_fock_seed(parameters, bar_fock, o, v);
  correlation = ControlWeights{};
  correlation = control_dispatch(parameters, 0.0);
  std::vector<double>().swap(bar_fock);
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
#if GENERATIVEQC_HAS_CUDA
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

  // The control response and dense curvature oracle are dead after the checked
  // physical Z solve. Release them before the one complete derivative VJP.
  correlation = ControlWeights{};
  std::vector<double>().swap(independent);
  std::vector<double>().swap(response_matrix);
  std::vector<double>().swap(basis);
  std::vector<double>().swap(action);
  std::vector<double>().swap(d_rotation);
  std::vector<double>().swap(orbital_arena);

  for (std::size_t index = 0; index < dimension; ++index)
    parameters.fov[index] -= z.solution[index];
  auto total = hamiltonian_dispatch(parameters, 1.0);
  auto total_eri = eri_dispatch(parameters, 1.0);
  const double stationarity = max_abs(total.stationarity);
  if (stationarity > kStationarityTolerance)
    throw std::runtime_error("complete HF+RCCSD(T)+Z orbital stationarity failed");

  mp2::LagrangianWeights weights;
  weights.orbitals = n;
  weights.occupied = o;
  weights.one_electron = std::move(total.hcore);
  weights.two_electron = std::move(total_eri);
  weights.overlap = std::move(total.overlap);
  weights.stationarity_residual = stationarity;
  const auto derivative_started = Clock::now();
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
  result.triples_response_pages = triples ? triples->pages : 0;
  result.numeric_capacity_bytes = resources.peak_bytes;
  result.raw_source_reads = raw_work.source_reads;
  result.raw_device_source_reads = raw_work.device_source_reads;
  result.raw_source_values = raw_work.source_values;
  result.raw_transform_fmas = raw_work.transform_fmas;
  result.raw_source_seconds = raw_work.source_seconds;
  result.raw_provider_seconds = raw_work.provider_seconds;
  result.triples_seconds = std::chrono::duration<double>(lambda_started - triples_started).count();
  result.lambda_parameter_seconds =
      std::chrono::duration<double>(raw_started - lambda_started).count();
  result.orbital_seconds =
      std::chrono::duration<double>(derivative_started - orbital_started).count();
  result.derivative_seconds =
      std::chrono::duration<double>(Clock::now() - derivative_started).count();
#if GENERATIVEQC_HAS_CUDA
  if (cuda_response) {
    // These owners execute serially; the public device high-water mark must
    // include triples scratch even when it exceeds the later response arena.
    result.response_owned_device_bytes =
        std::max(cuda_response->owned_device_bytes(), triples ? triples->device_capacity_bytes : 0);
    result.response_h2d_bytes = cuda_response->h2d_bytes();
    result.response_d2h_bytes = cuda_response->d2h_bytes();
    result.response_synchronizations = cuda_response->synchronizations();
    result.cuda_response_actions = true;
  }
#endif
  result.response_operator_hash = generated::orbital_jvp_program_hash;
  // Publish successful completed phase observations without adding GPU fences.
  // The five outer intervals are disjoint; raw provider/read observations are
  // nested inside raw_hamiltonian_ns. CUDA source reads only time submission.
  runtime::df_progress::Scope trace("relaxed_cc_force_completed");
  if (trace.enabled()) {
    using runtime::df_progress::Scope;
    const auto nanoseconds = [](double seconds) {
      return static_cast<std::uint64_t>(seconds * 1e9);
    };
    Scope::number("ao_functions", n);
    Scope::number("include_triples", include_triples);
    Scope::number("cuda_response_actions", result.cuda_response_actions);
    Scope::label("triples_response_backend",
                 include_triples ? (cuda_derivative ? "cuda" : "cpu") : "absent");
    Scope::number("triples_response_capacity_bytes", triples ? triples->numeric_capacity_bytes : 0);
    Scope::number("triples_response_device_bytes", triples ? triples->device_capacity_bytes : 0);
    Scope::number("triples_response_h2d_bytes", triples ? triples->host_to_device_bytes : 0);
    Scope::number("triples_response_d2h_bytes", triples ? triples->device_to_host_bytes : 0);
    Scope::number("triples_response_kernel_launches", triples ? triples->kernel_launches : 0);
    Scope::number("triples_response_ns", nanoseconds(result.triples_seconds));
    Scope::number("lambda_parameter_ns", nanoseconds(result.lambda_parameter_seconds));
    Scope::number("raw_hamiltonian_ns", std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            orbital_started - raw_started)
                                            .count());
    Scope::number("orbital_response_ns", nanoseconds(result.orbital_seconds));
    Scope::number("derivative_ns", nanoseconds(result.derivative_seconds));
    Scope::number("raw_provider_ns", nanoseconds(result.raw_provider_seconds));
    Scope::number("raw_source_read_ns", nanoseconds(result.raw_source_seconds));
    Scope::number("raw_source_reads", result.raw_source_reads);
    Scope::number("raw_device_source_reads", result.raw_device_source_reads);
    Scope::number("raw_source_values", result.raw_source_values);
    Scope::number("raw_transform_fmas", result.raw_transform_fmas);
    Scope::number("triples_response_pages", result.triples_response_pages);
    Scope::number("lambda_iterations", result.lambda.iterations);
    Scope::number("lambda_operator_actions", result.lambda.operator_actions);
    Scope::number("lambda_diagonal_preconditioned", result.lambda.diagonal_preconditioned);
    Scope::number("lambda_preconditioner_actions", result.lambda.preconditioner_actions);
    Scope::number("lambda_h2d_bytes", result.lambda.h2d_bytes);
    Scope::number("lambda_d2h_bytes", result.lambda.d2h_bytes);
    Scope::number("lambda_synchronizations", result.lambda.synchronizations);
    Scope::number("orbital_iterations", result.orbital_response.iterations);
    Scope::number("response_h2d_bytes", result.response_h2d_bytes);
    Scope::number("response_d2h_bytes", result.response_d2h_bytes);
    Scope::number("response_synchronizations", result.response_synchronizations);
    Scope::number("numeric_capacity_bytes", result.numeric_capacity_bytes);
  }
  return result;
}

RccsdtForceResult rccsd_force_cpu(const core::System& system,
                                  const integrals::ElectronInteractionSource& source,
                                  const hf::PhysicalReference& reference, const Problem& problem,
                                  const SolverResult& cc_result, std::span<const double> eps_o,
                                  std::span<const double> eps_v, std::size_t max_bytes) {
  return relaxed_rccsd_force_impl(system, source, reference, problem, cc_result, eps_o, eps_v,
                                  max_bytes, false, false, 0, 0, 1e-10);
}

RccsdtForceResult rccsd_force_cpu(const core::System& system,
                                  const hf::PhysicalReference& reference, const Problem& problem,
                                  const SolverResult& cc_result, std::span<const double> eps_o,
                                  std::span<const double> eps_v, std::size_t max_bytes) {
  posthf::RawSource source(system);
  return rccsd_force_cpu(system, source, reference, problem, cc_result, eps_o, eps_v, max_bytes);
}

RccsdtForceResult rccsd_force_cuda(const core::System& system,
                                   const integrals::ElectronInteractionSource& source,
                                   const hf::PhysicalReference& reference, const Problem& problem,
                                   const SolverResult& cc_result, std::span<const double> eps_o,
                                   std::span<const double> eps_v, std::size_t max_bytes,
                                   int device_id, std::size_t derivative_stage_budget) {
  return relaxed_rccsd_force_impl(system, source, reference, problem, cc_result, eps_o, eps_v,
                                  max_bytes, false, true, device_id, derivative_stage_budget,
                                  1e-10);
}

RccsdtForceResult rccsd_force_cuda(const core::System& system,
                                   const hf::PhysicalReference& reference, const Problem& problem,
                                   const SolverResult& cc_result, std::span<const double> eps_o,
                                   std::span<const double> eps_v, std::size_t max_bytes,
                                   int device_id, std::size_t derivative_stage_budget) {
  posthf::RawSource source(system);
  return rccsd_force_cuda(system, source, reference, problem, cc_result, eps_o, eps_v, max_bytes,
                          device_id, derivative_stage_budget);
}

RccsdtForceResult rccsdt_force_cpu(const core::System& system,
                                   const integrals::ElectronInteractionSource& source,
                                   const hf::PhysicalReference& reference, const Problem& problem,
                                   const SolverResult& cc_result, std::span<const double> eps_o,
                                   std::span<const double> eps_v, std::size_t max_bytes,
                                   double denominator_threshold) {
  return relaxed_rccsd_force_impl(system, source, reference, problem, cc_result, eps_o, eps_v,
                                  max_bytes, true, false, 0, 0, denominator_threshold);
}

RccsdtForceResult rccsdt_force_cpu(const core::System& system,
                                   const hf::PhysicalReference& reference, const Problem& problem,
                                   const SolverResult& cc_result, std::span<const double> eps_o,
                                   std::span<const double> eps_v, std::size_t max_bytes,
                                   double denominator_threshold) {
  posthf::RawSource source(system);
  return rccsdt_force_cpu(system, source, reference, problem, cc_result, eps_o, eps_v, max_bytes,
                          denominator_threshold);
}

RccsdtForceResult rccsdt_force_cuda(const core::System& system,
                                    const integrals::ElectronInteractionSource& source,
                                    const hf::PhysicalReference& reference, const Problem& problem,
                                    const SolverResult& cc_result, std::span<const double> eps_o,
                                    std::span<const double> eps_v, std::size_t max_bytes,
                                    int device_id, std::size_t derivative_stage_budget,
                                    double denominator_threshold) {
  return relaxed_rccsd_force_impl(system, source, reference, problem, cc_result, eps_o, eps_v,
                                  max_bytes, true, true, device_id, derivative_stage_budget,
                                  denominator_threshold);
}

RccsdtForceResult rccsdt_force_cuda(const core::System& system,
                                    const hf::PhysicalReference& reference, const Problem& problem,
                                    const SolverResult& cc_result, std::span<const double> eps_o,
                                    std::span<const double> eps_v, std::size_t max_bytes,
                                    int device_id, std::size_t derivative_stage_budget,
                                    double denominator_threshold) {
  posthf::RawSource source(system);
  return rccsdt_force_cuda(system, source, reference, problem, cc_result, eps_o, eps_v, max_bytes,
                           device_id, derivative_stage_budget, denominator_threshold);
}

}  // namespace generativeqc::cc
