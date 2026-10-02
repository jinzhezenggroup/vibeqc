#include "posthf/cuda_derivative.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "scf/cuda_weighted_eri.hpp"

namespace generativeqc::posthf {

#if GENERATIVEQC_HAS_CUDA
namespace {
// Shared public-basis normalization and primitive record producer. Both the
// synchronous one-shell API and streamed batches retain this exact algebra.
generativeqc_status validate_shell(const core::System& system,
                                   const std::array<std::size_t, 4>& shell_indices,
                                   std::span<const double> weights,
                                   std::array<const core::Shell*, 4>& selected,
                                   std::size_t& expansion_terms, std::string& detail) {
  std::size_t expected = 1;
  expansion_terms = 0;
  for (std::size_t slot = 0; slot < selected.size(); ++slot) {
    if (shell_indices[slot] >= system.shells.size()) {
      detail = "weighted ERI shell index is out of range";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    selected[slot] = &system.shells[shell_indices[slot]];
    if (selected[slot]->atom_index >= system.atoms.size()) {
      detail = "weighted ERI shell atom is out of range";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    const auto cartesian = molecule::cartesian_count(selected[slot]->angular_momentum);
    const auto public_count = system.basis_representation == GENERATIVEQC_BASIS_SPHERICAL
                                  ? 2 * selected[slot]->angular_momentum + 1
                                  : cartesian;
    expected = checked_mul(expected, public_count);
    expansion_terms = checked_add(expansion_terms, checked_mul(public_count, cartesian));
  }
  if (weights.size() != expected || !std::all_of(weights.begin(), weights.end(), [](double value) {
        return std::isfinite(value);
      })) {
    detail = "weighted ERI shell weights have the wrong shape or are nonfinite";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  return GENERATIVEQC_STATUS_SUCCESS;
}

template <class Emit>
generativeqc_status emit_shell_records(const core::System& system,
                                       const std::array<const core::Shell*, 4>& selected,
                                       std::span<const double> weights, Emit&& emit) {
  const std::array<std::vector<molecule::AoExpansion>, 4> expansions{
      molecule::ao_expansions(selected[0]->angular_momentum, system.basis_representation),
      molecule::ao_expansions(selected[1]->angular_momentum, system.basis_representation),
      molecule::ao_expansions(selected[2]->angular_momentum, system.basis_representation),
      molecule::ao_expansions(selected[3]->angular_momentum, system.basis_representation)};
  for (std::size_t i = 0; i < expansions[0].size(); ++i)
    for (std::size_t j = 0; j < expansions[1].size(); ++j)
      for (std::size_t k = 0; k < expansions[2].size(); ++k)
        for (std::size_t l = 0; l < expansions[3].size(); ++l) {
          const auto weight = weights[((i * expansions[1].size() + j) * expansions[2].size() + k) *
                                          expansions[3].size() +
                                      l];
          if (weight == 0.0) continue;
          for (const auto& ei : expansions[0][i])
            for (const auto& ej : expansions[1][j])
              for (const auto& ek : expansions[2][k])
                for (const auto& el : expansions[3][l]) {
                  const std::array<const molecule::CartesianExpansionTerm*, 4> terms{&ei, &ej, &ek,
                                                                                     &el};
                  double component_weight = weight;
                  for (const auto* term : terms)
                    component_weight *=
                        term->coefficient *
                        molecule::cartesian_component_normalization(term->component);
                  for (const auto& pi : selected[0]->primitives)
                    for (const auto& pj : selected[1]->primitives)
                      for (const auto& pk : selected[2]->primitives)
                        for (const auto& pl : selected[3]->primitives) {
                          scf::CudaWeightedEriPrimitive record{};
                          record.kind = 0;
                          const std::array<const core::Primitive*, 4> primitives{&pi, &pj, &pk,
                                                                                 &pl};
                          for (std::size_t slot = 0; slot < 4; ++slot) {
                            for (std::size_t axis = 0; axis < 3; ++axis) {
                              record.angular[slot][axis] = terms[slot]->component[axis];
                              record.centers[slot][axis] =
                                  system.atoms[selected[slot]->atom_index].position[axis];
                            }
                            record.exponents[slot] = primitives[slot]->exponent;
                          }
                          record.weights[0] = component_weight * pi.coefficient * pj.coefficient *
                                              pk.coefficient * pl.coefficient;
                          const auto status = emit(record);
                          if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
                        }
                }
        }
  return GENERATIVEQC_STATUS_SUCCESS;
}

void check_consumer(generativeqc_status status, const std::string& detail) {
  if (status == GENERATIVEQC_STATUS_SUCCESS) return;
  if (status == GENERATIVEQC_STATUS_OUT_OF_MEMORY) throw std::length_error(detail);
  if (status == GENERATIVEQC_STATUS_INVALID_ARGUMENT) throw std::invalid_argument(detail);
  throw std::runtime_error(detail);
}
}  // namespace
#endif

generativeqc_status contract_weighted_eri_shell_derivative_cuda(
    int device_id, const core::System& system, const std::array<std::size_t, 4>& shell_indices,
    std::span<const double> weights, std::size_t stage_budget,
    std::array<double, 12>& center_gradient, std::string& detail,
    CudaShellDerivativeDiagnostic* diagnostic) {
  if (diagnostic) *diagnostic = {};
#if !GENERATIVEQC_HAS_CUDA
  (void)device_id;
  (void)system;
  (void)shell_indices;
  (void)weights;
  (void)stage_budget;
  (void)center_gradient;
  detail = "CUDA weighted ERI shell derivatives are unavailable in this build";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
#else
  detail.clear();
  if (device_id < 0 || !stage_budget) {
    detail = "invalid weighted ERI shell-gradient request";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  try {
    std::array<const core::Shell*, 4> selected{};
    std::size_t expansion_terms = 0;
    const auto validation =
        validate_shell(system, shell_indices, weights, selected, expansion_terms, detail);
    if (validation != GENERATIVEQC_STATUS_SUCCESS) return validation;

    constexpr std::size_t record_bytes = sizeof(scf::CudaWeightedEriPrimitive);
    constexpr std::size_t result_bytes = 2 * sizeof(scf::CudaWeightedEriResult);
    auto fixed_bytes = checked_add(checked_mul(std::size_t{12}, sizeof(double)), record_bytes);
    fixed_bytes = checked_add(
        fixed_bytes, checked_mul(expansion_terms, sizeof(molecule::CartesianExpansionTerm)));
    if (stage_budget <= checked_add(fixed_bytes, result_bytes)) {
      detail = "weighted ERI shell-gradient stage budget is too small";
      return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
    }
    const auto capacity = std::min<std::size_t>(
        4096, (stage_budget - fixed_bytes - result_bytes) / (2 * record_bytes));
    if (!capacity) {
      detail = "weighted ERI shell-gradient cannot hold one record";
      return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
    }
    const auto record_storage = checked_mul(capacity, record_bytes);
    const auto consumer_budget = stage_budget - fixed_bytes - record_storage;
    std::vector<scf::CudaWeightedEriPrimitive> records;
    records.reserve(capacity);
    std::vector<scf::CudaWeightedEriResult> output;
    std::array<double, 12> candidate{};
    generativeqc_status status = GENERATIVEQC_STATUS_SUCCESS;
    auto flush = [&] {
      if (records.empty() || status != GENERATIVEQC_STATUS_SUCCESS) return;
      scf::CudaWeightedEriDiagnostic consumer_diagnostic;
      using Clock = std::chrono::steady_clock;
      const auto started = diagnostic ? Clock::now() : Clock::time_point{};
      status = scf::contract_cuda_weighted_eri_primitives(device_id, records.data(), records.size(),
                                                          1, consumer_budget, false, output,
                                                          consumer_diagnostic, detail);
      if (status != GENERATIVEQC_STATUS_SUCCESS) return;
      if (diagnostic) {
        diagnostic->primitive_records += records.size();
        ++diagnostic->consumer_calls;
        diagnostic->consumer_nanoseconds +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started).count();
      }
      if (output.size() != 1) {
        detail = "weighted ERI shell contraction returned no result";
        status = GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
        return;
      }
      for (std::size_t slot = 0; slot < 4; ++slot)
        for (std::size_t axis = 0; axis < 3; ++axis)
          candidate[3 * slot + axis] += output[0].center[slot][axis];
      records.clear();
    };
    status = emit_shell_records(system, selected, weights,
                                [&](const auto& record) -> generativeqc_status {
                                  if (records.size() == capacity) flush();
                                  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
                                  records.push_back(record);
                                  return GENERATIVEQC_STATUS_SUCCESS;
                                });
    flush();
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    if (!std::all_of(candidate.begin(), candidate.end(),
                     [](double value) { return std::isfinite(value); })) {
      detail = "weighted ERI shell gradient is nonfinite";
      return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
    }
    center_gradient = candidate;
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::overflow_error& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::bad_alloc&) {
    detail = "weighted ERI shell-gradient allocation failed";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::exception& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  }
#endif
}

struct CudaEriDerivativeBatch::Impl {
  int device_id;
  const core::System& system;
  std::size_t stage_budget, budget, fixed_bytes{}, record_capacity{}, tile_capacity{};
  std::size_t planned_bytes{};
  bool trace;
  CudaShellDerivativeDiagnostic diagnostic;
#if GENERATIVEQC_HAS_CUDA
  std::vector<scf::CudaWeightedEriPrimitive> records;
  std::vector<std::array<std::size_t, 4>> atoms;

  void validate_gradient(std::span<double> gradient) const {
    if (gradient.size() != checked_mul(3, system.atoms.size()))
      throw std::invalid_argument("CUDA ERI batch gradient shape mismatch");
  }

  void flush(std::span<double> gradient) {
    if (records.empty()) return;
    // Charge actual retained host capacities, the current shell's expansion,
    // and its one producer record before admitting the device consumer.
    const auto host = checked_add(checked_mul(records.capacity(), sizeof(records[0])),
                                  checked_mul(atoms.capacity(), sizeof(atoms[0])));
    if (checked_add(fixed_bytes, host) > budget)
      throw std::length_error("CUDA ERI batch host capacity exceeds its stage budget");
    const auto consumer_budget = budget - fixed_bytes - host;
    std::vector<scf::CudaWeightedEriResult> output;
    scf::CudaWeightedEriDiagnostic consumer;
    std::string detail;
    using Clock = std::chrono::steady_clock;
    const auto started = trace ? Clock::now() : Clock::time_point{};
    const auto status = scf::contract_cuda_weighted_eri_primitives(
        device_id, records.data(), records.size(), atoms.size(), consumer_budget, false, output,
        consumer, detail);
    check_consumer(status, detail);
    diagnostic.primitive_records += records.size();
    ++diagnostic.consumer_calls;
    if (trace)
      diagnostic.consumer_nanoseconds +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started).count();
    if (output.size() != atoms.size())
      throw std::runtime_error("CUDA ERI batch returned an incomplete tile set");
    for (std::size_t tile = 0; tile < atoms.size(); ++tile)
      for (std::size_t slot = 0; slot < 4; ++slot)
        for (std::size_t axis = 0; axis < 3; ++axis)
          gradient[3 * atoms[tile][slot] + axis] += output[tile].center[slot][axis];
    records.clear();
    atoms.clear();
  }
#endif

  Impl(int device, const core::System& source, std::size_t stage, bool tracing)
      : device_id(device),
        system(source),
        stage_budget(stage),
        budget(std::min(stage, source_scratch_bytes)),
        trace(tracing) {
    if (device_id < 0 || !stage_budget)
      throw std::invalid_argument("invalid CUDA ERI derivative batch request");
#if GENERATIVEQC_HAS_CUDA
    std::size_t max_terms = 0;
    for (const auto& shell : system.shells) {
      // The retained primitive consumer validates through f. Leave any other
      // regime on the existing shell path, including its established errors.
      if (shell.angular_momentum > 3) return;
      const auto cart = molecule::cartesian_count(shell.angular_momentum);
      const auto pub = system.basis_representation == GENERATIVEQC_BASIS_SPHERICAL
                           ? 2 * shell.angular_momentum + 1
                           : cart;
      max_terms = std::max(max_terms, checked_mul(pub, cart));
    }
    fixed_bytes = checked_add(
        sizeof(scf::CudaWeightedEriPrimitive),
        checked_mul(checked_mul(4, max_terms), sizeof(molecule::CartesianExpansionTerm)));
    if (budget <= fixed_bytes) return;
    const auto available = budget - fixed_bytes;
    constexpr auto tile_bytes =
        sizeof(std::array<std::size_t, 4>) + 2 * sizeof(scf::CudaWeightedEriResult);
    tile_capacity = std::min<std::size_t>(256, available / 8 / tile_bytes);
    if (tile_capacity < 2) {
      tile_capacity = 0;
      return;
    }
    record_capacity = std::min<std::size_t>(16384, (available - tile_capacity * tile_bytes) /
                                                       (2 * sizeof(scf::CudaWeightedEriPrimitive)));
    if (!record_capacity) {
      tile_capacity = 0;
      return;
    }
    planned_bytes = fixed_bytes + tile_capacity * tile_bytes +
                    2 * record_capacity * sizeof(scf::CudaWeightedEriPrimitive);
#else
    throw std::runtime_error("CUDA ERI derivative batches are unavailable in this build");
#endif
  }
};

CudaEriDerivativeBatch::CudaEriDerivativeBatch(int device_id, const core::System& system,
                                               std::size_t stage_budget, bool trace)
    : impl_(std::make_unique<Impl>(device_id, system, stage_budget, trace)) {}
CudaEriDerivativeBatch::~CudaEriDerivativeBatch() = default;
const CudaShellDerivativeDiagnostic& CudaEriDerivativeBatch::diagnostic() const {
  return impl_->diagnostic;
}
bool CudaEriDerivativeBatch::batched() const { return impl_->record_capacity != 0; }
std::size_t CudaEriDerivativeBatch::numeric_capacity_bytes() const { return impl_->planned_bytes; }

void CudaEriDerivativeBatch::append(const std::array<std::size_t, 4>& shells,
                                    std::span<const double> weights, std::span<double> gradient) {
#if GENERATIVEQC_HAS_CUDA
  auto& state = *impl_;
  state.validate_gradient(gradient);
  std::string detail;
  if (!batched()) {
    std::array<double, 12> center{};
    CudaShellDerivativeDiagnostic diagnostic;
    check_consumer(contract_weighted_eri_shell_derivative_cuda(
                       state.device_id, state.system, shells, weights, state.stage_budget, center,
                       detail, state.trace ? &diagnostic : nullptr),
                   detail);
    state.diagnostic.primitive_records += diagnostic.primitive_records;
    state.diagnostic.consumer_calls += diagnostic.consumer_calls;
    state.diagnostic.consumer_nanoseconds += diagnostic.consumer_nanoseconds;
    for (std::size_t slot = 0; slot < 4; ++slot)
      for (std::size_t axis = 0; axis < 3; ++axis)
        gradient[3 * state.system.shells[shells[slot]].atom_index + axis] +=
            center[3 * slot + axis];
    return;
  }
  std::array<const core::Shell*, 4> selected{};
  std::size_t expansion_terms = 0;
  check_consumer(validate_shell(state.system, shells, weights, selected, expansion_terms, detail),
                 detail);
  if (state.records.capacity() == 0) {
    // Allocate only during the first two-electron callback. The one-electron
    // consumer has already completed, so its scratch does not coexist here.
    state.records.reserve(state.record_capacity);
    state.atoms.reserve(state.tile_capacity);
  }
  std::size_t tile = 0;
  bool has_tile = false;
  const auto status = emit_shell_records(state.system, selected, weights, [&](auto& record) {
    if (state.records.size() == state.record_capacity ||
        (!has_tile && state.atoms.size() == state.tile_capacity)) {
      state.flush(gradient);
      has_tile = false;
    }
    if (!has_tile) {
      tile = state.atoms.size();
      state.atoms.push_back({selected[0]->atom_index, selected[1]->atom_index,
                             selected[2]->atom_index, selected[3]->atom_index});
      has_tile = true;
    }
    record.output_tile = static_cast<std::uint32_t>(tile);
    state.records.push_back(record);
    return GENERATIVEQC_STATUS_SUCCESS;
  });
  check_consumer(status, detail);
#else
  (void)shells;
  (void)weights;
  (void)gradient;
  throw std::runtime_error("CUDA ERI derivative batches are unavailable in this build");
#endif
}

void CudaEriDerivativeBatch::finish(std::span<double> gradient) {
#if GENERATIVEQC_HAS_CUDA
  impl_->validate_gradient(gradient);
  impl_->flush(gradient);
  if (!std::all_of(gradient.begin(), gradient.end(), [](double x) { return std::isfinite(x); }))
    throw std::runtime_error("CUDA ERI derivative batch produced a nonfinite gradient");
#else
  (void)gradient;
  throw std::runtime_error("CUDA ERI derivative batches are unavailable in this build");
#endif
}

}  // namespace generativeqc::posthf
