#include "posthf/cuda_derivative.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "scf/cuda_weighted_eri.hpp"

namespace vibeqc::posthf {

vibeqc_status contract_weighted_eri_shell_derivative_cuda(
    int device_id, const core::System& system, const std::array<std::size_t, 4>& shell_indices,
    std::span<const double> weights, std::size_t stage_budget,
    std::array<double, 12>& center_gradient, std::string& detail) {
#if !VIBEQC_HAS_CUDA
  (void)device_id;
  (void)system;
  (void)shell_indices;
  (void)weights;
  (void)stage_budget;
  (void)center_gradient;
  detail = "CUDA weighted ERI shell derivatives are unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#else
  detail.clear();
  if (device_id < 0 || !stage_budget) {
    detail = "invalid weighted ERI shell-gradient request";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  try {
    std::array<const core::Shell*, 4> selected{};
    std::size_t expected = 1;
    std::size_t expansion_terms = 0;
    for (std::size_t slot = 0; slot < selected.size(); ++slot) {
      if (shell_indices[slot] >= system.shells.size()) {
        detail = "weighted ERI shell index is out of range";
        return VIBEQC_STATUS_INVALID_ARGUMENT;
      }
      selected[slot] = &system.shells[shell_indices[slot]];
      if (selected[slot]->atom_index >= system.atoms.size()) {
        detail = "weighted ERI shell atom is out of range";
        return VIBEQC_STATUS_INVALID_ARGUMENT;
      }
      const auto cartesian = molecule::cartesian_count(selected[slot]->angular_momentum);
      const auto public_count = system.basis_representation == VIBEQC_BASIS_SPHERICAL
                                    ? 2 * selected[slot]->angular_momentum + 1
                                    : cartesian;
      expected = checked_mul(expected, public_count);
      expansion_terms = checked_add(expansion_terms, checked_mul(public_count, cartesian));
    }
    if (weights.size() != expected ||
        !std::all_of(weights.begin(), weights.end(),
                     [](double value) { return std::isfinite(value); })) {
      detail = "weighted ERI shell weights have the wrong shape or are nonfinite";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }

    constexpr std::size_t record_bytes = sizeof(scf::CudaWeightedEriPrimitive);
    constexpr std::size_t result_bytes = 2 * sizeof(scf::CudaWeightedEriResult);
    auto fixed_bytes = checked_mul(std::size_t{12}, sizeof(double));
    fixed_bytes = checked_add(
        fixed_bytes, checked_mul(expansion_terms, sizeof(molecule::CartesianExpansionTerm)));
    if (stage_budget <= checked_add(fixed_bytes, result_bytes)) {
      detail = "weighted ERI shell-gradient stage budget is too small";
      return VIBEQC_STATUS_OUT_OF_MEMORY;
    }
    const auto capacity = std::min<std::size_t>(
        4096, (stage_budget - fixed_bytes - result_bytes) / (2 * record_bytes));
    if (!capacity) {
      detail = "weighted ERI shell-gradient cannot hold one record";
      return VIBEQC_STATUS_OUT_OF_MEMORY;
    }
    const auto record_storage = checked_mul(capacity, record_bytes);
    const auto consumer_budget = stage_budget - fixed_bytes - record_storage;
    const std::array<std::vector<molecule::AoExpansion>, 4> expansions{
        molecule::ao_expansions(selected[0]->angular_momentum, system.basis_representation),
        molecule::ao_expansions(selected[1]->angular_momentum, system.basis_representation),
        molecule::ao_expansions(selected[2]->angular_momentum, system.basis_representation),
        molecule::ao_expansions(selected[3]->angular_momentum, system.basis_representation)};
    std::vector<scf::CudaWeightedEriPrimitive> records;
    records.reserve(capacity);
    std::vector<scf::CudaWeightedEriResult> output;
    std::array<double, 12> candidate{};
    vibeqc_status status = VIBEQC_STATUS_SUCCESS;
    auto flush = [&] {
      if (records.empty() || status != VIBEQC_STATUS_SUCCESS) return;
      scf::CudaWeightedEriDiagnostic diagnostic;
      status = scf::contract_cuda_weighted_eri_primitives(device_id, records.data(), records.size(),
                                                          1, consumer_budget, false, output,
                                                          diagnostic, detail);
      if (status != VIBEQC_STATUS_SUCCESS) return;
      if (output.size() != 1) {
        detail = "weighted ERI shell contraction returned no result";
        status = VIBEQC_STATUS_NUMERICAL_FAILURE;
        return;
      }
      for (std::size_t slot = 0; slot < 4; ++slot)
        for (std::size_t axis = 0; axis < 3; ++axis)
          candidate[3 * slot + axis] += output[0].center[slot][axis];
      records.clear();
    };
    for (std::size_t i = 0; i < expansions[0].size(); ++i)
      for (std::size_t j = 0; j < expansions[1].size(); ++j)
        for (std::size_t k = 0; k < expansions[2].size(); ++k)
          for (std::size_t l = 0; l < expansions[3].size(); ++l) {
            const auto weight =
                weights[((i * expansions[1].size() + j) * expansions[2].size() + k) *
                            expansions[3].size() +
                        l];
            if (weight == 0.0) continue;
            for (const auto& ei : expansions[0][i])
              for (const auto& ej : expansions[1][j])
                for (const auto& ek : expansions[2][k])
                  for (const auto& el : expansions[3][l]) {
                    const std::array<const molecule::CartesianExpansionTerm*, 4> terms{&ei, &ej,
                                                                                       &ek, &el};
                    double component_weight = weight;
                    for (const auto* term : terms)
                      component_weight *=
                          term->coefficient *
                          molecule::cartesian_component_normalization(term->component);
                    for (const auto& pi : selected[0]->primitives)
                      for (const auto& pj : selected[1]->primitives)
                        for (const auto& pk : selected[2]->primitives)
                          for (const auto& pl : selected[3]->primitives) {
                            if (records.size() == capacity) flush();
                            if (status != VIBEQC_STATUS_SUCCESS) return status;
                            auto& record = records.emplace_back();
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
                          }
                  }
          }
    flush();
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    if (!std::all_of(candidate.begin(), candidate.end(),
                     [](double value) { return std::isfinite(value); })) {
      detail = "weighted ERI shell gradient is nonfinite";
      return VIBEQC_STATUS_NUMERICAL_FAILURE;
    }
    center_gradient = candidate;
    return VIBEQC_STATUS_SUCCESS;
  } catch (const std::overflow_error& error) {
    detail = error.what();
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::bad_alloc&) {
    detail = "weighted ERI shell-gradient allocation failed";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::exception& error) {
    detail = error.what();
    return VIBEQC_STATUS_NUMERICAL_FAILURE;
  }
#endif
}

}  // namespace vibeqc::posthf
