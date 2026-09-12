#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "integrals/ecp.hpp"
#include "integrals/ecp_cuda.hpp"
#include "molecule/basis.hpp"

extern "C" {
vibeqc_status vibeqc_system_create_ecp(vibeqc_context* context,
                                       const vibeqc_system_descriptor* descriptor,
                                       const int32_t* core_electrons, const vibeqc_ecp_term* terms,
                                       size_t term_count, vibeqc_system** system) {
  if (system) *system = nullptr;
  if (!context || !descriptor || !system || !core_electrons || !terms || term_count == 0)
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  if (!vibeqc::api::valid_descriptor(descriptor)) return VIBEQC_STATUS_ABI_MISMATCH;
  if (!descriptor->atoms || !descriptor->shells || descriptor->atom_count > 128 ||
      term_count > 4096)
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  // The nested system constructor and error detail share this context lock.
  std::lock_guard<std::recursive_mutex> context_lock(context->mutex);
  try {
    std::int64_t removed = 0;
    std::vector<bool> local(descriptor->atom_count, false), present(descriptor->atom_count, false);
    for (unsigned a = 0; a < descriptor->atom_count; ++a) {
      if (core_electrons[a] < 0 || core_electrons[a] >= descriptor->atoms[a].atomic_number)
        throw std::invalid_argument("ECP core count must leave positive effective ionic charge");
      removed += core_electrons[a];
    }
    for (unsigned s = 0; s < descriptor->shell_count; ++s)
      if (descriptor->shells[s].angular_momentum > 2)
        throw std::invalid_argument("scalar ECP execution currently supports orbital s/p/d shells");
    std::vector<vibeqc::core::EcpTerm> owned;
    for (std::size_t i = 0; i < term_count; ++i) {
      const auto& t = terms[i];
      if (t.atom_index >= descriptor->atom_count || core_electrons[t.atom_index] == 0 ||
          t.channel < -1 || t.channel > 2 || t.power > 4 || !std::isfinite(t.exponent) ||
          t.exponent <= 0 || !std::isfinite(t.coefficient))
        throw std::invalid_argument("unsupported or invalid scalar Gaussian ECP term");
      local[t.atom_index] = local[t.atom_index] || t.channel == -1;
      present[t.atom_index] = true;
      owned.push_back({t.atom_index, t.channel, t.power, t.exponent, t.coefficient});
    }
    for (unsigned a = 0; a < descriptor->atom_count; ++a)
      if ((core_electrons[a] != 0) != present[a] || (present[a] && !local[a]))
        throw std::invalid_argument("every ECP core count requires a local potential");
    const auto adjusted_charge = static_cast<std::int64_t>(descriptor->charge) + removed;
    if (adjusted_charge < std::numeric_limits<int32_t>::min() ||
        adjusted_charge > std::numeric_limits<int32_t>::max())
      throw std::invalid_argument("ECP electron bookkeeping overflows native charge range");
    auto adjusted = *descriptor;
    adjusted.charge = static_cast<int32_t>(adjusted_charge);
    vibeqc_system* created = nullptr;
    const auto status = vibeqc_system_create(context, &adjusted, &created);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    std::unique_ptr<vibeqc_system> candidate(created);
    candidate->data.charge = descriptor->charge;
    for (unsigned a = 0; a < descriptor->atom_count; ++a)
      candidate->data.atoms[a].ecp_core = core_electrons[a];
    candidate->data.ecp_terms = std::move(owned);
    *system = candidate.release();
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&context->last_detail);
  }
}

vibeqc_status vibeqc_system_ecp_integrals(vibeqc_context* context, const vibeqc_system* system,
                                          uint32_t radial_points, uint32_t polar_points,
                                          int32_t derivatives, double* output,
                                          size_t output_count) {
  if (!context || !system || !output || (derivatives != 0 && derivatives != 1))
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> context_lock(context->mutex);
  try {
    const auto n = vibeqc::molecule::ao_count(system->data);
    const auto count = 2 * n * n * (1 + (derivatives ? 3 * system->data.atoms.size() : 0));
    if (output_count != count) throw std::invalid_argument("ECP output buffer size mismatch");
    vibeqc::integrals::EcpData result;
    if (context->state.executed_backend == VIBEQC_BACKEND_CUDA && !system->data.ecp_terms.empty()) {
      const auto status = vibeqc::integrals::ecp_integrals_cuda(
          context->state.device_id, system->data, radial_points, polar_points, derivatives != 0,
          result, context->last_detail);
      if (status != VIBEQC_STATUS_SUCCESS) return status;
    } else {
      result = vibeqc::integrals::ecp_integrals(system->data, radial_points, polar_points,
                                                derivatives != 0);
    }
    auto cursor = std::copy(result.local.begin(), result.local.end(), output);
    cursor = std::copy(result.local_derivative.begin(), result.local_derivative.end(), cursor);
    cursor = std::copy(result.nonlocal.begin(), result.nonlocal.end(), cursor);
    std::copy(result.nonlocal_derivative.begin(), result.nonlocal_derivative.end(), cursor);
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&context->last_detail);
  }
}
}
