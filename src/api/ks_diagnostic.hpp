#pragma once

#include <cstdint>
#include <limits>
#include <optional>

#include "api/error.hpp"
#include "dft/scf_diagnostic.hpp"

namespace vibeqc::api {

/** Copy only validated caller-owned descriptors, without allocating another
 * history or changing the method's numerical state. Validate every row before
 * writing any output so a malformed tail row cannot publish a partial result. */
inline vibeqc_status copy_ks_diagnostic(const std::optional<dft::ScfDiagnostic>& source,
                                        vibeqc_ks_diagnostic* out, vibeqc_ks_iteration* history,
                                        std::uint32_t history_capacity) {
  if (out && !valid_descriptor(out)) return VIBEQC_STATUS_ABI_MISMATCH;
  if (!history && history_capacity) return VIBEQC_STATUS_INVALID_ARGUMENT;
  if (!source) return VIBEQC_STATUS_NOT_IMPLEMENTED;
  const auto& value = *source;
  if (value.history.size() > std::numeric_limits<std::uint32_t>::max())
    return VIBEQC_STATUS_INTERNAL_ERROR;
  if (history) {
    if (history_capacity < value.history.size()) return VIBEQC_STATUS_INVALID_ARGUMENT;
    for (std::size_t i = 0; i < value.history.size(); ++i)
      if (!valid_descriptor(history + i)) return VIBEQC_STATUS_ABI_MISMATCH;
  }
  if (out) {
    *out = {sizeof(*out),
            VIBEQC_ABI_VERSION,
            value.scf_domain_version,
            static_cast<std::uint32_t>(value.ao_order),
            static_cast<std::uint32_t>(value.history.size()),
            value.initial_density_used ? 1 : 0,
            {value.occupations[0], value.occupations[1]},
            value.grid_points,
            value.tile_points,
            value.fock_builds,
            {value.electrons[0], value.electrons[1]},
            value.components.nuclear,
            value.components.one_electron,
            value.components.hartree,
            value.components.xc + value.components.exact_exchange,
            value.density_change,
            value.physical_residual};
  }
  if (history)
    for (std::size_t i = 0; i < value.history.size(); ++i) {
      const auto& row = value.history[i];
      history[i] = {sizeof(*history),       VIBEQC_ABI_VERSION,
                    row.iteration,          row.occupation_stabilized ? 1 : 0,
                    row.components.nuclear, row.components.one_electron,
                    row.components.hartree, row.components.xc + row.components.exact_exchange,
                    row.energy_change,      row.density_change,
                    row.physical_residual,  {row.electrons[0], row.electrons[1]}};
    }
  return VIBEQC_STATUS_SUCCESS;
}
}  // namespace vibeqc::api
