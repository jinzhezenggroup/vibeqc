#ifndef VIBEQC_XTB_RUNTIME_MOLECULAR_REQUEST_HPP
#define VIBEQC_XTB_RUNTIME_MOLECULAR_REQUEST_HPP

#include <string>

#include "runtime/types.hpp"

namespace vibeqc::xtb::detail {

// The native adapter serves VibeQC's molecular energy/force endpoint. The
// imported periodic/solvation/interaction APIs were never exposed by that
// endpoint. Reject their descriptors before staging, so retiring an owner
// cannot silently drop an interaction from the Hamiltonian or energy.
inline vibeqc_xtb_status_t validate_molecular_request(const vibeqc_xtb_batch_t& batch,
                                                      const vibeqc_xtb_compute_options_t& options,
                                                      std::string& error) {
  constexpr auto supported = VIBEQC_XTB_COMPUTE_ENERGY | VIBEQC_XTB_COMPUTE_FORCES;
  if (batch.batch_size != 1 || options.model != VIBEQC_XTB_MODEL_GFN2_XTB ||
      options.scc_start_mode != VIBEQC_XTB_SCC_START_FRESH || (options.flags & ~supported) != 0u ||
      batch.total_point_charges != 0 || batch.total_charge_response_elements != 0 ||
      batch.total_interactions != 0 || batch.point_charge_offsets.data != nullptr ||
      batch.point_charge_offsets.size_bytes != 0u || batch.point_charge_positions.data != nullptr ||
      batch.point_charge_positions.size_bytes != 0u || batch.point_charge_values.data != nullptr ||
      batch.point_charge_values.size_bytes != 0u || batch.point_charge_gammas.data != nullptr ||
      batch.point_charge_gammas.size_bytes != 0u || batch.charge_response_offsets.data != nullptr ||
      batch.charge_response_offsets.size_bytes != 0u || batch.cell_matrices.data != nullptr ||
      batch.cell_matrices.size_bytes != 0u || batch.periodic_axes.data != nullptr ||
      batch.periodic_axes.size_bytes != 0u || batch.atomic_potential_shifts.data != nullptr ||
      batch.atomic_potential_shifts.size_bytes != 0u ||
      batch.charge_response_matrix.data != nullptr ||
      batch.charge_response_matrix.size_bytes != 0u ||
      batch.interaction_descriptors.data != nullptr ||
      batch.interaction_descriptors.size_bytes != 0u || batch.interaction_payload.data != nullptr ||
      batch.interaction_payload.size_bytes != 0u) {
    error = "native GFN2 supports a single molecular energy/force request with fresh SCC";
    return VIBEQC_XTB_STATUS_NOT_SUPPORTED;
  }
  return VIBEQC_XTB_STATUS_SUCCESS;
}

}  // namespace vibeqc::xtb::detail
#endif
