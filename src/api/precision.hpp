#ifndef VIBEQC_API_PRECISION_HPP
#define VIBEQC_API_PRECISION_HPP

#include "api/error.hpp"
#include "scf/types.hpp"

namespace vibeqc::api {

/** Share descriptor validation and copy-out across single and batched queries. */
inline vibeqc_status copy_precision_provenance(const scf::PrecisionProvenance& source,
                                               vibeqc_precision_provenance* out) {
  if (out == nullptr) return VIBEQC_STATUS_SUCCESS;
  if (!valid_descriptor(out)) return VIBEQC_STATUS_ABI_MISMATCH;
  out->struct_size = sizeof(vibeqc_precision_provenance);
  out->abi_version = VIBEQC_ABI_VERSION;
  out->policy_version = source.policy_version;
  out->requested_mode = source.requested_mode;
  out->effective_bits = source.effective_bits;
  out->mixed_precision_fock_threshold = source.mixed_precision_fock_threshold;
  out->strict_refinement_applied = source.strict_refinement_applied ? 1 : 0;
  out->mixed_precision_reserved_error = source.mixed_precision_reserved_error;
  out->refinement_iterations = static_cast<int32_t>(source.refinement_iterations);
  return VIBEQC_STATUS_SUCCESS;
}

}  // namespace vibeqc::api

#endif
