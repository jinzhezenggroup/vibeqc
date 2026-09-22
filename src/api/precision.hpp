#ifndef VIBEQC_API_PRECISION_HPP
#define VIBEQC_API_PRECISION_HPP

#include <algorithm>
#include <cstddef>
#include <cstring>

#include "api/error.hpp"
#include "scf/types.hpp"

namespace vibeqc::api {

/** Share descriptor validation and copy-out across single and batched queries. */
inline vibeqc_status copy_precision_provenance(const scf::PrecisionProvenance& source,
                                               vibeqc_precision_provenance* out) {
  if (out == nullptr) return VIBEQC_STATUS_SUCCESS;
  constexpr std::size_t legacy_size =
      offsetof(vibeqc_precision_provenance, mixed_stage_fock_builds);
  const std::size_t caller_size = out->struct_size;
  if (caller_size < legacy_size || out->abi_version != VIBEQC_ABI_VERSION)
    return VIBEQC_STATUS_ABI_MISMATCH;

  vibeqc_precision_provenance record{};
  record.struct_size = static_cast<uint32_t>(caller_size);
  record.abi_version = VIBEQC_ABI_VERSION;
  record.policy_version = source.policy_version;
  record.requested_mode = source.requested_mode;
  record.effective_bits = source.effective_bits;
  record.mixed_precision_fock_threshold = source.mixed_precision_fock_threshold;
  record.strict_refinement_applied = source.strict_refinement_applied ? 1 : 0;
  record.mixed_precision_reserved_error = source.mixed_precision_reserved_error;
  record.refinement_iterations = static_cast<int32_t>(source.refinement_iterations);
  record.mixed_stage_fock_builds = source.mixed_stage_fock_builds;
  record.strict_stage_fock_builds = source.strict_stage_fock_builds;
  record.post_scf_fock_builds = source.post_scf_fock_builds;
  record.execution_retries = source.execution_retries;
  record.mixed_admission_census = source.mixed_admission_census;
  record.final_residual_audits = source.final_residual_audits;
  record.skipped_final_fock_builds = source.skipped_final_fock_builds;
  record.operator_work_counters_valid = source.operator_work_counters_valid;
  std::memcpy(out, &record, std::min(caller_size, sizeof(record)));
  // Preserve the caller's capacity so a legacy prefix stays a legacy prefix
  // on repeated queries.
  out->struct_size = static_cast<uint32_t>(caller_size);
  return VIBEQC_STATUS_SUCCESS;
}

}  // namespace vibeqc::api

#endif
