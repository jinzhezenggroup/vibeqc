#ifndef VIBEQC_SCF_CUDA_FOCK_EXECUTION_HPP
#define VIBEQC_SCF_CUDA_FOCK_EXECUTION_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <string>

#include "scf/fock_prepared.hpp"

namespace vibeqc::scf {

/** Device-resident execution binding for one prepared Fock owner.
 *
 * Method consumers use this seam instead of borrowing a concrete Direct-J/K
 * or DF handle. The opaque source identity participates only in owner-local
 * replay invalidation; scientific and provider identity remain in the prepared
 * plan. Unsupported compositions return an empty binding rather than silently
 * selecting another provider.
 */
struct PreparedCudaFockBinding {
  int device_id{-1};
  cudaStream_t stream{};
  const void* source_identity{};
  std::size_t nbf{};

  explicit operator bool() const noexcept {
    return device_id >= 0 && stream != nullptr && source_identity != nullptr && nbf != 0;
  }
};

/** Return the ordinary-stream device binding when the prepared plan can
 * execute its complete requested value-side Fock model through one resident
 * provider. The value seam covers full-range Coulomb plus exact full-/short-/
 * long-range exchange. Range-separated exchange may reuse the conservative
 * full-range Schwarz bounds for value screening; operator-specific tighter
 * bounds and range derivatives remain separate follow-ups. DF/mixed-provider
 * compositions remain explicit follow-ups.
 */
PreparedCudaFockBinding prepared_cuda_fock_binding(const PreparedFockPlan& plan) noexcept;

/** Enqueue the prepared plan's complete raw J/K request on caller-owned device
 * buffers. Output pointers follow FockBuildSpec presence/spin semantics.
 * mixed_coulomb changes only the qualified Coulomb recurrence precision; it
 * never changes K, scientific coefficients, screening, or provider selection.
 */
vibeqc_status enqueue_prepared_cuda_fock(const PreparedFockPlan& plan, const double* density,
                                         const double* beta, std::size_t matrix_elements,
                                         double* coulomb, double* alpha_exchange,
                                         double* beta_exchange, int* numerical_error,
                                         bool mixed_coulomb, std::string& detail);

}  // namespace vibeqc::scf

#endif
