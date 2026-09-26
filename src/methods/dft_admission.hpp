#ifndef VIBEQC_METHODS_DFT_ADMISSION_HPP
#define VIBEQC_METHODS_DFT_ADMISSION_HPP

#include <cmath>
#include <string>
#include <string_view>

#include "vibeqc/vibeqc.h"

#if VIBEQC_HAS_CUDA
#include "generated_split_hybrid_registry.cuh"
#endif

namespace vibeqc::methods::detail {

/** Descriptor-only admission before either single or batch scientific owners.
 * A registered split semilocal pair requires its exact-exchange contribution:
 * a missing/zero K term must not take the ordinary pure-semilocal bypass.
 * Names/coefficients come from the generated registry, never another table. */
inline vibeqc_status validate_split_hybrid_descriptor(const vibeqc_method_descriptor& descriptor,
                                                      vibeqc_backend backend, std::string& detail) {
  const auto reject = [&](vibeqc_status status, const char* message) {
    detail = message;
    return status;
  };
  detail.clear();
  const auto* input = descriptor.ks_options;
  if (!input) return VIBEQC_STATUS_SUCCESS;
  if (input->struct_size < sizeof(vibeqc_ks_options) || input->abi_version != VIBEQC_ABI_VERSION)
    return reject(VIBEQC_STATUS_ABI_MISMATCH, "KS execution-plan ABI mismatch");
  if ((input->semilocal_components == nullptr) != (input->semilocal_component_count == 0) ||
      (input->exchange_terms == nullptr) != (input->exchange_term_count == 0))
    return reject(VIBEQC_STATUS_INVALID_ARGUMENT, "inconsistent KS component pointer/count");
#if VIBEQC_HAS_CUDA
  if (input->semilocal_component_count != 2) return VIBEQC_STATUS_SUCCESS;
  const auto& first = input->semilocal_components[0];
  const auto& second = input->semilocal_components[1];
  if (!first.component_id || !second.component_id || !*first.component_id ||
      !*second.component_id || !std::isfinite(first.coefficient) ||
      !std::isfinite(second.coefficient))
    return reject(VIBEQC_STATUS_INVALID_ARGUMENT, "invalid KS semilocal component");
  const auto code = dft::generated::split_hybrid_functional_code(
      std::string_view(first.component_id), std::string_view(second.component_id));
  if (!code) return VIBEQC_STATUS_SUCCESS;
  if (input->spin_channels != 1 && input->spin_channels != 2)
    return reject(VIBEQC_STATUS_INVALID_ARGUMENT, "invalid KS spin channel count");
  if (backend != VIBEQC_BACKEND_CUDA || descriptor.precision_mode != VIBEQC_PRECISION_FP64 ||
      descriptor.density_fitting_mode != VIBEQC_DENSITY_FITTING_NONE ||
      input->xc_execution_schedule != VIBEQC_XC_EXECUTION_DEVICE_FUSED)
    return reject(VIBEQC_STATUS_NOT_IMPLEMENTED,
                  "split-global-hybrid KS requires CUDA, direct J/K, device-fused XC and FP64");
  if (first.coefficient != 1.0 || second.coefficient != 1.0 ||
      input->semilocal_range_omega != 0.0 || input->has_nonlocal_correlation != 0 ||
      descriptor.method == VIBEQC_METHOD_PBE_D4_RKS)
    return reject(VIBEQC_STATUS_NOT_IMPLEMENTED,
                  "split-global-hybrid KS requires its unmodified electronic composition");
  if (input->exchange_term_count != 1)
    return reject(VIBEQC_STATUS_NOT_IMPLEMENTED,
                  "split-global-hybrid KS requires one nonzero full-range exact-exchange term");
  const auto composition = dft::generated::split_hybrid_composition(code);
  if (!composition.matched || !composition.exact_exchange_denominator)
    return reject(VIBEQC_STATUS_INTERNAL_ERROR, "split-hybrid registry composition is missing");
  const auto& exchange = input->exchange_terms[0];
  const double exact = static_cast<double>(composition.exact_exchange_numerator) /
                       static_cast<double>(composition.exact_exchange_denominator);
  const double divisor = input->spin_channels == 1 ? 2.0 : 1.0;
  if (exchange.operator_kind != VIBEQC_KS_EXCHANGE_FULL_RANGE || exchange.omega != 0.0 ||
      exchange.coefficient != exact || exchange.fock_coefficient != -exact / divisor)
    return reject(VIBEQC_STATUS_NOT_IMPLEMENTED,
                  "split-global-hybrid KS exact exchange disagrees with the generated composition");
#else
  (void)backend;
#endif
  return VIBEQC_STATUS_SUCCESS;
}

}  // namespace vibeqc::methods::detail
#endif
