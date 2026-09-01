#include "api/error.hpp"
#include "methods/method.hpp"
#include "vibeqc/vibeqc.h"

extern "C" {

uint32_t vibeqc_get_abi_version(void) { return VIBEQC_ABI_VERSION; }

const char* vibeqc_status_message(vibeqc_status status) {
  switch (status) {
    case VIBEQC_STATUS_SUCCESS:
      return "success";
    case VIBEQC_STATUS_INVALID_ARGUMENT:
      return "invalid argument";
    case VIBEQC_STATUS_ABI_MISMATCH:
      return "ABI mismatch";
    case VIBEQC_STATUS_NOT_IMPLEMENTED:
      return "requested capability is not implemented";
    case VIBEQC_STATUS_NOT_CONVERGED:
      return "SCF did not converge";
    case VIBEQC_STATUS_NUMERICAL_FAILURE:
      return "numerical failure";
    case VIBEQC_STATUS_CUDA_ERROR:
      return "CUDA runtime error";
    case VIBEQC_STATUS_OUT_OF_MEMORY:
      return "out of memory";
    case VIBEQC_STATUS_INTERNAL_ERROR:
      return "internal error";
  }
  return "unknown status";
}

vibeqc_status vibeqc_method_available(vibeqc_method method, int32_t* available) {
  if (available == nullptr) return VIBEQC_STATUS_INVALID_ARGUMENT;
  const vibeqc::methods::Capabilities* capabilities = vibeqc::methods::find_capabilities(method);
  if (capabilities == nullptr) return VIBEQC_STATUS_INVALID_ARGUMENT;
  *available = capabilities->available ? 1 : 0;
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status vibeqc_method_get_capabilities(vibeqc_method method,
                                             vibeqc_method_capabilities_descriptor* output) {
  if (!vibeqc::api::valid_descriptor(output)) {
    return output == nullptr ? VIBEQC_STATUS_INVALID_ARGUMENT : VIBEQC_STATUS_ABI_MISMATCH;
  }
  const vibeqc::methods::Capabilities* capabilities = vibeqc::methods::find_capabilities(method);
  if (capabilities == nullptr) return VIBEQC_STATUS_INVALID_ARGUMENT;
  output->method = capabilities->method;
  output->family = capabilities->family;
  output->supported_properties = capabilities->supported_properties;
  output->available = capabilities->available ? 1 : 0;
  output->supports_batch = capabilities->supports_batch ? 1 : 0;
  return VIBEQC_STATUS_SUCCESS;
}

}  // extern "C"
