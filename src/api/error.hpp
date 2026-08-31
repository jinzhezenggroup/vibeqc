#ifndef VIBEQC_API_ERROR_HPP
#define VIBEQC_API_ERROR_HPP

#include "vibeqc/vibeqc.h"

#include <cstddef>
#include <string>

namespace vibeqc::api {

template <typename T>
bool valid_descriptor(const T* descriptor) {
  return descriptor != nullptr && descriptor->struct_size >= sizeof(T) &&
         descriptor->abi_version == VIBEQC_ABI_VERSION;
}

/**
 * Method descriptors gained optional DF fields without changing ABI-0
 * callers. Accept the original prefix and let the method layer probe each
 * appended field using struct_size before reading it.
 */
inline bool valid_method_descriptor(const vibeqc_method_descriptor* descriptor) {
  return descriptor != nullptr &&
         descriptor->struct_size >=
             offsetof(vibeqc_method_descriptor, density_fitting_mode) &&
         descriptor->abi_version == VIBEQC_ABI_VERSION;
}

/** Map the active C++ exception to the stable public status vocabulary. */
vibeqc_status map_exception(std::string* detail = nullptr) noexcept;

}  // namespace vibeqc::api

#endif
