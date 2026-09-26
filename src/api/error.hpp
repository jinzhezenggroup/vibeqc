#ifndef VIBEQC_API_ERROR_HPP
#define VIBEQC_API_ERROR_HPP

#include <cstddef>
#include <string>

#include "vibeqc/vibeqc.h"

namespace vibeqc::api {

template <typename T>
bool valid_descriptor(const T* descriptor) {
  return descriptor != nullptr && descriptor->struct_size >= sizeof(T) &&
         descriptor->abi_version == VIBEQC_ABI_VERSION;
}

/** Map the active C++ exception to the stable public status vocabulary. */
vibeqc_status map_exception(std::string* detail = nullptr) noexcept;

}  // namespace vibeqc::api

#endif
