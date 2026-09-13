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

/**
 * Method descriptors gained optional DF fields without changing ABI-0
 * callers. Accept the original prefix and let the method layer probe each
 * appended field using struct_size before reading it.
 */
inline bool valid_method_descriptor(const vibeqc_method_descriptor* descriptor) {
  return descriptor != nullptr &&
         descriptor->struct_size >= offsetof(vibeqc_method_descriptor, density_fitting_mode) &&
         descriptor->abi_version == VIBEQC_ABI_VERSION;
}

/** Result diagnostics were appended to ABI-0 outputs. Accept the original
 * prefix and only write
 * appended fields when the caller supplied them. */
inline bool valid_result_descriptor(const vibeqc_result_descriptor* descriptor) {
  return descriptor != nullptr &&
         descriptor->struct_size >= offsetof(vibeqc_result_descriptor, residual_rms) &&
         descriptor->abi_version == VIBEQC_ABI_VERSION;
}

inline bool valid_batch_result_descriptor(const vibeqc_batch_item_result_descriptor* descriptor) {
  return descriptor != nullptr &&
         descriptor->struct_size >= offsetof(vibeqc_batch_item_result_descriptor, residual_rms) &&
         descriptor->abi_version == VIBEQC_ABI_VERSION;
}

template <typename T>
bool has_output_bytes(const T* descriptor, std::size_t offset, std::size_t size) {
  return descriptor != nullptr && descriptor->struct_size >= offset + size;
}

/** Map the active C++ exception to the stable public status vocabulary. */
vibeqc_status map_exception(std::string* detail = nullptr) noexcept;

}  // namespace vibeqc::api

#endif
