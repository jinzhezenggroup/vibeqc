#pragma once

#include <algorithm>
#include <cstring>

#include "vibeqc/vibeqc.h"

namespace vibeqc::api {

/** Materialize the validated caller prefix before passing a typed C++ reference.
 * A size-gated read in a callee is not a portable protection for a physically
 * short caller allocation: interprocedural argument promotion may hoist an
 * optional field load into its caller. Preserve struct_size so zero padding
 * never changes absent-option defaults into explicitly supplied zero values.
 * Nested pointees remain borrowed for the duration of preparation. */
inline vibeqc_method_descriptor snapshot_method_descriptor(
    const vibeqc_method_descriptor* input) noexcept {
  vibeqc_method_descriptor result{};
  std::memcpy(&result, input, std::min<std::size_t>(input->struct_size, sizeof(result)));
  return result;
}

}  // namespace vibeqc::api
