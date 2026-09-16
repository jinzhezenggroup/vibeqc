#pragma once

#include "scf/cuda_density_fitting_final_state.hpp"

namespace vibeqc::scf::cuda_df {
/** One serialized workspace per prepared plan; never aliases captured state. */
void destroy_final_validation(void*& opaque) noexcept;
}  // namespace vibeqc::scf::cuda_df
