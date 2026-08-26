#ifndef VIBEQC_RUNTIME_CONTEXT_HPP
#define VIBEQC_RUNTIME_CONTEXT_HPP

#include "core/types.hpp"

#include <string>

namespace vibeqc::runtime {

vibeqc_status initialize_context(core::ContextState& state, std::string& detail);

#if VIBEQC_HAS_CUDA
vibeqc_status initialize_cuda_context(core::ContextState& state, std::string& detail);
#endif

}  // namespace vibeqc::runtime

#endif
