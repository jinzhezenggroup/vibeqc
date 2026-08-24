#ifndef QCE_RUNTIME_CONTEXT_HPP
#define QCE_RUNTIME_CONTEXT_HPP

#include "core/types.hpp"

#include <string>

namespace qce::runtime {

qce_status initialize_context(core::ContextState& state, std::string& detail);

#if QCE_HAS_CUDA
qce_status initialize_cuda_context(core::ContextState& state, std::string& detail);
#endif

}  // namespace qce::runtime

#endif
