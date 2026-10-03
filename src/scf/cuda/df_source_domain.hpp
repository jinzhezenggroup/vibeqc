#pragma once

#include <string>

#include "core/types.hpp"

namespace generativeqc::scf::cuda_execution {

/** Host-only capability preflight shared by source factories and their callers.
 * Check this before expensive reference work; the factory repeats it before
 * allocating a source. This query creates no CUDA context or numerical state.
 */
bool cuda_df_shell_domain(const core::System& system, const char* role, std::string& detail);

/** Value-only source domain: orbital through f, auxiliary through g.
 * Legacy exporters/derivatives continue to use the stricter f query above.
 */
bool cuda_df_value_domain(const core::System& orbital, const core::System& auxiliary,
                          std::string& detail);

}  // namespace generativeqc::scf::cuda_execution
