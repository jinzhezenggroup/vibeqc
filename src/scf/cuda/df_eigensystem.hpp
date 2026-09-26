#pragma once

namespace vibeqc::scf::cuda_df {
/** Teardown for the existing DF plan's opaque ordinary AO eigen workspace.
 * The plan has selected its owning device and retains stream/solver handles. */
void destroy_ordinary_eigensystem(void*& opaque) noexcept;
}  // namespace vibeqc::scf::cuda_df
