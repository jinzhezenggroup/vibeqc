// Validation adapter only: publish arrays/diagnostics after the complete native
// response succeeds; allocation or arithmetic failures leave caller sentinels.
#include <algorithm>
#include <array>
#include <cstdio>
#include <exception>

#include "cc/df_triples.hpp"

extern "C" int df_triples_response_probe(std::size_t o, std::size_t v, std::size_t q,
                                         const double* const* inputs, double threshold,
                                         std::size_t budget, std::size_t caller_bytes,
                                         std::size_t panels, double* const* output, double* values,
                                         std::size_t* counts, char* error,
                                         std::size_t error_size) noexcept {
  try {
    const auto result = generativeqc::cc::triples::pullback_df_cuda(
        o, v, q, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6],
        inputs[7], inputs[8], threshold, budget, 0, caller_bytes, panels);
    const auto& d = result.diagnostic;
    const std::array arrays{&result.bov, &result.bvv, &result.ovoo,  &result.ovov, &result.fov,
                            &result.t1,  &result.t2,  &result.eps_o, &result.eps_v};
    for (std::size_t x = 0; x < 9; ++x) std::copy(arrays[x]->begin(), arrays[x]->end(), output[x]);
    const double scalars[]{d.energy, d.minimum_absolute_denominator, d.seconds};
    const std::size_t work[]{result.numeric_capacity_bytes,
                             result.borrowed_host_bytes,
                             d.workspace_bytes,
                             d.arena_bytes,
                             d.provider_retained_bytes,
                             d.panel_capacity,
                             d.panel_gemms,
                             d.moment_gemms,
                             result.reverse_gemms,
                             d.contraction_summands,
                             d.occupied_tiles,
                             d.epilogue_points,
                             result.scalar_response_evaluations,
                             result.reverse_kernels,
                             result.audit_kernels,
                             d.h2d_bytes,
                             d.d2h_bytes};
    std::copy(std::begin(scalars), std::end(scalars), values);
    std::copy(std::begin(work), std::end(work), counts);
    return 0;
  } catch (const std::exception& e) {
    if (error && error_size) std::snprintf(error, error_size, "%s", e.what());
    return 1;
  }
}
