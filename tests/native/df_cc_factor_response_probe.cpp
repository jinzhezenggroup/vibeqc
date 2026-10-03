// Validation-only adapter for arbitrary physical factors and Lagrangian seeds.
// Copy outputs only after the complete native action and finite audit succeed.
#include <algorithm>
#include <array>
#include <cstdio>
#include <exception>

#include "cc/df_source.hpp"

extern "C" int df_cc_factor_response_probe(std::size_t o, std::size_t v, std::size_t q,
                                           const double* const* input, std::size_t budget,
                                           std::size_t caller_bytes, double* const* output,
                                           std::size_t* counts, char* error,
                                           std::size_t error_size) noexcept {
  try {
    const auto block = o * o * v * v;
    const generativeqc::cc::DFFactorResponseView views{
        {input[0], q * o * o},     {input[1], q * o * v},     {input[2], q * v * v},
        {input[3], block},         {input[4], block},         {input[5], block},
        {input[6], o * v * o * o}, {input[7], o * o * o * o}, {input[8], q * o * v},
        {input[9], q * v * v}};
    const auto result =
        generativeqc::cc::pullback_df_factors_cuda(o, v, q, views, budget, 0, caller_bytes);
    const std::array arrays{&result.boo, &result.bov, &result.bvv};
    for (std::size_t i = 0; i < arrays.size(); ++i)
      std::copy(arrays[i]->begin(), arrays[i]->end(), output[i]);
    const std::size_t work[]{result.numeric_capacity_bytes,
                             result.owned_device_bytes,
                             result.h2d_bytes,
                             result.d2h_bytes,
                             result.contraction_terms,
                             result.generated_kernels};
    std::copy(std::begin(work), std::end(work), counts);
    return 0;
  } catch (const std::exception& e) {
    if (error && error_size) std::snprintf(error, error_size, "%s", e.what());
    return 1;
  }
}
