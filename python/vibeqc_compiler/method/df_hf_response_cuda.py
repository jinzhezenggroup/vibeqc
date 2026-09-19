"""Production CUDA/host lowering for the DF-HF stationary response plan."""

from __future__ import annotations

from .df_hf_response_contract import (
    CONTRACT_IDENTITY,
    RHF_COULOMB_COEFFICIENT,
    RHF_EXCHANGE_COEFFICIENT,
)


def emit_df_hf_response_contract() -> str:
    """Emit native RHF coefficient constants from the stationary plan."""
    return f"""// Generated from DensityFittingRHFResponsePlan.
// stationary-contract: {CONTRACT_IDENTITY}
#ifndef VIBEQC_GENERATED_DF_HF_RESPONSE_CONTRACT_HPP
#define VIBEQC_GENERATED_DF_HF_RESPONSE_CONTRACT_HPP

namespace vibeqc::scf::generated {{
inline constexpr double df_rhf_coulomb_coefficient =
    {float(RHF_COULOMB_COEFFICIENT):.17g};
inline constexpr double df_rhf_exchange_coefficient =
    {float(RHF_EXCHANGE_COEFFICIENT):.17g};
}}  // namespace vibeqc::scf::generated
#endif
"""


def emit_df_hf_response_cuda() -> str:
    """Emit runtime-sized source-weight kernels bound to the stationary plan."""
    return f"""// Generated from DensityFittingRHFResponsePlan.
// stationary-contract: {CONTRACT_IDENTITY}
#ifndef VIBEQC_GENERATED_DF_HF_RESPONSE_CUH
#define VIBEQC_GENERATED_DF_HF_RESPONSE_CUH

#include <cuda_runtime.h>

#include <cstddef>

namespace vibeqc::scf {{
static __global__ void coulomb_weights_kernel(
    std::size_t matrix, std::size_t a, std::size_t begin, std::size_t count,
    double coefficient, const double* density, const double* potential,
    double* weights) {{
  const auto pi = std::size_t{{blockIdx.x}} * blockDim.x + threadIdx.x;
  if (pi >= count * matrix) return;
  weights[pi] += coefficient * density[pi % matrix] *
                 potential[begin + pi / matrix];
}}

static __global__ void coulomb_metric_kernel(
    std::size_t a, double coefficient, const double* charges,
    double* bar_inverse) {{
  const auto pq = std::size_t{{blockIdx.x}} * blockDim.x + threadIdx.x;
  if (pq < a * a)
    bar_inverse[pq] +=
        .5 * coefficient * charges[pq / a] * charges[pq % a];
}}
static __global__ void exchange_weights_kernel(
    std::size_t matrix, std::size_t a, std::size_t begin, std::size_t count,
    std::size_t q, double coefficient, const double* inverse,
    const double* response, double* weights) {{
  const auto pi = std::size_t{{blockIdx.x}} * blockDim.x + threadIdx.x;
  if (pi >= count * matrix) return;
  weights[pi] += -2 * coefficient *
                 inverse[(begin + pi / matrix) * a + q] *
                 response[pi % matrix];
}}

static __global__ void exchange_metric_kernel(
    std::size_t matrix, std::size_t a, std::size_t begin, std::size_t count,
    std::size_t q, double coefficient, const double* raw,
    const double* response, double* bar_inverse) {{
  const auto p = std::size_t{{blockIdx.x}} * blockDim.x + threadIdx.x;
  if (p >= count) return;
  double value = 0;
  for (std::size_t ij = 0; ij < matrix; ++ij)
    value += raw[p * matrix + ij] * response[ij];
  bar_inverse[(begin + p) * a + q] -= coefficient * value;
}}
static __global__ void fitted_exchange_weights_kernel(
    std::size_t matrix, double coefficient, const double* response,
    double* weights) {{
  const auto ij = std::size_t{{blockIdx.x}} * blockDim.x + threadIdx.x;
  if (ij < matrix) weights[ij] -= 2 * coefficient * response[ij];
}}

}}  // namespace vibeqc::scf
#endif
"""
