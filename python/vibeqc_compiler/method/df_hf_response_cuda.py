"""Production CUDA/host lowering for the DF-HF stationary response plan."""

from __future__ import annotations

from vibeqc_compiler.common.layout import DenseLayout
from vibeqc_compiler.tensor.cuda_gemm import direct_gemm_kind, gemm_contract
from vibeqc_compiler.tensor.ir import einsum, input_tensor
from vibeqc_compiler.tensor.types import Index, IndexSpace, TensorSpec

from .df_hf_response_contract import (
    CONTRACT_IDENTITY,
    RHF_COULOMB_COEFFICIENT,
    RHF_EXCHANGE_COEFFICIENT,
)

from .df_occupied_response_cuda import emit_occupied_response_helpers

_CHARGE_EQUATION = "tij,pij->tp"


def df_rhf_charge_gemm_kind() -> str:
    """Return the compiler lowering required by the production charge contraction.

    Runtime DF response may carry more than one density term, so the production
    contraction adds an explicit term batch to the stationary-plan
    ``ij,pij->p`` charge. Representative extents are sufficient here: GEMM
    eligibility and physical transpose choice depend on index incidence/layout,
    not on the concrete AO or auxiliary sizes.
    """
    term_space = IndexSpace("df_rhf_term", "batch", 2)
    ao_space = IndexSpace("df_rhf_charge_ao", "ao", 4)
    aux_space = IndexSpace("df_rhf_charge_aux", "auxiliary", 3)
    t = Index("t", term_space)
    i = Index("i", ao_space)
    j = Index("j", ao_space)
    p = Index("p", aux_space)
    densities = input_tensor(
        "densities", TensorSpec((t, i, j), role="input", differentiable=False)
    )
    fitted = input_tensor(
        "fitted", TensorSpec((p, i, j), role="input", differentiable=False)
    )
    charge = einsum(_CHARGE_EQUATION, densities, fitted)
    contract = gemm_contract(charge)
    if contract is None:
        raise RuntimeError("DF-RHF charge contraction lost TensorIR GEMM eligibility")
    kind = direct_gemm_kind(
        contract,
        (
            DenseLayout(densities.spec.shape),
            DenseLayout(fitted.spec.shape),
            DenseLayout(charge.spec.shape),
        ),
    )
    if kind != "direct-NT":
        raise RuntimeError(
            "DF-RHF charge contraction changed its qualified dense GEMM layout: "
            f"{kind!r}"
        )
    return kind


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
    charge_kind = df_rhf_charge_gemm_kind()
    return f"""// Generated from DensityFittingRHFResponsePlan.
// stationary-contract: {CONTRACT_IDENTITY}
// charge-contraction: {_CHARGE_EQUATION}
// tensorir-charge-lowering: {charge_kind}
#ifndef VIBEQC_GENERATED_DF_HF_RESPONSE_CUH
#define VIBEQC_GENERATED_DF_HF_RESPONSE_CUH

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <limits>

namespace vibeqc::scf {{
namespace generated {{

{emit_occupied_response_helpers()}

/** Lower the complete q[t,P] = sum_ij D[t,ij] B[P,ij] contraction at once.
 *
 * TensorIR recognizes the dense C-layout equation as direct-NT. cuBLAS is
 * column-major, so the equivalent physical call computes [P,t] with T,N and
 * writes the existing row-major [t,P] buffer using the full auxiliary stride.
 * Keeping P inside this compiler-owned contraction prevents the historical
 * one-kernel-per-P scalar reduction from reappearing in production BLAS mode.
 */
inline cublasStatus_t df_rhf_charge_contract(
    cublasHandle_t blas, int matrix, int auxiliary_stride, int begin, int count,
    int terms, const double* densities, const double* fitted,
    double* potentials) {{
  const double one = 1.0, zero = 0.0;
  return cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, count, terms, matrix,
                     &one, fitted, matrix, densities, matrix, &zero,
                     potentials + begin, auxiliary_stride);
}}

}}  // namespace generated

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
