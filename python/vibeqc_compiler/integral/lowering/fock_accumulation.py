"""Emit the shared Coulomb and RHF/UHF Direct-Fock scatter contractions."""

from __future__ import annotations


def emit_fock_accumulation_cuda(
    *,
    function_name: str,
    parameters: str,
    setup: str,
    permutation_setup: str,
    matrix_index: str,
    density_offset: str,
    spin_offset: str,
    description: str,
    coulomb_only: str,
    unroll_permutations: bool = True,
) -> str:
    """Emit one canonical-ERI-to-Fock scatter with shared spin semantics."""

    unroll_directive = "#pragma unroll\n" if unroll_permutations else ""
    return f"""/** {description} */
template <bool Unrestricted>
__device__ __forceinline__ void {function_name}(
{parameters}) {{
{setup}
{unroll_directive}  for (unsigned permutation = 0; permutation < 8U; ++permutation) {{
{permutation_setup}
    const std::size_t ab = {matrix_index}(a, b, n);
    const std::size_t ac = {matrix_index}(a, c, n);
    const std::size_t cd = {matrix_index}(c, d, n);
    const std::size_t bd = {matrix_index}(b, d, n);
    if constexpr (Unrestricted) {{
      const double alpha_cd = density[{spin_offset} + cd];
      const double beta_cd = density[{spin_offset} + matrix_size + cd];
      const double total_cd = alpha_cd + beta_cd;
      if (total_cd != 0.0) {{
        atomicAdd(fock + {spin_offset} + ab, total_cd * integral);
        atomicAdd(
            fock + {spin_offset} + matrix_size + ab,
            total_cd * integral);
      }}
      if (!({coulomb_only})) {{
        const double alpha_bd = density[{spin_offset} + bd];
        const double beta_bd = density[{spin_offset} + matrix_size + bd];
        if (alpha_bd != 0.0) {{
          atomicAdd(fock + {spin_offset} + ac, -alpha_bd * integral);
        }}
        if (beta_bd != 0.0) {{
          atomicAdd(
              fock + {spin_offset} + matrix_size + ac,
              -beta_bd * integral);
        }}
      }}
    }} else {{
      const double density_cd = density[{density_offset} + cd];
      if (density_cd != 0.0) {{
        atomicAdd(fock + {density_offset} + ab, density_cd * integral);
      }}
      if (!({coulomb_only})) {{
        const double density_bd = density[{density_offset} + bd];
        if (density_bd != 0.0) {{
          atomicAdd(
              fock + {density_offset} + ac,
              -0.5 * density_bd * integral);
        }}
      }}
    }}
  }}
}}
"""


def emit_direct_force_density_coefficient() -> str:
    """Emit the exact symmetry-reduced Direct force density coefficient."""

    return """template <bool Unrestricted>
__device__ __forceinline__ double direct_force_density_coefficient(
    std::size_t n, std::size_t physical_offset, std::size_t spin_offset,
    const double* density,
    std::size_t i, std::size_t j, std::size_t k, std::size_t l) {
  const std::size_t matrix_size = n * n;
  double coefficient = 0.0;
  for (unsigned permutation = 0; permutation < 8; ++permutation) {
    if (!unique_eri_symmetry_permutation(permutation, i, j, k, l)) {
      continue;
    }
    std::size_t a = 0;
    std::size_t b = 0;
    std::size_t c = 0;
    std::size_t d = 0;
    eri_symmetry_permutation(permutation, i, j, k, l, a, b, c, d);
    const std::size_t ab = matrix_index(a, b, n);
    const std::size_t ac = matrix_index(a, c, n);
    const std::size_t cd = matrix_index(c, d, n);
    const std::size_t bd = matrix_index(b, d, n);
    if constexpr (Unrestricted) {
      const double total_ab =
          density[spin_offset + ab] + density[spin_offset + matrix_size + ab];
      const double total_cd =
          density[spin_offset + cd] + density[spin_offset + matrix_size + cd];
      coefficient += 0.5 * total_ab * total_cd;
      coefficient -=
          0.5 * (density[spin_offset + ac] * density[spin_offset + bd] +
                 density[spin_offset + matrix_size + ac] *
                     density[spin_offset + matrix_size + bd]);
    } else {
      coefficient +=
          0.5 * density[physical_offset + ab] * density[physical_offset + cd] -
          0.25 * density[physical_offset + ac] * density[physical_offset + bd];
    }
  }
  return coefficient;
}
"""



def emit_generated_shell_fock_accumulation() -> str:
    """Emit the scatter helper embedded in compiler-generated shell kernels."""

    return emit_fock_accumulation_cuda(
        function_name="generated_dppp_accumulate_fock",
        parameters="""    const GeneratedDpppShellTask& task,
    const double* density,
    double* fock,
    std::size_t i, std::size_t j, std::size_t k, std::size_t l,
    double integral""",
        setup="""  const std::size_t n = static_cast<std::size_t>(task.matrix_order);
  const std::size_t matrix_size = n * n;""",
        permutation_setup="""    std::size_t a = 0, b = 0, c = 0, d = 0;
    generated_dppp_eri_permutation(
        permutation, i, j, k, l, a, b, c, d);
    if (!generated_dppp_unique_permutation(
            permutation, i, j, k, l, a, b, c, d)) continue;""",
        matrix_index="generated_dppp_matrix_index",
        density_offset="task.density_offset",
        spin_offset="task.spin_offset",
        coulomb_only="(task.reversed_shell_pair_mask & kGeneratedDpppCoulombConsumerBit) != 0U",
        description=(
            "Scatter one canonical integral using VIBEQC's existing RHF/UHF convention."
        ),
    ).rstrip("\n")


def emit_direct_fock_accumulation_header() -> str:
    """Emit the native fallback adapter from the same compiler-owned equations."""

    function = emit_fock_accumulation_cuda(
        function_name="accumulate_direct_fock_integral",
        parameters="""    std::size_t n, std::size_t physical_offset, std::size_t spin_offset,
    const double* density, double* fock, std::size_t i, std::size_t j,
    std::size_t k, std::size_t l, double integral, bool coulomb_only = false""",
        setup="  const std::size_t matrix_size = n * n;",
        permutation_setup="""    if (!unique_eri_symmetry_permutation(permutation, i, j, k, l)) {
      continue;
    }
    std::size_t a = 0;
    std::size_t b = 0;
    std::size_t c = 0;
    std::size_t d = 0;
    eri_symmetry_permutation(permutation, i, j, k, l, a, b, c, d);""",
        matrix_index="matrix_index",
        density_offset="physical_offset",
        spin_offset="spin_offset",
        coulomb_only="coulomb_only",
        description="Scatter one symmetry-canonical ERI into the direct RHF/UHF Fock matrix.",
        unroll_permutations=False,
    )
    return f"""#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "scf/cuda/direct_eri_symmetry.cuh"
#include "scf/cuda/matrix_index.cuh"

// Generated by the scientific compiler. Native Direct-HF owns only the
// surrounding task/runtime plumbing; the RHF/UHF contraction lives here.

namespace vibeqc::scf::cuda_execution {{

{function}
{emit_direct_force_density_coefficient()}
}}  // namespace vibeqc::scf::cuda_execution
"""
