"""CUDA lowering for canonical SCF density TensorIR equations.

The compiler owns plain/energy-weighted density contractions. Plain density
uses the admitted symmetric AO-pair schedule; weighted density deliberately
keeps the historical full-square FP64 order. Native CUDA keeps launch geometry,
streams, active-item routing, warm-start repair, and solver/runtime state.
"""

from __future__ import annotations

import typing

from vibeqc_compiler.common.provenance import canonical_hash

from .scf import density_program, weighted_density_program


def _kind_signature(node: typing.Any) -> tuple[str, ...]:
    return tuple(index.space.kind for index in node.spec.indices)


def _input_name(node: typing.Any) -> str | None:
    return node.attrs.get("name") if node.op == "input" else None


def _signature(node: typing.Any) -> dict[str, typing.Any]:
    attrs: dict[str, typing.Any] = {}
    for key in ("coefficient", "labels", "output"):
        if key in node.attrs:
            attrs[key] = node.attrs[key]
    if node.op == "input":
        attrs["name"] = node.attrs["name"]
    return {
        "op": node.op,
        "kinds": _kind_signature(node),
        "inputs": tuple(_signature(value) for value in node.inputs),
        "attrs": attrs,
    }


def _template_hash(program: typing.Any) -> str:
    return canonical_hash(
        {
            "outputs": {
                name: _signature(node) for name, node in sorted(program.outputs.items())
            }
        }
    )


def _validated_density_program() -> typing.Any:
    program = density_program(1, 3, spin_count=2, orbital_count=2)
    if tuple(program.outputs) != ("density",):
        raise ValueError("SCF CUDA density lowering requires one density output")
    contraction = program.outputs["density"]
    if contraction.op != "einsum" or contraction.attrs.get("coefficient") != (1, 1):
        raise ValueError("SCF CUDA density must be one unit-coefficient einsum")
    if _kind_signature(contraction) != ("batch", "spin", "ao", "ao"):
        raise ValueError("SCF CUDA density output domains changed")
    if contraction.attrs.get("labels") != (
        (0, 1, 2, 3),
        (0, 1, 3),
        (0, 1, 4, 3),
    ) or contraction.attrs.get("output") != (0, 1, 2, 4):
        raise ValueError("SCF CUDA density contraction topology changed")
    if tuple(_input_name(node) for node in contraction.inputs) != (
        "coefficients",
        "occupations",
        "coefficients",
    ):
        raise ValueError("SCF CUDA density operand topology changed")
    if contraction.inputs[0] is not contraction.inputs[2]:
        raise ValueError("SCF CUDA density coefficients must share one input")
    if tuple(_kind_signature(node) for node in contraction.inputs) != (
        ("batch", "spin", "ao", "orbital"),
        ("batch", "spin", "orbital"),
        ("batch", "spin", "ao", "orbital"),
    ):
        raise ValueError("SCF CUDA density operand layout changed")
    return program


def _validated_weighted_density_program() -> typing.Any:
    program = weighted_density_program(1, 3, spin_count=2, orbital_count=2)
    if tuple(program.outputs) != ("weighted_density",):
        raise ValueError("SCF CUDA weighted density requires one expected output")
    contraction = program.outputs["weighted_density"]
    if contraction.op != "einsum" or contraction.attrs.get("coefficient") != (1, 1):
        raise ValueError(
            "SCF CUDA weighted density must be one unit-coefficient einsum"
        )
    if _kind_signature(contraction) != ("batch", "spin", "ao", "ao"):
        raise ValueError("SCF CUDA weighted-density output domains changed")
    if contraction.attrs.get("labels") != (
        (0, 1, 2, 3),
        (0, 1, 3),
        (0, 1, 4, 3),
    ) or contraction.attrs.get("output") != (0, 1, 2, 4):
        raise ValueError("SCF CUDA weighted-density contraction topology changed")
    first, weights, third = contraction.inputs
    if (
        _input_name(first) != "coefficients"
        or first is not third
        or weights.op != "multiply"
    ):
        raise ValueError("SCF CUDA weighted-density operand topology changed")
    if {_input_name(node) for node in weights.inputs} != {
        "occupations",
        "orbital_energies",
    }:
        raise ValueError("SCF CUDA weighted-density weights changed")
    if _kind_signature(weights) != ("batch", "spin", "orbital"):
        raise ValueError("SCF CUDA weighted-density weight domains changed")
    return program


def density_template_hash() -> str:
    """Return the backend-neutral logical identity used by CUDA density AOT."""
    return _template_hash(_validated_density_program())


def weighted_density_template_hash() -> str:
    """Return the logical identity used by CUDA weighted-density AOT."""
    return _template_hash(_validated_weighted_density_program())


def emit_density_cuda() -> str:
    """Emit the bounded integer-occupation CUDA specialization."""
    density_hash = density_template_hash()
    weighted_hash = weighted_density_template_hash()
    return f"""// Generated from python/vibeqc_compiler/tensor/scf.py.
#pragma once

#include <cstddef>
#include <cstdint>

namespace vibeqc::scf::generated {{

inline constexpr const char* cuda_density_tensor_template_hash = "{density_hash}";
inline constexpr const char* cuda_weighted_density_tensor_template_hash = "{weighted_hash}";
inline constexpr const char* cuda_density_schedule =
    "dense-upper-triangle-mirror-v1";

template <int OccupationWeight>
__global__ void occupied_density_kernel(
    std::int32_t batch_size, std::int32_t spin_count, std::int32_t nbf,
    const std::int32_t* occupied, const double* coefficients,
    const std::uint8_t* active, double* density) {{
  static_assert(OccupationWeight == 1 || OccupationWeight == 2);
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t state_count =
      static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(spin_count);
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= state_count * matrix_size) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system =
      state / static_cast<std::size_t>(spin_count);
  if (active != nullptr && active[system] == 0) return;

  const std::size_t local = element % matrix_size;
  const std::size_t row = local % n;
  const std::size_t column = local / n;
  if (row > column) return;

  const std::size_t offset = state * matrix_size;
  double value = 0.0;
  for (std::int32_t orbital = 0; orbital < occupied[state]; ++orbital) {{
    const std::size_t orbital_index = static_cast<std::size_t>(orbital) * n;
    if constexpr (OccupationWeight == 2) {{
      value += 2.0 * coefficients[offset + row + orbital_index] *
               coefficients[offset + column + orbital_index];
    }} else {{
      value += coefficients[offset + row + orbital_index] *
               coefficients[offset + column + orbital_index];
    }}
  }}
  density[element] = value;
  if (row != column)
    density[offset + column + row * n] = value;
}}

template <int OccupationWeight>
__global__ void occupied_weighted_density_kernel(
    std::int32_t batch_size, std::int32_t spin_count, std::int32_t nbf,
    const std::int32_t* occupied, const double* coefficients,
    const double* orbital_energies, const std::uint8_t* active,
    double* weighted_density) {{
  static_assert(OccupationWeight == 1 || OccupationWeight == 2);
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t state_count =
      static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(spin_count);
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= state_count * matrix_size) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system =
      state / static_cast<std::size_t>(spin_count);
  if (active[system] == 0) return;

  const std::size_t local = element % matrix_size;
  const std::size_t row = local % n;
  const std::size_t column = local / n;
  const std::size_t offset = state * matrix_size;
  const std::size_t eigen_offset = state * n;
  double value = 0.0;
  for (std::int32_t orbital = 0; orbital < occupied[state]; ++orbital) {{
    const std::size_t orbital_index = static_cast<std::size_t>(orbital) * n;
    if constexpr (OccupationWeight == 2) {{
      value += 2.0 * orbital_energies[eigen_offset + orbital] *
               coefficients[offset + row + orbital_index] *
               coefficients[offset + column + orbital_index];
    }} else {{
      value += orbital_energies[eigen_offset + orbital] *
               coefficients[offset + row + orbital_index] *
               coefficients[offset + column + orbital_index];
    }}
  }}
  weighted_density[element] = value;
}}

}}  // namespace vibeqc::scf::generated
"""
