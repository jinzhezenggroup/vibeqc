"""Compiler-owned GFN2 SCC internal/free-energy composition.

Runtime code owns ragged iteration, component enablement, validation, diagnostics,
error propagation, and publication. TensorIR owns the ordered scientific scalar
composition shared by CPU and CUDA.
"""

from __future__ import annotations

from vibeqc_compiler.tensor.ir import Node, add, input_tensor, multiply
from vibeqc_compiler.tensor.program import Program
from vibeqc_compiler.tensor.types import TensorSpec

GFN2_SCC_FREE_ENERGY_RUNTIME_VERSION = "gfn2-scc-free-energy-runtime-ir-v1"

_INTERNAL_COMPONENTS = (
    "core",
    "es2",
    "es3",
    "aes2",
    "spin",
    "d4_two_body",
    "explicit_point_charge",
    "electric_field",
    "periodic_embedding",
)


def _input(name: str) -> Node:
    return input_tensor(name, TensorSpec((), role="input", differentiable=False))


def build_gfn2_scc_internal_energy_program() -> Program:
    """Preserve the production serial component-addition order."""

    values = [_input(name) for name in _INTERNAL_COMPONENTS]
    total = values[0]
    for value in values[1:]:
        total = add(total, value)
    return Program(
        {"internal_energy": total},
        provenance={
            "kind": "gfn2-scc-internal-energy",
            "version": GFN2_SCC_FREE_ENERGY_RUNTIME_VERSION,
            "component_order": list(_INTERNAL_COMPONENTS),
        },
    )


def build_gfn2_scc_free_energy_program() -> Program:
    """Build F = E_internal - T*S with the production fused rounding contract."""

    internal_energy = _input("internal_energy")
    electronic_temperature = _input("electronic_temperature")
    entropy = _input("entropy")
    free_energy = add(
        internal_energy,
        multiply(electronic_temperature, entropy),
        coefficients=(1, -1),
    )
    return Program(
        {"free_energy": free_energy},
        provenance={
            "kind": "gfn2-scc-free-energy",
            "version": GFN2_SCC_FREE_ENERGY_RUNTIME_VERSION,
            "rounding": "fma(-electronic_temperature, entropy, internal_energy)",
        },
    )
