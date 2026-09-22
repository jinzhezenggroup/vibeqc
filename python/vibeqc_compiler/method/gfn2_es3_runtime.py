"""Compiler-owned scalar GFN2 ES3 energy, potential, and charge response."""

from __future__ import annotations

from vibeqc_compiler.tensor.ad_program import linearize
from vibeqc_compiler.tensor.ir import constant, divide, input_tensor, multiply
from vibeqc_compiler.tensor.program import Program
from vibeqc_compiler.tensor.types import TensorSpec

GFN2_ES3_RUNTIME_VERSION = "gfn2-es3-runtime-ir-v1"


def build_gfn2_es3_primal() -> Program:
    """Build shell-local ES3 energy/potential with production FP64 ordering."""

    charge_spec = TensorSpec((), role="input", differentiable=True)
    parameter_spec = TensorSpec((), role="input", differentiable=False)
    charge = input_tensor("charge", charge_spec)
    gamma3 = input_tensor("gamma3", parameter_spec)

    charge_squared = multiply(charge, charge)
    potential = multiply(charge_squared, gamma3)
    charge_cubed = multiply(charge_squared, charge)
    energy = divide(multiply(charge_cubed, gamma3), constant(3))
    return Program(
        {"energy": energy, "potential": potential},
        provenance={
            "kind": "gfn2-es3-shell-primal",
            "version": GFN2_ES3_RUNTIME_VERSION,
            "source": "#560",
            "fp64_order": "q2=q*q; potential=q2*gamma3; energy=(q2*q*gamma3)/3",
        },
    )


def build_gfn2_es3_kernel() -> Program:
    """Attach a compiler-AD witness for d(energy)/d(charge) == potential."""

    primal = build_gfn2_es3_primal()
    jvp = linearize(primal, ("charge",), outputs=("energy",))
    definitions = tuple(dict.fromkeys([*primal.nodes, *jvp.program.nodes]))
    return Program(
        {
            "energy": primal.outputs["energy"],
            "potential": primal.outputs["potential"],
            "energy_charge_derivative": jvp.program.outputs["d_energy"],
        },
        definitions,
        provenance={
            "kind": "gfn2-es3-shell-kernel",
            "version": GFN2_ES3_RUNTIME_VERSION,
            "primal_logical_hash": primal.logical_hash,
            "charge_jvp_hash": jvp.derivative_hash,
            "generation": "TensorIR forward AD",
        },
    )
