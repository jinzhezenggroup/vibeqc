"""Retained DF RCCSD Lambda and parameter actions from the audited residual.

The native owner composes these with every auxiliary slice of df_equations.
Virtual residual inputs are independent corrections, so their amplitude and
factor derivatives must be supplied by that separate chain exactly once.
"""

from __future__ import annotations

from generativeqc_compiler.tensor import (
    Program,
    optimize,
    reassociate_einsums,
    transpose_program,
)

from .doubles import build_ccsd_program
from .lambda_equations import AMPLITUDES, PARAMETERS, RESIDUALS, build_parameter_vjp

RETAINED_PARAMETERS = tuple(name for name in PARAMETERS if name not in ("ovvv", "vvvv"))


def retained_response_programs(occupied: int, virtuals: int) -> dict[str, Program]:
    """Differentiate unpreconditioned equations, keeping shared/expanded audits.

    Neither solver iterates nor numerical denominators participate in AD.
    Outputs use the dense Frobenius metric and the original input symmetries.
    No action reconstructs the omitted three-/four-virtual integral blocks.
    """
    result = {}
    for form, prefix in (("shared", ""), ("expanded", "independent_")):
        primal = build_ccsd_program(
            occupied,
            virtuals,
            form=form,
            diagnostics=False,
            external_virtual_correction=True,
        )
        actions = {
            prefix + "rhs": transpose_program(
                primal, ("correlation_energy",), inputs=AMPLITUDES
            ).program,
            prefix + "transpose": transpose_program(
                primal, RESIDUALS, inputs=AMPLITUDES
            ).program,
        }
        if form == "shared":
            actions.update(
                {
                    "parameter_" + name: build_parameter_vjp(primal, name).program
                    for name in RETAINED_PARAMETERS
                }
            )
        for name, program in actions.items():
            prepared = optimize(
                reassociate_einsums(program, max_intermediate_axes={"virtual": 2})
            )
            if any(
                sum(i.space.kind == "virtual" for i in n.spec.indices) > 2
                for n in prepared.live_nodes
            ):
                raise ValueError("DF response reconstructed an omitted virtual block")
            result[name] = Program(
                prepared.outputs,
                provenance={
                    **prepared.provenance,
                    "native_execution_order": "dependencies",
                    "df_response_domain": "retained core; caller composes all virtual auxiliary actions",
                },
            )
    return result
