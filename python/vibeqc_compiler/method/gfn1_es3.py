"""Compiler-owned atom-resolved GFN1 onsite third-order electrostatics."""

from __future__ import annotations

from vibeqc_compiler.method.onsite_third_order import (
    ONSITE_THIRD_ORDER_FP64_ORDER,
    build_onsite_third_order_kernel,
    build_onsite_third_order_primal,
)
from vibeqc_compiler.method.xtb import GFN1_PARAMETER_SET
from vibeqc_compiler.tensor.program import Program

GFN1_ES3_VERSION = "gfn1-es3-atom-ir-v1"


def build_gfn1_es3_primal() -> Program:
    """Build the pinned atom-resolved GFN1 third-order energy and potential."""

    return build_onsite_third_order_primal(
        provenance={
            "kind": "gfn1-es3-atom-primal",
            "version": GFN1_ES3_VERSION,
            "source": "#837",
            "state_resolution": "atom",
            "third_order_mode": "atom",
            "parameter_set_identity": GFN1_PARAMETER_SET.identity,
            "parameter_revision": GFN1_PARAMETER_SET.revision,
            "fp64_order": ONSITE_THIRD_ORDER_FP64_ORDER,
        }
    )


def build_gfn1_es3_kernel() -> Program:
    """Attach the charge derivative witness for the GFN1 atom potential."""

    primal = build_gfn1_es3_primal()
    return build_onsite_third_order_kernel(
        primal,
        provenance={
            "kind": "gfn1-es3-atom-kernel",
            "version": GFN1_ES3_VERSION,
            "state_resolution": "atom",
            "parameter_set_identity": GFN1_PARAMETER_SET.identity,
        },
    )
