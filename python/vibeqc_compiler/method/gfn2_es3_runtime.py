"""Compiler-owned scalar GFN2 ES3 energy, potential, and charge response."""

from __future__ import annotations

from vibeqc_compiler.method.onsite_third_order import (
    ONSITE_THIRD_ORDER_FP64_ORDER,
    build_onsite_third_order_kernel,
    build_onsite_third_order_primal,
)
from vibeqc_compiler.tensor.program import Program

GFN2_ES3_RUNTIME_VERSION = "gfn2-es3-runtime-ir-v1"


def build_gfn2_es3_primal() -> Program:
    """Build shell-local ES3 energy/potential with production FP64 ordering."""

    return build_onsite_third_order_primal(
        provenance={
            "kind": "gfn2-es3-shell-primal",
            "version": GFN2_ES3_RUNTIME_VERSION,
            "source": "#560",
            "fp64_order": ONSITE_THIRD_ORDER_FP64_ORDER,
        }
    )


def build_gfn2_es3_kernel() -> Program:
    """Attach a compiler-AD witness for d(energy)/d(charge) == potential."""

    primal = build_gfn2_es3_primal()
    return build_onsite_third_order_kernel(
        primal,
        provenance={
            "kind": "gfn2-es3-shell-kernel",
            "version": GFN2_ES3_RUNTIME_VERSION,
        },
    )
