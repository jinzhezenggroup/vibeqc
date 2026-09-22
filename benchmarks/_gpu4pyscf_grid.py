"""Compatibility for independent GPU4PySCF comparisons on explicit grids.

This module is benchmark-only. It must not become a native DFT dependency or
change either engine's coordinates, weights, density cutoff or functional.
"""

from __future__ import annotations

from functools import wraps
from inspect import signature
from typing import Any


def preserve_reference_grid_order(engine: Any) -> None:
    """Keep zero-AO blocks in a GPU4PySCF RKS/UKS numerical integrator.

    GPU4PySCF 1.8.1 fills full-grid density arrays in yielded-block order. Its
    default iterator skips empty AO blocks, shifting densities relative to the
    original weights and leaving the tail uninitialized. Strict ordering keeps
    the offsets, but downstream AO scaling rejects zero-dimensional kernels.

    Represent an empty block by one exactly zero AO row mapped to AO zero. This
    contributes zero density and potential while retaining all point offsets,
    including a partial final tile and higher AO derivative components. The
    adapter is local to this numerical-integrator instance and idempotent.
    """
    import cupy as cp

    integrator = engine._numint
    original = integrator.block_loop
    if getattr(original, "_vibeqc_preserves_grid_order", False):
        return
    parameters = signature(original)
    if "strict_grid_order" not in parameters.parameters:
        raise RuntimeError("GPU4PySCF block_loop lacks the strict-grid-order contract")

    @wraps(original)
    def blocks(*args: Any, **kwargs: Any) -> Any:
        # Binding also handles callers that pass strict_grid_order positionally.
        bound = parameters.bind(*args, **kwargs)
        bound.arguments["strict_grid_order"] = True
        for ao, indices, weights, coords in original(*bound.args, **bound.kwargs):
            if len(indices) == 0:
                shape = list(ao.shape)
                shape[-2] = 1
                # The 1.8.1 strict empty-block branch leaves the singleton
                # component axis in LDA, although nonempty LDA AO is 2-D.
                if bound.arguments.get("deriv", 0) == 0 and len(shape) == 3:
                    shape = shape[1:]
                yield (
                    cp.zeros(shape, dtype=ao.dtype),
                    cp.zeros(1, dtype=cp.int32),
                    weights,
                    coords,
                )
            else:
                yield ao, indices, weights, coords

    blocks._vibeqc_preserves_grid_order = True  # type: ignore[attr-defined]
    integrator.block_loop = blocks
