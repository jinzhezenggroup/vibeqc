"""Shared legality and resource planning for TensorIR CUDA reductions.

The scientific reduction remains TensorIR-owned.  This module only decides
whether the existing cooperative CUDA reduction shape is active and which
lowering provider is requested.  CUB is qualification-only: the production
default remains the generated reduction.
"""

from __future__ import annotations

import typing
from math import prod
from typing import Literal

from .cuda_dtype import scalar_type

ReductionProvider = Literal["generated", "cub"]
REDUCTION_PROVIDERS: tuple[ReductionProvider, ...] = ("generated", "cub")


def reduction_extent(node: typing.Any) -> int:
    """Return the flattened reduction domain for one TensorIR reduce node."""

    if node.op != "reduce":
        raise ValueError("reduction extent requires a TensorIR reduce node")
    return prod(node.inputs[0].spec.shape[axis] for axis in node.attrs["axes"])


def cooperative_reduction_provider(
    plan: typing.Any, index: int
) -> ReductionProvider | None:
    """Return the active cooperative provider, or None for ordinary lowering."""

    step = plan.steps[index]
    if (
        not plan.schedule.stream_reductions
        or step.virtual
        or step.node.op != "reduce"
        or plan.target.warp_size != 32
        or reduction_extent(step.node) < plan.target.warp_size
    ):
        return None
    provider = plan.schedule.reduction_provider
    if provider not in REDUCTION_PROVIDERS:
        raise ValueError(f"unknown CUDA reduction provider {provider!r}")
    return typing.cast("ReductionProvider", provider)


def cooperative_reduction_shared_bytes(plan: typing.Any, index: int) -> int:
    """Planned shared-memory bytes for one cooperative reduction kernel.

    The generated warp-tree lowering has an exact partial-warp footprint.
    CUB TempStorage is toolkit/header implementation detail, so use one
    accumulator per thread as a conservative static bound.  PTXAS resource
    evidence remains authoritative before any performance promotion.
    """

    provider = cooperative_reduction_provider(plan, index)
    if provider is None:
        return 0
    accumulator = scalar_type(
        plan.precision_by_node[plan.steps[index].node].accumulation_dtype
    )
    if provider == "generated":
        warps = (
            plan.schedule.threads + plan.target.warp_size - 1
        ) // plan.target.warp_size
        return warps * accumulator.itemsize
    return plan.schedule.threads * accumulator.itemsize
