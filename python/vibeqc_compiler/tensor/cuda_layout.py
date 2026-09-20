"""Bounded producer/consumer layout choice for the existing CUDA lowering.

This opt-in CUDA slice is qualified for FP64 only. Ordinary FP32 execution
remains available; producer-layout FP32 qualification is separate.

Only internal materialized values may change layout. Inputs, constants, named
outputs and virtual expressions keep their existing ownership/ABI contracts.
A trial jointly assigns both operands and the result of one GEMM, then costs
ALL contractions and generic accesses in the region. Conflicting consumers do
not create extra copies. This is a deterministic heuristic, not a global optimum
or evidence that a statically cheaper plan will execute faster.
"""

from __future__ import annotations

import typing
from dataclasses import asdict, dataclass
from itertools import product

from .cuda_gemm import direct_gemm_kind, gemm_contract
from .layout import DenseLayout

MAX_LAYOUT_TRIALS = 256
MAX_LAYOUT_SWEEPS = 4


@dataclass(frozen=True)
class LayoutDecision:
    enabled: bool
    trials: int
    truncated: bool
    changed_steps: tuple[int, ...]
    baseline_conversion_bytes: int
    selected_conversion_bytes: int
    access_penalty_bytes: int

    @property
    def selected_cost(self) -> typing.Any:
        return self.selected_conversion_bytes + self.access_penalty_bytes

    def to_payload(self) -> typing.Any:
        return {
            "schema": "vibeqc.tensor.layout-planning.v1",
            **asdict(self),
            "selected_cost": self.selected_cost,
            "maximum_trials": MAX_LAYOUT_TRIALS,
            "maximum_sweeps": MAX_LAYOUT_SWEEPS,
            "cost_model": "panel-conversion-plus-strided-access-v1",
        }


def conversion_bytes(
    g: typing.Any, schedule: typing.Any, itemsize: typing.Any
) -> typing.Any:
    """Semantic packing/scatter reads+writes, not measured DRAM traffic.

    Include repeated A packing across N tiles and B packing across M tiles.
    cuBLAS's own traffic and cache behavior are not conversion traffic.
    """
    if not g or not min(g.batch, g.m, g.n, g.k):
        return 0
    m_tiles = (g.m + schedule.tile_m - 1) // schedule.tile_m
    n_tiles = (g.n + schedule.tile_n - 1) // schedule.tile_n
    return (
        2 * itemsize * g.batch * (g.m * g.k * n_tiles + g.k * g.n * m_tiles + g.m * g.n)
    )


def select_layouts(
    nodes: typing.Any,
    virtual: typing.Any,
    pinned: typing.Any,
    schedule: typing.Any,
    *,
    alignment: typing.Any,
    disabled_gemm: typing.Iterable[int] = (),
) -> typing.Any:
    """Return legal layouts, lowering kinds and a reproducible cost decision.

    The caller reruns this bounded choice when budget admission shrinks tiles,
    so reported costs and packing capacities describe the actual final schedule.
    """
    layouts = tuple(
        None if virtual[i] else DenseLayout(node.spec.shape, alignment=alignment)
        for i, (node, _) in enumerate(nodes)
    )
    disabled_gemm = frozenset(disabled_gemm)
    contracts = tuple(
        None if virtual[i] or i in disabled_gemm else gemm_contract(n)
        for i, (n, _) in enumerate(nodes)
    )
    leaves = []
    for _, operands in nodes:
        reads = set()
        for child in operands:
            reads.update(leaves[child] if virtual[child] else (child,))
        leaves.append(reads)

    def kinds_for(candidate: typing.Any) -> typing.Any:
        return tuple(
            "none"
            if g is None
            else (
                direct_gemm_kind(
                    g, tuple(candidate[c] for c in operands) + (candidate[i],)
                )
                if schedule.direct_gemm
                else None
            )
            or "packed"
            for i, ((_, operands), g) in enumerate(zip(nodes, contracts, strict=True))
        )

    def cost(candidate: typing.Any) -> typing.Any:
        kinds = kinds_for(candidate)
        conversion = sum(
            conversion_bytes(g, schedule, node.spec.itemsize)
            for (node, _), g, kind in zip(nodes, contracts, kinds, strict=True)
            if kind == "packed"
        )
        # One extra logical read/write's worth per non-C generic access is a
        # conservative scheduling penalty, NOT an additional allocation/DRAM
        # measurement. Producer writes themselves are physically coalesced.
        penalty = 0
        for i, (node, _) in enumerate(nodes):
            if virtual[i] or node.op in ("input", "constant") or contracts[i]:
                continue
            penalty += sum(
                nodes[c][0].spec.size * nodes[c][0].spec.itemsize
                for c in leaves[i]
                if not candidate[c].is_c_contiguous
            )
            if not candidate[i].is_c_contiguous:
                penalty += node.spec.size * node.spec.itemsize
        return conversion + penalty, conversion, penalty, kinds

    initial = current = cost(layouts)
    trials, truncated = 0, False
    if schedule.layouts and schedule.direct_gemm:
        for _ in range(MAX_LAYOUT_SWEEPS):
            changed = False
            for i in reversed(range(len(nodes))):
                g = contracts[i]
                if g is None or not min(g.batch, g.m, g.n, g.k):
                    continue
                if trials >= MAX_LAYOUT_TRIALS:
                    truncated = True
                    break
                _, operands = nodes[i]
                best_layouts, best_cost = layouts, current
                for a_order, b_order in product(
                    (g.a_order, g.batch_labels + g.k_labels + g.m_labels),
                    (g.b_order, g.batch_labels + g.n_labels + g.k_labels),
                ):
                    assignments: dict[int, DenseLayout] = {}
                    valid = True
                    for step, labels, order in zip(
                        (*operands, i),
                        (g.a_labels, g.b_labels, g.output_labels),
                        (a_order, b_order, g.c_order),
                        strict=True,
                    ):
                        required = DenseLayout(
                            nodes[step][0].spec.shape,
                            tuple(labels.index(label) for label in order),
                            alignment,
                        )
                        layout = layouts[step]
                        assigned = assignments.get(step)
                        if (
                            layout is None
                            or (step in pinned and not layout.equivalent(required))
                            or (
                                assigned is not None
                                and not assigned.equivalent(required)
                            )
                        ):
                            valid = False
                            break
                        assignments[step] = (
                            layout if layout.equivalent(required) else required
                        )
                    if not valid:
                        continue
                    candidate = tuple(
                        assignments.get(j, value) for j, value in enumerate(layouts)
                    )
                    if candidate == layouts:
                        continue
                    if trials >= MAX_LAYOUT_TRIALS:
                        truncated = True
                        break
                    trials += 1
                    measured = cost(candidate)
                    if measured[0] < best_cost[0]:
                        best_layouts, best_cost = candidate, measured
                if best_cost[0] < current[0]:
                    layouts, current = best_layouts, best_cost
                    changed = True
            if not changed or truncated:
                break
    changed_steps = tuple(
        i for i, layout in enumerate(layouts) if layout and not layout.is_c_contiguous
    )
    decision = LayoutDecision(
        schedule.layouts,
        trials,
        truncated,
        changed_steps,
        initial[1],
        current[1],
        current[2],
    )
    return layouts, current[3], decision
