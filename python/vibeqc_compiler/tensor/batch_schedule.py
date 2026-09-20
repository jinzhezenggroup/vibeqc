"""Batch-aware scheduling analysis for TensorIR ragged/indexed work.

This module derives backend scheduling facts from immutable TensorIR topology.
It does not own SCF/SCC convergence or method policy. Static maps are compiled
into deterministic work descriptors so CUDA lowering can avoid per-output scans
without changing mathematical or reduction-order semantics.
"""

from __future__ import annotations

import typing
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise

BATCH_SCHEDULE_SCHEMA = 1
RAGGED_OPS = frozenset(("indexed_gather", "scatter_add", "segment_sum"))


@dataclass(frozen=True)
class RaggedStepSchedule:
    """One deterministic scheduling decision for a ragged TensorIR primitive."""

    step_index: int
    op: str
    axis: int
    source_extent: int
    target_extent: int
    edge_count: int
    nonempty_groups: int
    empty_groups: int
    max_degree: int
    scan_work: int
    scheduled_work: int
    lowering: str
    degree_histogram: tuple[tuple[int, int], ...]

    @property
    def avoided_scan_work(self) -> int:
        return self.scan_work - self.scheduled_work

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "step_index": self.step_index,
            "op": self.op,
            "axis": self.axis,
            "source_extent": self.source_extent,
            "target_extent": self.target_extent,
            "edge_count": self.edge_count,
            "nonempty_groups": self.nonempty_groups,
            "empty_groups": self.empty_groups,
            "max_degree": self.max_degree,
            "scan_work": self.scan_work,
            "scheduled_work": self.scheduled_work,
            "avoided_scan_work": self.avoided_scan_work,
            "lowering": self.lowering,
            "degree_histogram": [list(item) for item in self.degree_histogram],
        }


@dataclass(frozen=True)
class BatchScheduleIR:
    """Backend-neutral batch/ragged scheduling facts for one execution plan."""

    batch_domains: tuple[tuple[str, int], ...]
    ragged_steps: tuple[RaggedStepSchedule, ...]

    @property
    def scan_work(self) -> int:
        return sum(step.scan_work for step in self.ragged_steps)

    @property
    def scheduled_work(self) -> int:
        return sum(step.scheduled_work for step in self.ragged_steps)

    @property
    def avoided_scan_work(self) -> int:
        return self.scan_work - self.scheduled_work

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "schema": BATCH_SCHEDULE_SCHEMA,
            "batch_domains": [
                {"name": name, "extent": extent} for name, extent in self.batch_domains
            ],
            "ragged_steps": [step.to_payload() for step in self.ragged_steps],
            "scan_work": self.scan_work,
            "scheduled_work": self.scheduled_work,
            "avoided_scan_work": self.avoided_scan_work,
        }


def _degree_histogram(degrees: typing.Iterable[int]) -> tuple[tuple[int, int], ...]:
    histogram: dict[int, int] = {}
    for degree in degrees:
        histogram[degree] = histogram.get(degree, 0) + 1
    return tuple(sorted(histogram.items()))


def scatter_add_inverted_table(node: typing.Any) -> tuple[int, ...]:
    """Compile a scatter map into offsets+members while preserving source order."""

    if node.op != "scatter_add":
        raise ValueError("inverted scatter table requires scatter_add")
    axis = node.attrs["axis"]
    positions = tuple(node.attrs["positions"])
    target_extent = node.spec.shape[axis]
    if not positions:
        return ()
    buckets: list[list[int]] = [[] for _ in range(target_extent)]
    for source, target in enumerate(positions):
        buckets[target].append(source)
    offsets = [0]
    members: list[int] = []
    for bucket in buckets:
        members.extend(bucket)
        offsets.append(len(members))
    return tuple(offsets + members)


def index_table_length(node: typing.Any) -> int | None:
    """Count topology bytes without materializing a host-side target map."""
    if node.op in ("gather", "indexed_gather"):
        return len(node.attrs["positions"])
    if node.op == "segment_sum":
        return len(node.attrs["offsets"])
    if node.op == "scatter_add":
        count = len(node.attrs["positions"])
        return node.spec.shape[node.attrs["axis"]] + 1 + count if count else 0
    return None


def index_table_values(node: typing.Any) -> tuple[int, ...]:
    """Return the exact static integer payload copied for one indexed primitive."""

    if node.op in ("gather", "indexed_gather"):
        return tuple(node.attrs["positions"])
    if node.op == "scatter_add":
        return scatter_add_inverted_table(node)
    if node.op == "segment_sum":
        return tuple(node.attrs["offsets"])
    raise ValueError(f"primitive has no static index table: {node.op}")


def _ragged_step(step_index: int, node: typing.Any) -> RaggedStepSchedule:
    axis = node.attrs["axis"]
    source_extent = node.inputs[0].spec.shape[axis]
    target_extent = node.spec.shape[axis]

    output_elements = node.spec.size
    outer_count = output_elements // target_extent if target_extent else 0
    if node.op == "scatter_add":
        degrees = tuple(Counter(node.attrs["positions"]).values())
        groups = target_extent
        scan_work = output_elements * source_extent
        scheduled_work = outer_count * source_extent
        lowering = "inverted-segments"
    elif node.op == "segment_sum":
        offsets = tuple(node.attrs["offsets"])
        degrees = [stop - start for start, stop in pairwise(offsets)]
        groups = len(degrees)
        scan_work = outer_count * source_extent
        scheduled_work = scan_work
        lowering = "contiguous-segments"
    elif node.op == "indexed_gather":
        degrees = tuple(Counter(node.attrs["positions"]).values())
        groups = source_extent
        scan_work = output_elements
        scheduled_work = output_elements
        lowering = "direct-index"
    else:
        raise ValueError(f"unsupported ragged scheduling primitive: {node.op}")

    nonempty = sum(degree != 0 for degree in degrees)
    # Unmapped groups are an arithmetic complement, not a target-sized list.
    # Zero-element tensors can have huge valid index domains under tiny budgets.
    histogram = dict(_degree_histogram(degrees))
    if missing := groups - len(degrees):
        histogram[0] = histogram.get(0, 0) + missing
    return RaggedStepSchedule(
        step_index=step_index,
        op=node.op,
        axis=axis,
        source_extent=source_extent,
        target_extent=target_extent,
        edge_count=sum(degrees),
        nonempty_groups=nonempty,
        empty_groups=groups - nonempty,
        max_degree=max(degrees, default=0),
        scan_work=scan_work,
        scheduled_work=scheduled_work,
        lowering=lowering,
        degree_histogram=tuple(sorted(histogram.items())),
    )


def analyze_batch_schedule(steps: typing.Iterable[typing.Any]) -> BatchScheduleIR:
    """Derive deterministic homogeneous/ragged scheduling facts from plan steps."""

    materialized = tuple(steps)
    batch_domains = {
        (index.space.name, index.extent)
        for step in materialized
        for index in step.node.spec.indices
        if index.space.kind == "batch"
    }
    ragged = tuple(
        _ragged_step(step_index, step.node)
        for step_index, step in enumerate(materialized)
        if step.node.op in RAGGED_OPS
    )
    return BatchScheduleIR(tuple(sorted(batch_domains)), ragged)
