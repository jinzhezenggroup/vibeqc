"""Deterministic typed tensor storage and contraction plans, without CUDA calls.

The byte budget is a combined numeric-buffer budget: device allocations plus
prepared host input staging and the larger of one detached host output set or
immutable static-data upload staging. Caller-owned inputs/old results,
Python/code objects, CUDA context/module/stack overhead,
provider host metadata,
and the CUDA allocator's page rounding are outside this scope. Retained
cuBLAS device allocations have a separate checked allowance. The runtime
reports its device-memory delta separately; it must never label that delta as
the plan's numeric-buffer peak.
"""

from __future__ import annotations

import typing
from dataclasses import asdict, dataclass, field, replace
from functools import cached_property
from math import prod

from vibeqc_compiler.common.backend import TargetScheduleShape
from vibeqc_compiler.common.cuda_target import CudaTargetInfo
from vibeqc_compiler.common.storage import (
    AliasKind,
    BufferOp,
    BufferValue,
    MemoryEffect,
    analyze_storage,
)

from .batch_schedule import (
    BatchScheduleIR,
    analyze_batch_schedule,
    index_table_length,
    index_table_values,
)
from .cuda_dtype import program_precision, scalar_type
from .cuda_gemm import gemm_contract
from .cuda_layout import LayoutDecision, conversion_bytes, select_layouts
from .ir import TRANSCENDENTALS, Node
from .layout import DenseLayout
from .precision import PrecisionSchedule, ValuePrecision, describe_precision
from .program import Program, _hash
from .types import checked_size

PLAN_SCHEMA = 4
ALIGNMENT = 256
INT_MAX = 2**31 - 1
MIN_PROVIDER_BYTES = 96 * 1024**2
VALIDATION_CHUNK = 4096
# Two NumPy iterator buffers, two reusable FP64 scratch buffers and one mask.
VALIDATION_BYTES = VALIDATION_CHUNK * (4 * 8 + 1)
VIEWS = frozenset(("transpose", "reshape", "slice", "broadcast"))
ELEMENTWISE = (
    frozenset(("add", "multiply", "divide", "scaled_bilinear")) | TRANSCENDENTALS
)


def strides(shape: typing.Any) -> tuple[int, ...]:
    """Element strides for the materialized logical C layout."""
    return tuple(prod(shape[i + 1 :]) for i in range(len(shape)))


def aligned(size: int) -> int:
    return checked_size(
        (size + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT, "aligned bytes"
    )


def _index_table_length(node: Node) -> int | None:
    """Compatibility boundary; the shared batch scheduler owns table layout."""
    return index_table_length(node)


def _index_table_values(node: Node) -> tuple[int, ...] | None:
    """Keep existing emitter/admission clients on the one shared table owner."""
    if node.op not in ("gather", "indexed_gather", "scatter_add", "segment_sum"):
        return None
    return index_table_values(node)


@dataclass(frozen=True)
class TensorSchedule:
    """Small explicit search space over one stable stream, optionally replayed.

    Recompute duplicates shared intermediates between output roots. It does
    not duplicate work within a root or promise arbitrary out-of-core output
    support. Fusion retains the order and finite checks of each scalar node.
    Streaming reductions keep a small materialized reduction frontier while
    evaluating fully-consumed higher-rank producers on demand. It is an
    opt-in memory schedule until real-device qualification proves a speed win.
    """

    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 128
    threads: int = 128
    views: bool = False
    fuse: bool = False
    recompute: bool = False
    stream_reductions: bool = field(default=False, kw_only=True)
    direct_gemm: bool = True
    layouts: bool = False
    elements_per_thread: int = 1
    reduction_unroll: int = 1
    staging_width: int = 1

    def __post_init__(self) -> None:
        for name in ("tile_m", "tile_n", "tile_k", "threads"):
            value = getattr(self, name)
            checked_size(value, name)
            if not 1 <= value <= INT_MAX:
                raise ValueError(f"{name} must be a positive cuBLAS-compatible integer")
        for name in ("elements_per_thread", "reduction_unroll", "staging_width"):
            value = getattr(self, name)
            if type(value) is not int or value not in (1, 2, 4, 8):
                raise ValueError(f"{name} must be one of 1, 2, 4, 8")
        for name in (
            "views",
            "fuse",
            "recompute",
            "stream_reductions",
            "direct_gemm",
            "layouts",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be boolean")


@dataclass(frozen=True)
class Reservations:
    """Device bytes held for a caller's T, R, DIIS and concurrent work.

    The plan physically reserves these bytes, so an advertised reservation
    cannot silently become available for intermediates. One stream needs no
    double buffer; concurrent plans must use separate budgets or reserve each
    other's complete peaks explicitly.
    """

    t: int = 0
    r: int = 0
    diis: int = 0
    concurrent: int = 0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            checked_size(value, f"{name} reservation")
        checked_size(self.total, "total reservations")

    @property
    def total(self) -> int:
        return sum(asdict(self).values())


@dataclass(frozen=True)
class Step:
    """An execution occurrence; repeated roots may use the same logical node."""

    node: Node
    inputs: tuple[int, ...]
    virtual: bool
    offset: int
    last_use: int
    gemm: str  # none, packed, direct-NN, direct-NT, direct-TN, direct-TT
    layout: DenseLayout | None


@dataclass(frozen=True)
class TensorPlan:
    """Immutable replayable storage plan with exact allocation capacities."""

    program: Program
    target: CudaTargetInfo
    schedule: TensorSchedule
    steps: tuple[Step, ...]
    inputs: tuple[int, ...]
    outputs: tuple[tuple[str, int], ...]
    index_tables: tuple[tuple[int, int], ...]
    reservations: Reservations
    max_bytes: int
    arena_bytes: int
    panel_bytes: int
    library_bytes: int
    provider_bytes: int
    host_bytes: int
    estimated_flops: int
    estimated_traffic_bytes: int
    layout_decision: LayoutDecision

    @cached_property
    def precision(self) -> str:
        return program_precision(self.program)

    @cached_property
    def precision_schedule(self) -> PrecisionSchedule:
        return describe_precision(self.program)

    @cached_property
    def precision_by_node(self) -> dict[Node, ValuePrecision]:
        """Resolve execution precision once for every live logical node."""
        names = self.program.debug_names
        values = {value.name: value for value in self.precision_schedule.values}
        return {
            node: values[names[node]]
            for node in self.program.live_nodes
            if names[node] in values
        }

    @property
    def batch_schedule(self) -> BatchScheduleIR:
        """Derive exact homogeneous/ragged scheduling facts for this plan."""
        return analyze_batch_schedule(self.steps)

    @property
    def allocation_bytes(self) -> int:
        # Error flag has a full alignment unit to keep every segment aligned.
        return (
            self.arena_bytes
            + self.panel_bytes
            + self.library_bytes
            + aligned(self.reservations.total)
            + ALIGNMENT
        )

    @property
    def index_table_bytes(self) -> int:
        """Exact arena bytes reserved for static ragged/index metadata."""
        total = 0
        for step_index, _ in self.index_tables:
            node = self.steps[step_index].node
            count = _index_table_length(node)
            if count is None:  # pragma: no cover - planner constructs table owners
                raise AssertionError(f"unexpected index-table owner: {node.op}")
            total = checked_size(
                total + aligned(count * 8),
                "index table bytes",
            )
        return total

    @property
    def static_data_bytes(self) -> int:
        """Compact host artifact bytes needed to initialize immutable device data."""
        return sum(item[4] for item in static_data_slices(self))

    @property
    def accumulation_workspace_bytes(self) -> int:
        """Extra ragged reduction workspace beyond materialized outputs."""
        # scatter_add and segment_sum assign one CUDA output element per thread
        # and reduce deterministically inside that owner. No atomics or side
        # accumulation buffer are required.
        return 0

    @property
    def ragged_resources(self) -> dict[str, int | bool]:
        """Auditable ragged storage contract; index tables live in arena_bytes."""
        return {
            "index_table_bytes": self.index_table_bytes,
            "accumulation_workspace_bytes": self.accumulation_workspace_bytes,
            "included_in_arena_bytes": True,
        }

    @property
    def device_bytes(self) -> int:
        return self.allocation_bytes + self.provider_bytes

    @property
    def peak_bytes(self) -> int:
        return self.device_bytes + self.host_bytes

    @property
    def identity(self) -> str:
        return _hash(self.to_payload())

    @property
    def semantic_traffic(self) -> dict:
        """Deterministic semantic byte accounting for this execution topology.

        Declared host copies and pack/scatter conversions are exact byte counts;
        the logical-tensor component reuses the planner's coarse node-traffic
        model. This is not a DRAM/cache counter and excludes cuBLAS internals.
        """
        input_bytes = sum(
            self.steps[i].node.spec.size * self.steps[i].node.spec.itemsize
            for i in self.inputs
        )
        output_bytes = sum(
            self.steps[i].node.spec.size * self.steps[i].node.spec.itemsize
            for _, i in self.outputs
        )
        conversion = sum(
            conversion_bytes(
                gemm_contract(step.node), self.schedule, step.node.spec.itemsize
            )
            for step in self.steps
            if step.gemm == "packed"
        )
        precision = self.precision_schedule
        total = checked_size(
            self.estimated_traffic_bytes + input_bytes + output_bytes + conversion,
            "tensor semantic traffic bytes",
        )
        return {
            "schema": "vibeqc.tensor.cuda.semantic-traffic.v1",
            "logical_tensor_bytes": self.estimated_traffic_bytes,
            "layout_conversion_bytes": conversion,
            "precision_cast_read_bytes": precision.cast_read_bytes,
            "precision_cast_write_bytes": precision.cast_write_bytes,
            "precision_cast_simultaneous_bytes": precision.maximum_cast_live_bytes,
            "host_to_device_bytes": input_bytes,
            "device_to_host_bytes": output_bytes,
            "total_bytes": total,
            "scope": "planner logical tensor bytes plus exact declared copies/packing; excludes hardware cache/DRAM transactions and cuBLAS internal workspace/traffic",
        }

    @property
    def layout_identity(self) -> str:
        """Physical-layout fact usable by #459 guards, without changing the IR."""
        return _hash(
            {
                "steps": [
                    {
                        "shape": s.node.spec.shape,
                        "inputs": s.inputs,
                        "layout": None if s.virtual else s.layout.to_payload(),
                        "view_map": s.node.attrs if s.virtual else None,
                    }
                    for s in self.steps
                ],
                "outputs": self.outputs,
            }
        )

    def storage_analysis(self) -> typing.Any:
        """Adapt TensorIR storage facts to the shared whole-region analysis."""

        storage_views = frozenset(("transpose", "reshape", "slice"))
        alias_owner: dict[int, int] = {}
        for index, step in enumerate(self.steps):
            if (
                not step.virtual
                or step.node.op not in storage_views
                or len(step.inputs) != 1
            ):
                continue
            owner = step.inputs[0]
            while self.steps[owner].virtual:
                parent = self.steps[owner]
                if parent.node.op not in storage_views or len(parent.inputs) != 1:
                    owner = -1
                    break
                owner = parent.inputs[0]
            if owner >= 0:
                alias_owner[index] = owner

        def boundary_reads(index: int) -> tuple[int, ...]:
            if index in alias_owner or not self.steps[index].virtual:
                return (index,)
            result: list[int] = []
            for child in self.steps[index].inputs:
                for value in boundary_reads(child):
                    if value not in result:
                        result.append(value)
            return tuple(result)

        values = []
        inputs = []
        for index, step in enumerate(self.steps):
            logical_bytes = checked_size(
                step.node.spec.size * step.node.spec.itemsize,
                "tensor storage-analysis bytes",
            )
            if not step.virtual:
                values.append(
                    BufferValue(
                        index,
                        aligned(logical_bytes),
                        "device",
                        step.layout,
                        compiler_owned=True,
                    )
                )
                if step.node.op in ("input", "constant"):
                    inputs.append(index)
            elif index in alias_owner:
                values.append(
                    BufferValue(
                        index,
                        logical_bytes,
                        "device",
                        alias=AliasKind.VIEW,
                        alias_of=alias_owner[index],
                        compiler_owned=False,
                    )
                )

        operations = []
        for index, step in enumerate(self.steps):
            if index in alias_owner:
                reads = boundary_reads(step.inputs[0])
            elif step.virtual or step.node.op in ("input", "constant"):
                continue
            else:
                ordered_reads: list[int] = []
                for child in step.inputs:
                    for value in boundary_reads(child):
                        if value not in ordered_reads:
                            ordered_reads.append(value)
                reads = tuple(ordered_reads)
            operations.append(
                BufferOp(
                    ("tensor_step", index),
                    reads,
                    (index,),
                    MemoryEffect.EXPLICIT,
                )
            )

        return analyze_storage(
            tuple(values),
            tuple(operations),
            inputs=tuple(inputs),
            outputs=tuple(index for _, index in self.outputs),
        )

    def to_payload(self) -> dict:
        """Include layouts, aliases, lifetimes, shapes, schedule and reservations."""
        names = self.program.debug_names
        return {
            "schema": PLAN_SCHEMA,
            "layout_planning": self.layout_decision.to_payload(),
            "layout_identity": self.layout_identity,
            "equation": self.program.logical_hash,
            "precision": self.precision,
            "precision_schedule": self.precision_schedule.to_payload(),
            "precision_schedule_identity": self.precision_schedule.identity,
            "arithmetic": "explicit casts; per-value storage/compute/accumulation; RN; fp32 SGEMM pedantic; qualified FP64 reduction accumulation; no TF32 or implicit casts",
            "fp32_flush_to_zero": False,
            "target": self.target.to_payload(),
            "schedule": asdict(self.schedule),
            "reservations": asdict(self.reservations),
            "max_bytes": self.max_bytes,
            "arena_bytes": self.arena_bytes,
            "panel_bytes": self.panel_bytes,
            "library_bytes": self.library_bytes,
            "provider_bytes": self.provider_bytes,
            "allocation_bytes": self.allocation_bytes,
            "host_bytes": self.host_bytes,
            "device_bytes": self.device_bytes,
            "peak_bytes": self.peak_bytes,
            "streams": 1,
            "double_buffer_bytes": 0,
            "estimated_flops": self.estimated_flops,
            "estimated_traffic_bytes": self.estimated_traffic_bytes,
            "semantic_traffic": self.semantic_traffic,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "index_tables": self.index_tables,
            "steps": [
                {
                    "node": names[s.node],
                    "inputs": s.inputs,
                    "virtual": s.virtual,
                    "offset": s.offset,
                    "last_use": s.last_use,
                    "gemm": s.gemm,
                    "shape": s.node.spec.shape,
                    "dtype": s.node.spec.dtype,
                    "itemsize": s.node.spec.itemsize,
                    "strides": None if s.virtual else s.layout.element_strides,
                    "layout": None if s.virtual else s.layout.to_payload(),
                    "view_map": s.node.attrs if s.virtual else None,
                }
                for s in self.steps
            ],
        }


def estimated_cuda_launches(plan: TensorPlan) -> int:
    """Count the emitted endpoint launch sequence without executing CUDA."""

    launches = 1  # per-run arithmetic-error reset
    for step in plan.steps:
        if (
            step.virtual
            or step.node.op in ("input", "constant")
            or not step.node.spec.size
        ):
            continue
        if step.gemm == "none":
            launches += 1
            continue
        contraction = gemm_contract(step.node)
        if contraction is None:
            raise ValueError("GEMM launch estimate requires a contraction node")
        if not contraction.k:
            launches += 1
        elif step.gemm.startswith("direct-"):
            launches += 2
        else:
            tiles = [
                (size + tile - 1) // tile
                for size, tile in zip(
                    (contraction.m, contraction.n, contraction.k),
                    (plan.schedule.tile_m, plan.schedule.tile_n, plan.schedule.tile_k),
                    strict=True,
                )
            ]
            launches += contraction.batch * prod(tiles[:2]) * (2 * tiles[2] + 1)
    return launches


def static_data_slices(
    plan: TensorPlan,
) -> tuple[tuple[int, str, int, int, int], ...]:
    """Map compact artifact payload slices onto aligned device-arena locations."""
    tables = dict(plan.index_tables)
    payload_offset = 0
    result = []
    for step_index, step in enumerate(plan.steps):
        node = step.node
        if node.op == "constant" and node.spec.size:
            size = node.spec.size * node.spec.itemsize
            result.append((step_index, "constant", step.offset, payload_offset, size))
            payload_offset += size
        values = _index_table_values(node)
        if values:
            size = len(values) * 8
            result.append(
                (step_index, "index", tables[step_index], payload_offset, size)
            )
            payload_offset += size
    return tuple(result)


def _occurrences(program: typing.Any, recompute: typing.Any) -> typing.Any:
    nodes, inputs, outputs = [], [], []
    live = program.live_nodes
    shared = {}
    for node in live:
        if node.op in ("input", "constant"):
            shared[node] = len(nodes)
            if node.op == "input":
                inputs.append(len(nodes))
            nodes.append((node, ()))
    if not recompute:
        mapping = dict(shared)
        for node in live:
            if node not in mapping:
                mapping[node] = len(nodes)
                nodes.append((node, tuple(mapping[n] for n in node.inputs)))
        outputs = [(name, mapping[node]) for name, node in program.outputs.items()]
    else:
        for name, root in program.outputs.items():
            needed, pending = set(), [root]
            while pending:
                node = pending.pop()
                if node not in needed:
                    needed.add(node)
                    pending.extend(node.inputs)
            mapping = dict(shared)
            for node in live:
                if node in needed and node not in mapping:
                    mapping[node] = len(nodes)
                    nodes.append((node, tuple(mapping[n] for n in node.inputs)))
            outputs.append((name, mapping[root]))
    return nodes, tuple(inputs), tuple(outputs)


def _fully_consumes_operand(parent: Node, child: Node) -> bool:
    """Prove that evaluating parent visits every element of child.

    Streaming a producer through a reduction is legal only when materializing
    the producer first cannot expose an arithmetic/bounds failure on an element
    that the consumer would otherwise skip. TensorIR nodes are pure, but their
    fail-closed diagnostics are observable semantics.
    """

    if parent.spec.size == 0 or child.spec.size == 0:
        return parent.spec.size == child.spec.size == 0
    if parent.op in ELEMENTWISE or parent.op in {
        "cast",
        "transpose",
        "reshape",
        "broadcast",
        "reduce",
        "scatter_add",
    }:
        return True
    if parent.op == "einsum":
        domains = {}
        for operand, labels in zip(parent.inputs, parent.attrs["labels"], strict=True):
            domains.update(zip(labels, operand.spec.shape, strict=True))
            if operand is child:
                child_labels = labels
        if child not in parent.inputs:
            return False
        # Repeated labels select a diagonal rather than the full tensor. An
        # empty sibling-only label also makes the contraction skip this child.
        return len(child_labels) == len(set(child_labels)) and all(
            extent for label, extent in domains.items() if label not in child_labels
        )
    # These operators can select only a subset of their source domain. Keep
    # upstream diagnostics materialized until a stronger full-consumption proof
    # exists for their concrete maps/ranges.
    return False


def _streaming_reduction_virtuals(
    nodes: list[tuple[Node, tuple[int, ...]]],
    users: list[set[int]],
) -> frozenset[int]:
    """Find full-consumption producer chains that can stream into reductions."""

    candidates: set[int] = set()
    region: set[int] = set()
    roots: set[int] = set()
    for root_index, (root, operands) in enumerate(nodes):
        if root.op != "reduce" or len(operands) != 1:
            continue
        source = root.inputs[0]
        reduced_domains = {
            source.spec.indices[axis].domain for axis in root.attrs["axes"]
        }
        if not reduced_domains:
            continue
        roots.add(root_index)
        pending = [(root_index, operands[0])]
        seen_edges: set[tuple[int, int]] = set()
        while pending:
            parent_index, child_index = pending.pop()
            edge = (parent_index, child_index)
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            parent = nodes[parent_index][0]
            child, child_operands = nodes[child_index]
            if child.op in ("input", "constant"):
                continue
            if not any(index.domain in reduced_domains for index in child.spec.indices):
                continue
            if not _fully_consumes_operand(parent, child):
                continue
            region.add(child_index)
            # Keep the reduction frontier materialized. Values whose axes are
            # entirely inside the reduced domain are small boundary tensors
            # (for example one scalar per runtime lane). Their kernels retain
            # parallelism while higher-rank producers stream through them.
            if any(index.domain not in reduced_domains for index in child.spec.indices):
                candidates.add(child_index)
            pending.extend((child_index, grandchild) for grandchild in child_operands)

    # A shared producer may stream only when every consumer stays inside a
    # proven streaming region (or is one of its reduction roots).
    changed = True
    while changed:
        changed = False
        for child_index in tuple(candidates):
            child = nodes[child_index][0]
            if any(
                (user not in region and user not in roots)
                or not _fully_consumes_operand(nodes[user][0], child)
                for user in users[child_index]
            ):
                candidates.remove(child_index)
                changed = True
    return frozenset(candidates)


BASELINE_SCHEDULE = TensorSchedule()
NO_RESERVATIONS = Reservations()


def plan_cuda(
    program: Program,
    target: CudaTargetInfo,
    *,
    max_bytes: int = 256 * 1024**2,
    schedule: TensorSchedule = BASELINE_SCHEDULE,
    reservations: Reservations = NO_RESERVATIONS,
    library_bytes: int = 4 * 1024**2,
    provider_bytes: int = MIN_PROVIDER_BYTES,
) -> TensorPlan:
    """Plan all allocations before preparation; shrink packing tiles to fit.

    Outputs are indivisible resident tensors. An infeasible minimum fails on
    the CPU, before compiling or touching a device. User data is never needed
    for shape/schedule selection. The baseline shares existing SSA nodes but
    neither rewrites the equation nor uses an external chemistry program.
    """
    if not isinstance(program, Program) or not isinstance(target, CudaTargetInfo):
        raise TypeError("plan_cuda requires a Program and CudaTargetInfo")
    checked_size(max_bytes, "tensor byte budget")
    checked_size(library_bytes, "library workspace")
    checked_size(provider_bytes, "provider allowance")
    if provider_bytes % ALIGNMENT:
        raise ValueError("provider allowance must be a multiple of 256 bytes")
    if library_bytes % ALIGNMENT:
        raise ValueError("library workspace must be a multiple of 256 bytes")
    TargetScheduleShape(schedule.threads, target.warp_size).validate_for(
        target.target_info
    )
    nodes, inputs, outputs = _occurrences(program, schedule.recompute)
    mixed_accumulation_steps: frozenset[int] = frozenset()
    if program.provenance.get("precision_execution") is not None:
        program_names = program.debug_names
        precision_values = {
            value.name: value for value in describe_precision(program).values
        }
        mixed_accumulation_steps = frozenset(
            i
            for i, (node, _) in enumerate(nodes)
            if node.spec.dtype != "int64"
            and precision_values[program_names[node]].compute_dtype
            != precision_values[program_names[node]].accumulation_dtype
        )
    if schedule.layouts and any(
        n.spec.dtype in ("float32", "float64") and n.spec.dtype != "float64"
        for n, _ in nodes
    ):
        raise ValueError("producer layout optimization is qualified only for float64")
    if any(n.op in TRANSCENDENTALS for n, _ in nodes) and len(nodes) > INT_MAX // 2:
        raise ValueError("too many steps for transcendental domain diagnostics")
    for node, _ in nodes:
        if node.op in TRANSCENDENTALS and node.spec.dtype != "float64":
            raise ValueError(
                "CUDA transcendental primitives are qualified only for float64"
            )
        scalar = None if node.spec.dtype == "int64" else scalar_type(node.spec.dtype)
        checked_size(node.spec.size * node.spec.itemsize, "tensor bytes")
        for stride in strides(node.spec.shape):
            checked_size(stride, "tensor stride")
        if node.op == "reduce":
            checked_size(
                prod(node.inputs[0].spec.shape[axis] for axis in node.attrs["axes"]),
                "reduction domain",
            )
        if node.op == "einsum":
            domains = {}
            for child, labels in zip(node.inputs, node.attrs["labels"], strict=True):
                domains.update(zip(labels, child.spec.shape, strict=True))
            checked_size(
                prod(
                    size
                    for label, size in domains.items()
                    if label not in node.attrs["output"]
                ),
                "einsum reduction domain",
            )
        if scalar is not None:
            for pair in node.attrs.get("coefficients", node.attrs.get("values", ())):
                scalar.coefficient(pair)
            if "coefficient" in node.attrs:
                scalar.coefficient(node.attrs["coefficient"])
            if "exponent" in node.attrs:
                scalar.coefficient(node.attrs["exponent"])
    pinned = {i for _, i in outputs} | {
        i for i, (n, _) in enumerate(nodes) if n.op in ("input", "constant")
    }
    users = [set() for _ in nodes]
    for i, (_, operands) in enumerate(nodes):
        for child in operands:
            users[child].add(i)
    streaming_virtuals = (
        _streaming_reduction_virtuals(nodes, users)
        if schedule.stream_reductions
        else frozenset()
    )
    virtual, depths = [], []
    for i, (node, operands) in enumerate(nodes):
        # Only complete same-domain elementwise consumers can fuse arithmetic:
        # slicing away an overflow/zero divisor would change error semantics.
        fuse = (
            schedule.fuse
            and node.op in ELEMENTWISE
            and len(users[i]) == 1
            and nodes[next(iter(users[i]))][0].op in ELEMENTWISE
        )
        depth = 1 + max((depths[c] for c in operands), default=0)
        is_virtual = i not in pinned and (
            i in streaming_virtuals
            or (depth <= 8 and ((schedule.views and node.op in VIEWS) or fuse))
        )
        virtual.append(is_virtual)
        depths.append(depth if is_virtual else 0)
    reads = []
    last = [len(nodes) if i in pinned else i for i in range(len(nodes))]
    for i, (_, operands) in enumerate(nodes):
        leaves = set()
        for child in operands:
            leaves.update(reads[child] if virtual[child] else (child,))
        reads.append(leaves)
        if not virtual[i]:
            for child in leaves:
                last[child] = max(last[child], i)
    offsets, active, free, capacity = {}, {}, [], 0
    tables = []
    for i, (node, _) in enumerate(nodes):
        count = _index_table_length(node)
        if count is not None:
            tables.append((i, capacity))
            capacity = checked_size(
                capacity + aligned(count * 8),
                "index table bytes",
            )
    steps, flops, traffic = [], 0, 0
    for i, (node, operands) in enumerate(nodes):
        for child in tuple(active):
            if last[child] < i:
                free.append(active.pop(child))
        free.sort()
        merged = []
        for offset, size in free:
            if merged and merged[-1][0] + merged[-1][1] == offset:
                begin, before = merged.pop()
                merged.append((begin, before + size))
            else:
                merged.append((offset, size))
        free = merged
        if virtual[i]:
            offsets[i] = -1
        else:
            size = aligned(node.spec.size * node.spec.itemsize)
            fitting = [
                (length, start, j)
                for j, (start, length) in enumerate(free)
                if length >= size
            ]
            if fitting and size:
                length, start, j = min(fitting)
                free.pop(j)
                if length > size:
                    free.append((start + size, length - size))
            else:
                start = capacity
                capacity = checked_size(capacity + size, "tensor arena bytes")
            offsets[i] = start
            if size:
                active[i] = (start, size)
            traffic += node.spec.size * node.spec.itemsize + sum(
                nodes[c][0].spec.size * nodes[c][0].spec.itemsize for c in reads[i]
            )
        g = gemm_contract(node)
        # Library admission depends on GEMM eligibility, not physical order.
        # The bounded layout pass assigns the final packed/direct kind below.
        kind = (
            "packed"
            if g is not None and not virtual[i] and i not in mixed_accumulation_steps
            else "none"
        )
        if g:
            flops += g.flops
        elif node.op == "einsum":
            domains = {}
            for child, labels in zip(node.inputs, node.attrs["labels"], strict=True):
                domains.update(zip(labels, child.spec.shape, strict=True))
            flops += len(node.inputs) * prod(domains.values())
        elif node.op not in VIEWS and node.op not in (
            "input",
            "constant",
            "gather",
            "indexed_gather",
            "runtime_indexed_select",
        ):
            flops += sum(child.spec.size for child in node.inputs)
        steps.append(
            Step(
                node,
                operands,
                virtual[i],
                offsets[i],
                last[i],
                kind,
                None
                if virtual[i]
                else DenseLayout(node.spec.shape, alignment=ALIGNMENT),
            )
        )
    static_host_bytes = checked_size(
        sum(
            step.node.spec.size * step.node.spec.itemsize
            for step in steps
            if step.node.op == "constant"
        )
        + sum((_index_table_length(step.node) or 0) * 8 for step in steps),
        "static host tensor bytes",
    )
    input_host_bytes = sum(
        nodes[i][0].spec.size * nodes[i][0].spec.itemsize for i in inputs
    )
    output_host_bytes = sum(
        nodes[i][0].spec.size * nodes[i][0].spec.itemsize for _, i in outputs
    )
    host = checked_size(
        input_host_bytes
        + (VALIDATION_BYTES if inputs else 0)
        + max(output_host_bytes, static_host_bytes),
        "host tensor bytes",
    )
    needs_blas = any(
        s.gemm != "none" and s.node.spec.size and gemm_contract(s.node).k for s in steps
    )
    if needs_blas and provider_bytes < MIN_PROVIDER_BYTES:
        raise ValueError("cuBLAS plans require at least a 96 MiB provider allowance")
    library_bytes = library_bytes if needs_blas else 0
    provider_bytes = provider_bytes if needs_blas else 0
    fixed = checked_size(
        capacity
        + host
        + library_bytes
        + provider_bytes
        + aligned(reservations.total)
        + ALIGNMENT,
        "fixed tensor bytes",
    )
    tile = [schedule.tile_m, schedule.tile_n, schedule.tile_k]
    while True:
        selected = replace(
            schedule, **dict(zip(("tile_m", "tile_n", "tile_k"), tile, strict=True))
        )
        layouts, kinds, layout_decision = select_layouts(
            nodes,
            virtual,
            pinned,
            selected,
            alignment=ALIGNMENT,
            disabled_gemm=mixed_accumulation_steps,
        )
        steps = [
            replace(s, layout=layouts[i], gemm=kinds[i]) for i, s in enumerate(steps)
        ]
        panel = aligned(
            max(
                (
                    gemm_contract(s.node).panel_bytes(*tile)
                    for s in steps
                    if s.gemm == "packed"
                ),
                default=0,
            )
        )
        checked_size(fixed + panel, "tensor peak bytes")
        if fixed + panel <= max_bytes:
            break
        if max(tile) == 1 or fixed > max_bytes:
            raise ValueError(
                f"infeasible tensor byte budget: at least {fixed + aligned(max((gemm_contract(s.node).panel_bytes(1, 1, 1) for s in steps if s.gemm == 'packed'), default=0))} bytes required, budget {max_bytes}"
            )
        axis = max(range(3), key=lambda k: tile[k])
        tile[axis] = max(1, tile[axis] // 2)
    return TensorPlan(
        program,
        target,
        selected,
        tuple(steps),
        inputs,
        outputs,
        tuple(tables),
        reservations,
        max_bytes,
        capacity,
        panel,
        library_bytes,
        provider_bytes,
        host,
        flops,
        traffic,
        layout_decision,
    )
