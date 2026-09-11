"""Deterministic FP64 tensor storage and contraction plans, without CUDA calls.

The byte budget is a combined numeric-buffer budget: device allocations plus
prepared host input staging and one detached host output set. Caller-owned
inputs/old results, Python/code objects, CUDA context/module/stack overhead,
provider host metadata,
and the CUDA allocator's page rounding are outside this scope. Retained
cuBLAS device allocations have a separate checked allowance. The runtime
reports its device-memory delta separately; it must never label that delta as
the plan's numeric-buffer peak.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import prod

from vibeqc_compiler.common.backend import TargetScheduleShape
from vibeqc_compiler.common.cuda_target import CudaTargetInfo

from .cuda_gemm import fp64_coefficient, gemm_contract
from .ir import Node
from .program import Program, _hash
from .types import checked_size

PLAN_SCHEMA = 1
ALIGNMENT = 256
INT_MAX = 2**31 - 1
MIN_PROVIDER_BYTES = 96 * 1024**2
VALIDATION_CHUNK = 4096
# Two NumPy iterator buffers, two reusable FP64 scratch buffers and one mask.
VALIDATION_BYTES = VALIDATION_CHUNK * (4 * 8 + 1)
VIEWS = frozenset(("transpose", "reshape", "slice", "broadcast"))
ELEMENTWISE = frozenset(("add", "multiply", "divide"))


def strides(shape) -> tuple[int, ...]:
    """Element strides for the materialized logical C layout."""
    return tuple(prod(shape[i + 1 :]) for i in range(len(shape)))


def aligned(size: int) -> int:
    return checked_size(
        (size + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT, "aligned bytes"
    )


@dataclass(frozen=True)
class TensorSchedule:
    """Small explicit search space; ordinary single-stream execution only.

    Recompute duplicates shared intermediates between output roots. It does
    not duplicate work within a root or promise arbitrary out-of-core output
    support. Fusion retains the order and finite checks of each scalar node.
    """

    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 128
    threads: int = 128
    views: bool = False
    fuse: bool = False
    recompute: bool = False
    direct_gemm: bool = True

    def __post_init__(self):
        for name in ("tile_m", "tile_n", "tile_k", "threads"):
            value = getattr(self, name)
            checked_size(value, name)
            if not 1 <= value <= INT_MAX:
                raise ValueError(f"{name} must be a positive cuBLAS-compatible integer")
        for name in ("views", "fuse", "recompute", "direct_gemm"):
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

    def __post_init__(self):
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
    def device_bytes(self) -> int:
        return self.allocation_bytes + self.provider_bytes

    @property
    def peak_bytes(self) -> int:
        return self.device_bytes + self.host_bytes

    @property
    def identity(self) -> str:
        return _hash(self.to_payload())

    def to_payload(self) -> dict:
        """Include layouts, aliases, lifetimes, shapes, schedule and reservations."""
        names = self.program.debug_names
        return {
            "schema": PLAN_SCHEMA,
            "equation": self.program.logical_hash,
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
                    "strides": None if s.virtual else strides(s.node.spec.shape),
                    "view_map": s.node.attrs if s.virtual else None,
                }
                for s in self.steps
            ],
        }


def _occurrences(program, recompute):
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


def _direct_kind(node, virtual_operands):
    """Recognize contiguous grouped matrices with explicit cuBLAS transposes.

    Extent-one labels do not constrain strides. Other layouts use the packed
    path, including affine views until their full physical map is supported.
    """
    g = gemm_contract(node)
    if g is None or virtual_operands or min(g.batch, g.m, g.n, g.k) == 0:
        return None
    if max(g.batch, g.m, g.n, g.k) > INT_MAX:
        return None

    def norm(labels):
        return tuple(i for i in labels if g.extents[i] != 1)

    if norm(g.output_labels) != norm(g.c_order):
        return None
    a = (
        "N"
        if norm(g.a_labels) == norm(g.a_order)
        else "T"
        if norm(g.a_labels) == norm(g.batch_labels + g.k_labels + g.m_labels)
        else None
    )
    b = (
        "N"
        if norm(g.b_labels) == norm(g.b_order)
        else "T"
        if norm(g.b_labels) == norm(g.batch_labels + g.n_labels + g.k_labels)
        else None
    )
    return None if a is None or b is None else "direct-" + a + b


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
    for node, _ in nodes:
        if node.spec.dtype != "float64":
            raise ValueError("CUDA tensor baseline supports only float64")
        checked_size(node.spec.size * 8, "tensor bytes")
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
        for pair in node.attrs.get("coefficients", node.attrs.get("values", ())):
            fp64_coefficient(pair)
        if "coefficient" in node.attrs:
            fp64_coefficient(node.attrs["coefficient"])
    pinned = {i for _, i in outputs} | {
        i for i, (n, _) in enumerate(nodes) if n.op in ("input", "constant")
    }
    users = [set() for _ in nodes]
    for i, (_, operands) in enumerate(nodes):
        for child in operands:
            users[child].add(i)
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
        is_virtual = (
            i not in pinned
            and depth <= 8
            and ((schedule.views and node.op in VIEWS) or fuse)
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
        if node.op == "gather":
            tables.append((i, capacity))
            capacity = checked_size(
                capacity + aligned(len(node.attrs["positions"]) * 8),
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
            size = aligned(node.spec.size * 8)
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
            traffic += node.spec.size * 8 + sum(
                nodes[c][0].spec.size * 8 for c in reads[i]
            )
        g = gemm_contract(node)
        kind = (
            "none"
            if g is None
            else (
                _direct_kind(node, any(virtual[c] for c in operands))
                if schedule.direct_gemm
                else None
            )
            or "packed"
        )
        if virtual[i]:
            kind = "none"
        if g:
            flops += g.flops
        elif node.op == "einsum":
            domains = {}
            for child, labels in zip(node.inputs, node.attrs["labels"], strict=True):
                domains.update(zip(labels, child.spec.shape, strict=True))
            flops += len(node.inputs) * prod(domains.values())
        elif node.op not in VIEWS and node.op not in ("input", "constant", "gather"):
            flops += sum(child.spec.size for child in node.inputs)
        steps.append(Step(node, operands, virtual[i], offsets[i], last[i], kind))
    host = checked_size(
        sum(nodes[i][0].spec.size * 8 for i in inputs)
        + sum(nodes[i][0].spec.size * 8 for _, i in outputs)
        + (VALIDATION_BYTES if inputs else 0),
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
    selected = TensorSchedule(
        **{
            **asdict(schedule),
            **dict(zip(("tile_m", "tile_n", "tile_k"), tile, strict=True)),
        }
    )
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
    )
