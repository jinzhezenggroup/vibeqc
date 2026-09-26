"""ProgramIR-level logical sharding and explicit collective schedule contracts.

This module is compiler metadata only.  It does not initialize a GPU backend,
create communicators, choose a physical topology, or execute scientific kernels.
The same ProgramIR can be lowered against different logical meshes while its
scientific identity remains unchanged.
"""

from __future__ import annotations

import json
import math
import typing
from dataclasses import dataclass

from .program import ProgramIR
from .provenance import canonical_hash
from .resources import (
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourceRequest,
    checked_bytes,
)

COLLECTIVE_KINDS = frozenset(("all_reduce", "reduce_scatter", "all_gather"))
PLACEMENT_KINDS = frozenset(("replicated", "sharded"))
_REDUCTION_KINDS = frozenset(("sum",))


def _text(value: typing.Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _strict_keys(
    payload: typing.Any, expected: typing.Iterable[str], name: str
) -> None:
    if not isinstance(payload, dict) or set(payload) != set(expected):
        raise ValueError(f"invalid {name} fields")


def partition_bounds(extent: int, parts: int, rank: int) -> tuple[int, int]:
    """Balanced contiguous partition with deterministic low-rank remainder."""
    checked_bytes(extent, "partition extent")
    if type(parts) is not int or parts <= 0:
        raise ValueError("partition count must be a positive integer")
    if type(rank) is not int or not 0 <= rank < parts:
        raise ValueError("partition rank is outside the partition count")
    width, remainder = divmod(extent, parts)
    start = rank * width + min(rank, remainder)
    stop = start + width + int(rank < remainder)
    return start, stop


@dataclass(frozen=True)
class DeviceMesh:
    """Logical mesh only; physical device binding belongs to the runtime."""

    axes: tuple[tuple[str, int], ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported device mesh schema")
        if not isinstance(self.axes, (tuple, list)) or not self.axes:
            raise ValueError("device mesh requires axes")
        axes: list[tuple[str, int]] = []
        for row in self.axes:
            if not isinstance(row, (tuple, list)) or len(row) != 2:
                raise ValueError("device mesh axes must be (name, size) pairs")
            name, size = row
            _text(name, "mesh axis")
            if type(size) is not int or size <= 0:
                raise ValueError("mesh axis size must be a positive integer")
            checked_bytes(size, "mesh axis size")
            axes.append((name, size))
        if len({name for name, _ in axes}) != len(axes):
            raise ValueError("duplicate device mesh axis")
        checked_bytes(math.prod(size for _, size in axes), "device mesh size")
        object.__setattr__(self, "axes", tuple(axes))

    @property
    def size(self) -> int:
        return math.prod(size for _, size in self.axes)

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    def axis_size(self, name: str) -> int:
        _text(name, "mesh axis")
        for axis, size in self.axes:
            if axis == name:
                return size
        raise ValueError(f"unknown device mesh axis {name!r}")

    def coordinates(self, rank: int) -> tuple[int, ...]:
        if type(rank) is not int or not 0 <= rank < self.size:
            raise ValueError("logical shard rank is outside the device mesh")
        remainder = rank
        coordinates = [0] * len(self.axes)
        for index in range(len(self.axes) - 1, -1, -1):
            size = self.axes[index][1]
            coordinates[index] = remainder % size
            remainder //= size
        return tuple(coordinates)

    def axis_coordinate(self, rank: int, axis: str) -> int:
        names = tuple(name for name, _ in self.axes)
        if axis not in names:
            raise ValueError(f"unknown device mesh axis {axis!r}")
        return self.coordinates(rank)[names.index(axis)]

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "axes": [[name, size] for name, size in self.axes],
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_payload(cls, payload: typing.Any) -> DeviceMesh:
        _strict_keys(payload, ("axes", "schema_version"), "DeviceMesh")
        return cls(
            tuple(tuple(row) for row in payload["axes"]), payload["schema_version"]
        )


@dataclass(frozen=True)
class BufferPlacement:
    """Static local storage placement for one ProgramIR boundary buffer."""

    buffer: str
    kind: str
    mesh_axis: str | None = None
    tensor_axis: int | None = None

    def __post_init__(self) -> None:
        _text(self.buffer, "placement buffer")
        if self.kind not in PLACEMENT_KINDS:
            raise ValueError("unknown buffer placement kind")
        if self.kind == "replicated":
            if self.mesh_axis is not None or self.tensor_axis is not None:
                raise ValueError("replicated placement cannot name shard axes")
            return
        _text(self.mesh_axis, "placement mesh axis")
        if type(self.tensor_axis) is not int or self.tensor_axis < 0:
            raise ValueError("sharded placement requires a nonnegative tensor axis")

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "buffer": self.buffer,
            "kind": self.kind,
            "mesh_axis": self.mesh_axis,
            "tensor_axis": self.tensor_axis,
        }

    @classmethod
    def from_payload(cls, payload: typing.Any) -> BufferPlacement:
        _strict_keys(
            payload,
            ("buffer", "kind", "mesh_axis", "tensor_axis"),
            "BufferPlacement",
        )
        return cls(**payload)


@dataclass(frozen=True)
class CollectiveSpec:
    """One explicit synchronization over a named ProgramIR buffer."""

    name: str
    kind: str
    buffer: str
    mesh_axis: str
    after_call: str
    reduction: str | None = None
    tensor_axis: int | None = None

    def __post_init__(self) -> None:
        for field in ("name", "buffer", "mesh_axis", "after_call"):
            _text(getattr(self, field), f"collective {field}")
        if self.kind not in COLLECTIVE_KINDS:
            raise ValueError("unknown collective kind")
        if self.kind in ("all_reduce", "reduce_scatter"):
            if self.reduction not in _REDUCTION_KINDS:
                raise ValueError("reducing collective requires a supported reduction")
        elif self.reduction is not None:
            raise ValueError("all_gather does not accept a reduction")
        if self.kind == "all_reduce":
            if self.tensor_axis is not None:
                raise ValueError("all_reduce does not accept a tensor axis")
        elif type(self.tensor_axis) is not int or self.tensor_axis < 0:
            raise ValueError("scatter/gather collective requires a tensor axis")

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "buffer": self.buffer,
            "mesh_axis": self.mesh_axis,
            "after_call": self.after_call,
            "reduction": self.reduction,
            "tensor_axis": self.tensor_axis,
        }

    @classmethod
    def from_payload(cls, payload: typing.Any) -> CollectiveSpec:
        _strict_keys(
            payload,
            (
                "name",
                "kind",
                "buffer",
                "mesh_axis",
                "after_call",
                "reduction",
                "tensor_axis",
            ),
            "CollectiveSpec",
        )
        return cls(**payload)


@dataclass(frozen=True)
class CollectiveCost:
    """Architecture-neutral logical communication work, not a latency model."""

    name: str
    kind: str
    participants: int
    payload_bytes: int
    logical_work_bytes: int


@dataclass(frozen=True)
class SpmdPlan:
    """Deterministic SPMD lowering metadata for one immutable ProgramIR."""

    program: ProgramIR
    mesh: DeviceMesh
    placements: tuple[BufferPlacement, ...]
    collectives: tuple[CollectiveSpec, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.program, ProgramIR):
            raise TypeError("SPMD plan requires ProgramIR")
        if not isinstance(self.mesh, DeviceMesh):
            raise TypeError("SPMD plan requires DeviceMesh")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported SPMD plan schema")
        placements = tuple(self.placements)
        collectives = tuple(self.collectives)
        if not all(isinstance(item, BufferPlacement) for item in placements):
            raise TypeError("SPMD plan requires structured placements")
        if not all(isinstance(item, CollectiveSpec) for item in collectives):
            raise TypeError("SPMD plan requires structured collectives")
        if len({item.buffer for item in placements}) != len(placements):
            raise ValueError("duplicate SPMD buffer placement")
        if len({item.name for item in collectives}) != len(collectives):
            raise ValueError("duplicate SPMD collective name")
        declared = {buffer.name: buffer for buffer in self.program.buffers}
        if {item.buffer for item in placements} != set(declared):
            raise ValueError(
                "SPMD placements must cover every ProgramIR buffer exactly"
            )
        placement_by_buffer = {item.buffer: item for item in placements}
        for placement in placements:
            if placement.kind == "sharded":
                self.mesh.axis_size(typing.cast("str", placement.mesh_axis))
                buffer = declared[placement.buffer]
                if buffer.layout is None or buffer.itemsize is None:
                    raise ValueError("sharded ProgramIR buffer requires dense layout")
                axis = typing.cast("int", placement.tensor_axis)
                if axis >= len(buffer.layout.shape):
                    raise ValueError("sharded tensor axis is outside the buffer rank")
        call_phase = {
            call.name: phase for phase, call in enumerate(self.program.calls, 1)
        }
        produced = dict.fromkeys(self.program.inputs, 0)
        for phase, call in enumerate(self.program.calls, 1):
            produced.update({name: phase for name in call.writes})
        for collective in collectives:
            axis_size = self.mesh.axis_size(collective.mesh_axis)
            if axis_size == 1:
                raise ValueError(
                    "single-participant collective must be canonicalized away"
                )
            if collective.buffer not in declared:
                raise ValueError("collective references an unknown ProgramIR buffer")
            if collective.after_call not in call_phase:
                raise ValueError("collective references an unknown ProgramIR call")
            if produced[collective.buffer] > call_phase[collective.after_call]:
                raise ValueError(
                    "collective buffer is unavailable at its synchronization point"
                )
            placement = placement_by_buffer[collective.buffer]
            buffer = declared[collective.buffer]
            if collective.kind == "all_reduce":
                if placement.kind != "replicated":
                    raise ValueError("all_reduce requires replicated result placement")
            elif collective.kind == "reduce_scatter":
                if (
                    placement.kind != "sharded"
                    or placement.mesh_axis != collective.mesh_axis
                    or placement.tensor_axis != collective.tensor_axis
                ):
                    raise ValueError(
                        "reduce_scatter result placement must match its shard axes"
                    )
            else:
                if placement.kind != "replicated":
                    raise ValueError("all_gather requires replicated result placement")
                if buffer.layout is None or buffer.itemsize is None:
                    raise ValueError("all_gather requires a dense ProgramIR buffer")
                axis = typing.cast("int", collective.tensor_axis)
                if axis >= len(buffer.layout.shape):
                    raise ValueError(
                        "collective tensor axis is outside the buffer rank"
                    )
        object.__setattr__(self, "placements", placements)
        object.__setattr__(self, "collectives", collectives)

    @classmethod
    def lower(
        cls,
        program: ProgramIR,
        mesh: DeviceMesh,
        placements: typing.Iterable[BufferPlacement],
        collectives: typing.Iterable[CollectiveSpec] = (),
    ) -> SpmdPlan:
        """Lower one scientific program; size-one mesh is the exact fallback."""
        requested = tuple(collectives)
        effective = tuple(
            item for item in requested if mesh.axis_size(item.mesh_axis) > 1
        )
        return cls(program, mesh, tuple(placements), effective)

    @property
    def program_identity(self) -> str:
        return self.program.identity

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    @property
    def required_collectives(self) -> tuple[str, ...]:
        return tuple(sorted({item.kind for item in self.collectives}))

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "schema_version": self.schema_version,
            "program_identity": self.program.identity,
            "mesh": self.mesh.to_payload(),
            "placements": [item.to_payload() for item in self.placements],
            "collectives": [item.to_payload() for item in self.collectives],
        }

    @classmethod
    def from_payload(cls, program: ProgramIR, payload: typing.Any) -> SpmdPlan:
        _strict_keys(
            payload,
            (
                "schema_version",
                "program_identity",
                "mesh",
                "placements",
                "collectives",
            ),
            "SpmdPlan",
        )
        if payload["program_identity"] != program.identity:
            raise ValueError("SPMD plan does not match the supplied ProgramIR")
        if not isinstance(payload["placements"], (list, tuple)):
            raise TypeError("invalid SPMD placement sequence")
        if not isinstance(payload["collectives"], (list, tuple)):
            raise TypeError("invalid SPMD collective sequence")
        return cls(
            program,
            DeviceMesh.from_payload(payload["mesh"]),
            tuple(BufferPlacement.from_payload(item) for item in payload["placements"]),
            tuple(CollectiveSpec.from_payload(item) for item in payload["collectives"]),
            payload["schema_version"],
        )

    def provenance(self) -> dict[str, typing.Any]:
        return json.loads(
            json.dumps(
                {
                    "schema": "vibeqc.program_ir.spmd.v1",
                    "program_identity": self.program.identity,
                    "spmd_identity": self.identity,
                    "mesh": self.mesh.to_payload(),
                    "placements": [item.to_payload() for item in self.placements],
                    "collectives": [item.to_payload() for item in self.collectives],
                }
            )
        )

    def placement(self, buffer: str) -> BufferPlacement:
        for item in self.placements:
            if item.buffer == buffer:
                return item
        raise ValueError("unknown SPMD buffer")

    def local_shape(self, buffer: str, rank: int) -> tuple[int, ...] | None:
        placement = self.placement(buffer)
        owner = next(item for item in self.program.buffers if item.name == buffer)
        self.mesh.coordinates(rank)
        if owner.layout is None:
            return None
        shape = list(owner.layout.shape)
        if placement.kind == "sharded":
            mesh_axis = typing.cast("str", placement.mesh_axis)
            tensor_axis = typing.cast("int", placement.tensor_axis)
            shard = self.mesh.axis_coordinate(rank, mesh_axis)
            start, stop = partition_bounds(
                shape[tensor_axis],
                self.mesh.axis_size(mesh_axis),
                shard,
            )
            shape[tensor_axis] = stop - start
        return tuple(shape)

    def local_bytes(self, buffer: str, rank: int) -> int:
        placement = self.placement(buffer)
        owner = next(item for item in self.program.buffers if item.name == buffer)
        self.mesh.coordinates(rank)
        if placement.kind == "replicated":
            return owner.bytes
        shape = self.local_shape(buffer, rank)
        assert shape is not None and owner.itemsize is not None
        return checked_bytes(owner.itemsize * math.prod(shape), "local shard bytes")

    def _collective_local_bytes(
        self, collective: CollectiveSpec, rank: int
    ) -> tuple[int, int]:
        """Return source/result bytes for one rank around a collective."""
        owner = next(
            item for item in self.program.buffers if item.name == collective.buffer
        )
        self.mesh.coordinates(rank)
        if collective.kind == "all_reduce":
            return owner.bytes, owner.bytes
        axis = typing.cast("int", collective.tensor_axis)
        if collective.kind == "reduce_scatter":
            return owner.bytes, self.local_bytes(collective.buffer, rank)
        assert owner.layout is not None and owner.itemsize is not None
        shape = list(owner.layout.shape)
        shard = self.mesh.axis_coordinate(rank, collective.mesh_axis)
        start, stop = partition_bounds(
            shape[axis],
            self.mesh.axis_size(collective.mesh_axis),
            shard,
        )
        shape[axis] = stop - start
        source = checked_bytes(
            owner.itemsize * math.prod(shape), "all_gather local source bytes"
        )
        return source, owner.bytes

    def shard_identity(self, rank: int) -> str:
        coordinates = self.mesh.coordinates(rank)
        return canonical_hash(
            {
                "spmd_plan": self.identity,
                "rank": rank,
                "coordinates": coordinates,
            }
        )

    def require_collective_support(self, supported: typing.Iterable[str]) -> None:
        supported_set = set(supported)
        if any(not isinstance(item, str) for item in supported_set):
            raise TypeError("collective capabilities must be strings")
        missing = tuple(sorted(set(self.required_collectives) - supported_set))
        if missing:
            raise NotImplementedError(
                "backend lacks required collectives: " + ", ".join(missing)
            )

    def require_complete_shards(self, completed: typing.Iterable[int]) -> None:
        ranks = tuple(completed)
        if any(type(rank) is not int for rank in ranks):
            raise TypeError("completed shard ranks must be integers")
        if len(set(ranks)) != len(ranks):
            raise ValueError("duplicate completed shard rank")
        expected = set(range(self.mesh.size))
        observed = set(ranks)
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise RuntimeError(
                f"cannot publish partial SPMD result; missing={missing}, extra={extra}"
            )

    def collective_costs(self) -> tuple[CollectiveCost, ...]:
        buffers = {item.name: item for item in self.program.buffers}
        costs = []
        for collective in self.collectives:
            participants = self.mesh.axis_size(collective.mesh_axis)
            logical = checked_bytes(
                sum(
                    sum(self._collective_local_bytes(collective, rank))
                    for rank in range(self.mesh.size)
                ),
                "collective logical work bytes",
            )
            costs.append(
                CollectiveCost(
                    collective.name,
                    collective.kind,
                    participants,
                    buffers[collective.buffer].bytes,
                    logical,
                )
            )
        return tuple(costs)

    def communication_resource_request(self) -> ResourceRequest | None:
        """Account logical-device communication scratch and profitability work."""
        if not self.collectives:
            return None
        call_phase = {
            call.name: phase for phase, call in enumerate(self.program.calls, 1)
        }
        estimates = []
        for collective in self.collectives:
            phase = call_phase[collective.after_call]
            for rank in range(self.mesh.size):
                source, result = self._collective_local_bytes(collective, rank)
                estimates.append(
                    ResourceEstimate(
                        f"comm:{collective.name}:rank{rank}",
                        max(source, result),
                        f"device:{rank}",
                        phase,
                        phase,
                        streamed_bytes=checked_bytes(
                            source + result,
                            "collective logical traffic bytes",
                        ),
                    )
                )
        cost = checked_bytes(
            sum(item.logical_work_bytes for item in self.collective_costs()),
            "SPMD communication relative cost",
        )
        return ResourceRequest(
            f"{self.program.name}.spmd_communication",
            ResourceIdentity(
                "opaque_program",
                "ProgramIR.SPMD",
                "logical_mesh",
                "described",
                json.dumps(
                    {
                        "program": self.program.identity,
                        "spmd": self.identity,
                        "mesh": self.mesh.identity,
                    }
                ),
                self.program.outputs,
                f"spmd:{self.identity}",
            ),
            (
                ResourceCandidate(
                    "explicit_collectives",
                    "streamed",
                    tuple(estimates),
                    relative_cost=cost,
                    decisions=(
                        ("spmd_identity", self.identity),
                        ("communication_cost", str(cost)),
                    ),
                ),
            ),
            (
                "Backend collective-library internal allocations",
                "Physical interconnect topology, latency and bandwidth",
                "Concurrent communication not declared by this SPMD plan",
            ),
        )


def reference_collective(
    kind: str,
    shard_values: typing.Iterable[typing.Iterable[typing.Any]],
) -> tuple[tuple[typing.Any, ...], ...]:
    """Pure CPU reference semantics for the bounded v1 collective set."""
    if kind not in COLLECTIVE_KINDS:
        raise ValueError("unknown collective kind")
    rows = tuple(tuple(row) for row in shard_values)
    if not rows:
        raise ValueError("collective requires at least one shard")
    if kind == "all_gather":
        gathered = tuple(item for row in rows for item in row)
        return tuple(gathered for _ in rows)
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("reducing collectives require equal local extents")
    reduced = tuple(sum(row[index] for row in rows) for index in range(width))
    if kind == "all_reduce":
        return tuple(reduced for _ in rows)
    return tuple(
        reduced[slice(*partition_bounds(width, len(rows), rank))]
        for rank in range(len(rows))
    )
