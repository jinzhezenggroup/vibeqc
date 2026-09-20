"""Minimal serial ProgramIR: opaque provider calls and boundary-buffer lifetimes.

This is not an allocator, tracer, scientific IR or asynchronous scheduler. Calls
complete before the next call starts. Buffers are disjoint allocation owners;
views, in-place writes, hidden retention and asynchronous leases are unsupported.
Opaque provider interiors remain outside this boundary-only accounting scope.
"""

import json
import typing
from dataclasses import asdict, dataclass

from .layout import DenseLayout
from .provenance import canonical_hash
from .resources import (
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourceRequest,
    byte_product,
    checked_bytes,
)


def _text(value: typing.Any, name: typing.Any) -> typing.Any:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _names(
    values: typing.Any, name: typing.Any, *, unique: typing.Any = True
) -> typing.Any:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be a sequence of names")
    values = tuple(_text(value, name) for value in values)
    if unique and len(set(values)) != len(values):
        raise ValueError(f"duplicate {name}")
    return values


def _keys(payload: typing.Any, expected: typing.Any, name: typing.Any) -> None:
    if not isinstance(payload, dict) or set(payload) != set(expected):
        raise ValueError(f"invalid {name} fields")


@dataclass(frozen=True)
class ProgramBuffer:
    """One disjoint boundary ownership group, with its numeric capacity in bytes."""

    name: str
    bytes: int
    space: str = "pageable"
    layout: DenseLayout | None = None
    itemsize: int | None = None

    def __post_init__(self) -> None:
        # Reuse #203's checked byte/space ABI instead of another resource model.
        _text(self.space, "buffer space")
        ResourceEstimate(self.name, self.bytes, self.space, 0, 0)
        if self.layout is None:
            if self.itemsize is not None:
                raise ValueError("itemsize requires an explicit dense layout")
            return
        if not isinstance(self.layout, DenseLayout):
            raise TypeError("buffer layout must be DenseLayout")
        if self.itemsize is None:
            raise ValueError("dense layout requires itemsize")
        checked_bytes(self.itemsize, "buffer itemsize")
        if not self.itemsize:
            raise ValueError("buffer itemsize must be positive")
        expected = byte_product(self.itemsize, *self.layout.shape)
        if self.bytes != expected:
            raise ValueError("buffer bytes do not match dense layout capacity")


@dataclass(frozen=True)
class PlanCall:
    """An opaque, synchronous provider invocation in a selected serial schedule.

    The provider identity is a binding supplied by the owning subsystem, not a
    proof of executable availability, convergence or current numerical state.
    Every written buffer is newly owned, disjoint from all reads and writes.
    """

    name: str
    provider: str
    identity: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in ("name", "provider", "identity"):
            _text(getattr(self, field), field)
        object.__setattr__(self, "reads", _names(self.reads, "reads", unique=False))
        object.__setattr__(self, "writes", _names(self.writes, "writes"))
        if not self.writes:
            raise ValueError("a serial provider call must declare its outputs")


@dataclass(frozen=True)
class ProgramIR:
    """A finite SSA dependency graph with an explicit, already-selected order.

    Borrowed inputs remain charged throughout the region, even after last read:
    this prototype does not implement donation. Exported outputs survive until
    the region's publication boundary. A release applies *after* its provider
    completes; its inputs and newly allocated outputs overlap during that call.
    No operation is pruned or reordered merely because an output is unused.
    """

    name: str
    buffers: tuple[ProgramBuffer, ...]
    inputs: tuple[str, ...]
    calls: tuple[PlanCall, ...]
    outputs: tuple[str, ...]
    schema_version: int = 2

    def __post_init__(self) -> None:
        _text(self.name, "program name")
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("unsupported ProgramIR schema")
        for field, item_type in (("buffers", ProgramBuffer), ("calls", PlanCall)):
            items = getattr(self, field)
            if not isinstance(items, (tuple, list)) or not items:
                raise ValueError(f"program requires {field}")
            if not all(isinstance(item, item_type) for item in items):
                raise TypeError(f"program requires structured {field}")
            if len({item.name for item in items}) != len(items):
                raise ValueError(f"duplicate {field} name")
            object.__setattr__(self, field, tuple(items))
        for field in ("inputs", "outputs"):
            object.__setattr__(self, field, _names(getattr(self, field), field))
        if not self.outputs:
            raise ValueError("program requires exported outputs")
        declared = {buffer.name for buffer in self.buffers}
        available = set(self.inputs)
        if not available <= declared:
            raise ValueError("undeclared input buffer")
        for call in self.calls:
            if not set(call.reads) <= available:
                raise ValueError(f"{call.name}: missing, forward or cyclic dependency")
            if not set(call.writes) <= declared:
                raise ValueError(f"{call.name}: undeclared output buffer")
            if set(call.writes) & available:
                raise ValueError(f"{call.name}: duplicate owner or in-place write")
            available.update(call.writes)
        if available != declared:
            raise ValueError("declared buffer has no producer")
        if not set(self.outputs) <= available:
            raise ValueError("missing exported output")

    def to_payload(self) -> typing.Any:
        """Return detached JSON-compatible data; never executable Python."""
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_payload(cls, payload: typing.Any) -> typing.Any:
        """Strict replay validates dependencies anew; no saved analysis is trusted."""
        _keys(payload, cls.__dataclass_fields__, "ProgramIR")
        data = dict(payload)
        for field, item_type in (("buffers", ProgramBuffer), ("calls", PlanCall)):
            if not isinstance(data[field], (list, tuple)):
                raise TypeError(f"invalid {field} sequence")
            converted = []
            for item in data[field]:
                _keys(item, item_type.__dataclass_fields__, field)
                item_data = dict(item)
                if item_type is ProgramBuffer and item_data["layout"] is not None:
                    _keys(
                        item_data["layout"],
                        DenseLayout.__dataclass_fields__,
                        "buffer layout",
                    )
                    item_data["layout"] = DenseLayout(**item_data["layout"])
                converted.append(item_type(**item_data))
            data[field] = tuple(converted)
        return cls(**data)

    @property
    def identity(self) -> typing.Any:
        return canonical_hash(self.to_payload())

    def lifetimes(self, *, retain_temporaries: typing.Any = False) -> typing.Any:
        """Derive inclusive #203 intervals, without assuming buffer alias/reuse.

        ``retain_temporaries`` is a diagnostic retain-all comparison, not a
        description of an existing production allocator or a speedup baseline.
        """
        if type(retain_temporaries) is not bool:
            raise ValueError("retain_temporaries must be bool")
        end = len(self.calls) + 1
        first = dict.fromkeys(self.inputs, 0)
        last = dict.fromkeys(self.inputs, end)
        for phase, call in enumerate(self.calls, 1):
            for name in call.reads:
                last[name] = max(last[name], phase)
            for name in call.writes:
                first[name] = last[name] = phase
        for name in self.outputs:
            last[name] = end
        return tuple(
            ResourceEstimate(
                buffer.name,
                buffer.bytes,
                buffer.space,
                first[buffer.name],
                end if retain_temporaries else last[buffer.name],
                kind=(
                    "persistent"
                    if buffer.name in self.inputs
                    else "output"
                    if buffer.name in self.outputs
                    else "workspace"
                ),
            )
            for buffer in self.buffers
        )

    def release_after(self, call_name: typing.Any) -> typing.Any:
        """Owned non-exported buffers whose final use completes at this call."""
        names = tuple(call.name for call in self.calls)
        if call_name not in names:
            raise ValueError("unknown program call")
        phase = names.index(call_name) + 1
        return tuple(
            estimate.name
            for estimate in self.lifetimes()
            if estimate.kind == "workspace" and estimate.last_phase == phase
        )

    def resource_request(self, *, retain_temporaries: typing.Any = False) -> typing.Any:
        """Feed the existing planner; this does not change any native allocation."""
        estimates = self.lifetimes(retain_temporaries=retain_temporaries)
        return ResourceRequest(
            self.name,
            ResourceIdentity(
                "opaque_program",
                "ProgramIR",
                "described",
                "described",
                json.dumps({"program": self.identity}),
                self.outputs,
                "serial_retain_all" if retain_temporaries else "serial_last_use",
            ),
            (ResourceCandidate("boundary", "streamed", estimates),),
            (
                "Opaque provider-internal scratch, copies and library workspaces",
                "Allocator rounding, Python objects and native runtime overhead",
                "Caller-retained output history and asynchronous execution",
            ),
        )
