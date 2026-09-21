"""Backend-neutral alias, lifetime, interference, and buffer assignment analysis.

The analysis owns compiler-visible storage facts only. Runtime allocation, streams,
scientific semantics, and opaque library internals remain with their existing owners.
Unknown alias/effect semantics deliberately disable reuse rather than guessing.
"""

from __future__ import annotations

import typing
from collections.abc import Hashable
from dataclasses import dataclass
from enum import Enum

from .layout import DenseLayout
from .resources import checked_bytes


class AliasKind(str, Enum):
    """Relationship between a logical value and physical storage."""

    OWNER = "owner"
    VIEW = "view"
    UNKNOWN = "unknown"


class MemoryEffect(str, Enum):
    """Whether an operation's memory behavior is fully declared."""

    EXPLICIT = "explicit"
    OPAQUE = "opaque"


def _sequence(values: typing.Any, name: str) -> tuple[Hashable, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(values)
    if any(not isinstance(value, Hashable) for value in result):
        raise TypeError(f"{name} values must be hashable")
    return result


@dataclass(frozen=True)
class BufferValue:
    """One logical value and its compiler-visible storage contract."""

    key: Hashable
    bytes: int
    space: str
    layout: DenseLayout | None = None
    alias: AliasKind = AliasKind.OWNER
    alias_of: Hashable | None = None
    compiler_owned: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.key, Hashable):
            raise TypeError("buffer key must be hashable")
        checked_bytes(self.bytes, "buffer bytes")
        if not isinstance(self.space, str) or not self.space:
            raise ValueError("buffer space must be a nonempty string")
        if self.layout is not None and not isinstance(self.layout, DenseLayout):
            raise TypeError("buffer layout must be DenseLayout")
        if not isinstance(self.alias, AliasKind):
            raise TypeError("buffer alias must be AliasKind")
        if type(self.compiler_owned) is not bool:
            raise TypeError("compiler_owned must be bool")
        if self.alias is AliasKind.VIEW:
            if not isinstance(self.alias_of, Hashable) or self.alias_of == self.key:
                raise ValueError("view buffer requires a distinct alias owner")
            if self.compiler_owned:
                raise ValueError("view buffer cannot own storage")
        elif self.alias_of is not None:
            raise ValueError("only view buffers may name an alias owner")
        if self.alias is AliasKind.UNKNOWN and self.compiler_owned:
            raise ValueError("unknown-alias storage cannot be compiler-owned")


@dataclass(frozen=True)
class BufferOp:
    """One ordered operation with explicit logical reads and writes."""

    key: Hashable
    reads: tuple[Hashable, ...]
    writes: tuple[Hashable, ...]
    effect: MemoryEffect = MemoryEffect.OPAQUE

    def __post_init__(self) -> None:
        if not isinstance(self.key, Hashable):
            raise TypeError("buffer operation key must be hashable")
        object.__setattr__(self, "reads", _sequence(self.reads, "reads"))
        object.__setattr__(self, "writes", _sequence(self.writes, "writes"))
        if len(set(self.writes)) != len(self.writes):
            raise ValueError("buffer operation contains duplicate writes")
        if not isinstance(self.effect, MemoryEffect):
            raise TypeError("buffer operation effect must be MemoryEffect")


@dataclass(frozen=True)
class AllocationRange:
    """Inclusive lifetime of one physical ownership group."""

    owner: Hashable
    members: tuple[Hashable, ...]
    bytes: int
    space: str
    first_phase: int
    last_phase: int
    compiler_owned: bool
    reusable: bool
    retained_reason: str | None = None


@dataclass(frozen=True)
class BufferSlot:
    """One deterministic reusable storage slot."""

    index: int
    space: str
    bytes: int
    alignment: int
    owners: tuple[Hashable, ...]


@dataclass(frozen=True)
class StorageAnalysis:
    """Deterministic whole-region storage diagnostics."""

    ranges: tuple[AllocationRange, ...]
    assignments: tuple[tuple[Hashable, int], ...]
    slots: tuple[BufferSlot, ...]
    interference: tuple[tuple[Hashable, Hashable], ...]
    peak_live_bytes: tuple[tuple[str, int], ...]
    blocked_spaces: tuple[str, ...]

    def slot_for(self, owner: Hashable) -> int | None:
        return dict(self.assignments).get(owner)

    @property
    def peak_by_space(self) -> dict[str, int]:
        return dict(self.peak_live_bytes)


def _overlap(left: AllocationRange, right: AllocationRange) -> bool:
    return not (
        left.last_phase < right.first_phase or right.last_phase < left.first_phase
    )


def analyze_storage(
    values: typing.Any,
    operations: typing.Any,
    *,
    inputs: typing.Any,
    outputs: typing.Any,
) -> StorageAnalysis:
    """Analyze aliases/lifetimes and greedily assign reusable owned storage."""

    if not isinstance(values, (tuple, list)) or not values:
        raise ValueError("storage analysis requires buffer values")
    values = tuple(values)
    if any(not isinstance(value, BufferValue) for value in values):
        raise TypeError("storage analysis requires BufferValue records")
    if len({value.key for value in values}) != len(values):
        raise ValueError("duplicate buffer value key")
    by_key = {value.key: value for value in values}
    order = {value.key: i for i, value in enumerate(values)}

    operations = tuple(operations)
    if any(not isinstance(op, BufferOp) for op in operations):
        raise TypeError("storage analysis requires BufferOp records")
    if len({op.key for op in operations}) != len(operations):
        raise ValueError("duplicate buffer operation key")

    roots: dict[Hashable, Hashable] = {}

    def root(key: Hashable, stack: tuple[Hashable, ...] = ()) -> Hashable:
        if key in roots:
            return roots[key]
        value = by_key[key]
        if value.alias is not AliasKind.VIEW:
            roots[key] = key
            return key
        if value.alias_of not in by_key:
            raise ValueError("view alias owner is undeclared")
        if value.alias_of in stack:
            raise ValueError("cyclic buffer alias")
        owner = root(value.alias_of, (*stack, key))
        if by_key[owner].space != value.space:
            raise ValueError("view alias crosses memory spaces")
        roots[key] = owner
        return owner

    for value in values:
        owner = root(value.key)
        if value.alias is AliasKind.VIEW and value.bytes > by_key[owner].bytes:
            raise ValueError("view buffer exceeds alias owner capacity")

    inputs = _sequence(inputs, "inputs")
    outputs = _sequence(outputs, "outputs")
    if len(set(inputs)) != len(inputs) or len(set(outputs)) != len(outputs):
        raise ValueError("duplicate storage input/output")
    if not set(inputs) <= set(by_key):
        raise ValueError("undeclared storage input")

    end = len(operations) + 1
    available = set(inputs)
    first = {key: 0 for key in inputs}
    last = {key: end for key in inputs}
    opaque_roots: set[Hashable] = set()
    for phase, op in enumerate(operations, 1):
        missing = tuple(key for key in op.reads if key not in available)
        if missing:
            raise ValueError(
                f"{op.key!r}: missing, forward, or cyclic storage dependency"
            )
        undeclared = tuple(key for key in op.writes if key not in by_key)
        if undeclared:
            raise ValueError(f"{op.key!r}: undeclared storage output")
        duplicate = tuple(key for key in op.writes if key in available)
        if duplicate:
            raise ValueError(f"{op.key!r}: duplicate storage producer")
        for key in op.reads:
            last[key] = max(last.get(key, phase), phase)
        for key in op.writes:
            first[key] = last[key] = phase
        if op.effect is MemoryEffect.OPAQUE:
            opaque_roots.update(root(key) for key in (*op.reads, *op.writes))
        available.update(op.writes)

    if available != set(by_key):
        raise ValueError("declared storage value has no producer")
    if not set(outputs) <= available:
        raise ValueError("storage output has no producer")
    for key in outputs:
        last[key] = end

    members: dict[Hashable, list[Hashable]] = {}
    for value in values:
        members.setdefault(root(value.key), []).append(value.key)

    blocked_spaces = {
        value.space for value in values if value.alias is AliasKind.UNKNOWN
    }
    ranges = []
    for owner, aliases in members.items():
        base = by_key[owner]
        first_phase = min(first[key] for key in aliases)
        last_phase = max(last[key] for key in aliases)
        reason = None
        reusable = base.compiler_owned
        if base.space in blocked_spaces:
            last_phase = end
            reusable = False
            reason = "unknown_alias_space"
        if owner in opaque_roots:
            last_phase = end
            reusable = False
            reason = "opaque_effect"
        if base.alias is AliasKind.UNKNOWN:
            last_phase = end
            reusable = False
            reason = "unknown_alias"
        if not base.compiler_owned and reason is None:
            reusable = False
            reason = "external_owner"
        ranges.append(
            AllocationRange(
                owner,
                tuple(aliases),
                base.bytes,
                base.space,
                first_phase,
                last_phase,
                base.compiler_owned,
                reusable,
                reason,
            )
        )
    ranges.sort(key=lambda item: order[item.owner])

    interference = []
    for i, left in enumerate(ranges):
        for right in ranges[i + 1 :]:
            if left.space == right.space and _overlap(left, right):
                interference.append((left.owner, right.owner))
    peaks: dict[str, int] = {}
    for phase in range(end + 1):
        spaces: dict[str, int] = {}
        for item in ranges:
            if item.first_phase <= phase <= item.last_phase:
                spaces[item.space] = spaces.get(item.space, 0) + item.bytes
        for space, count in spaces.items():
            peaks[space] = max(peaks.get(space, 0), count)

    slot_state: list[dict[str, typing.Any]] = []
    assignments = []
    allocation_order = sorted(
        (item for item in ranges if item.compiler_owned),
        key=lambda item: (item.first_phase, order[item.owner]),
    )
    for item in allocation_order:
        # Views share a physical base and may impose stricter alignment,
        # including through a chain of aliases.
        alignment = max(
            1 if by_key[member].layout is None else by_key[member].layout.alignment
            for member in item.members
        )
        candidates = []
        if item.reusable:
            for index, slot in enumerate(slot_state):
                if (
                    slot["space"] == item.space
                    and slot["last"] < item.first_phase
                    and slot["bytes"] >= item.bytes
                ):
                    candidates.append((slot["bytes"], index))
        if candidates:
            _, index = min(candidates)
            slot = slot_state[index]
            slot["alignment"] = max(slot["alignment"], alignment)
            slot["last"] = item.last_phase
            slot["owners"].append(item.owner)
        else:
            index = len(slot_state)
            slot_state.append(
                {
                    "space": item.space,
                    "bytes": item.bytes,
                    "alignment": alignment,
                    "last": item.last_phase if item.reusable else end,
                    "owners": [item.owner],
                }
            )
        assignments.append((item.owner, index))

    slots = tuple(
        BufferSlot(
            index,
            slot["space"],
            slot["bytes"],
            slot["alignment"],
            tuple(slot["owners"]),
        )
        for index, slot in enumerate(slot_state)
    )
    return StorageAnalysis(
        tuple(ranges),
        tuple(assignments),
        slots,
        tuple(interference),
        tuple(sorted(peaks.items())),
        tuple(sorted(blocked_spaces)),
    )
